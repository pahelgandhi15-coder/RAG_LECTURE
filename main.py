"""
FastAPI backend for the Lecture RAG app.

Endpoints:
  POST /api/process   { url }                 -> { video_id, cached }
  GET  /api/status/{video_id}                  -> { stage, pct, done, error }
  POST /api/ask        { video_id, question }  -> { answer, context_chunks }

Processing (download/transcribe/embed) runs in a background thread so the
server stays responsive while the frontend polls /api/status.
"""
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import ingest, qa
from rag.config import DATA_DIR

app = FastAPI(title="Lecture RAG")

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


class ProcessRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    video_id: str
    question: str
    k: int | None = None


def _set_job(vid: str, **kwargs) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(vid, {"stage": "queued", "pct": 0, "done": False, "error": None})
        JOBS[vid].update(kwargs)


def _run_pipeline(url: str, vid: str) -> None:
    try:
        def cb(stage: str, pct: int):
            _set_job(vid, stage=stage, pct=pct)

        ingest.process_video(url, progress_cb=cb)
        _set_job(vid, stage="done", pct=100, done=True)
    except Exception as e:
        traceback.print_exc()
        _set_job(vid, stage="error", error=str(e), done=True)


@app.post("/api/process")
def process(req: ProcessRequest):
    vid = ingest.video_id_for(req.url)
    paths_chroma = DATA_DIR / vid / "chroma"

    already_cached = paths_chroma.exists() and any(paths_chroma.iterdir())
    if already_cached:
        _set_job(vid, stage="done", pct=100, done=True, error=None)
        return {"video_id": vid, "cached": True}

    with JOBS_LOCK:
        existing = JOBS.get(vid)
    if existing and not existing["done"]:
        return {"video_id": vid, "cached": False}  # already in progress

    _set_job(vid, stage="queued", pct=0, done=False, error=None)
    thread = threading.Thread(target=_run_pipeline, args=(req.url, vid), daemon=True)
    thread.start()
    return {"video_id": vid, "cached": False}


@app.get("/api/status/{video_id}")
def status(video_id: str):
    with JOBS_LOCK:
        job = JOBS.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown video_id")
    return job


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        db = ingest.load_vectorstore(req.video_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = qa.answer_question(db, req.question, k=req.k)
    return result


# ── Serve the frontend ───────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
