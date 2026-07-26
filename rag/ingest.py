"""
Ingestion pipeline for a single lecture video.

Efficiency choices vs. the original notebook:
  - faster-whisper (CTranslate2) instead of openai-whisper: several times faster
    on the same hardware, lower memory, int8 quantization option.
  - Whisper model and the embedding model are loaded ONCE per process (module-level
    singletons) instead of being reloaded on every call.
  - Every expensive step (download / transcribe / embed) is cached to disk, keyed
    by a hash of the URL, so re-asking questions about a video you already
    processed costs nothing extra.
"""
import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import yt_dlp
from faster_whisper import WhisperModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from . import config

ProgressCB = Optional[Callable[[str, int], None]]


def _noop(stage: str, pct: int) -> None:
    pass


def video_id_for(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def _paths(vid: str):
    d = config.DATA_DIR / vid
    d.mkdir(parents=True, exist_ok=True)
    return {
        "dir": d,
        "audio": d / "audio.wav",
        "transcript": d / "transcript.txt",
        "chroma": d / "chroma",
        "meta": d / "meta.txt",
    }


# ── Singletons (loaded once per process) ────────────────────────────────────
_whisper_model = None
_embedding_model = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embedding_model


# ── Pipeline steps ───────────────────────────────────────────────────────────
def download_audio(url: str, audio_path: Path) -> None:
    if audio_path.exists():
        return
    tmp_template = str(audio_path.with_suffix("")) + ".raw.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best[acodec!=none]/best",
        "outtmpl": tmp_template,
        "quiet": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }
    if config.YTDLP_COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = (config.YTDLP_COOKIES_FROM_BROWSER,)
    if config.YTDLP_COOKIES_FILE:
        ydl_opts["cookiefile"] = config.YTDLP_COOKIES_FILE
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # yt-dlp + ffmpeg postprocessor writes "<stem>.raw.wav" — normalize to audio.wav
    candidate = audio_path.with_name(audio_path.stem + ".raw.wav")
    if candidate.exists():
        candidate.rename(audio_path)
    if not audio_path.exists():
        raise RuntimeError("Audio download/conversion failed — check ffmpeg is installed.")


def transcribe_audio(audio_path: Path, transcript_path: Path, progress_cb: ProgressCB = None) -> str:
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8")

    model = get_whisper_model()
    segments, info = model.transcribe(str(audio_path), language="en", beam_size=1)

    parts = []
    total_duration = max(info.duration, 1.0)
    for seg in segments:
        parts.append(seg.text.strip())
        if progress_cb:
            pct = min(99, int(seg.end / total_duration * 100))
            progress_cb("transcribing", pct)

    text = " ".join(parts).strip()
    transcript_path.write_text(text, encoding="utf-8")
    return text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_or_load_vectorstore(vid: str, chroma_path: Path, chunks: list[str]) -> Chroma:
    embedding = get_embedding_model()
    if chroma_path.exists() and any(chroma_path.iterdir()):
        return Chroma(persist_directory=str(chroma_path), embedding_function=embedding)
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=str(chroma_path),
    )
    return db


def process_video(url: str, progress_cb: ProgressCB = None) -> str:
    """
    Runs the full pipeline for a URL, reusing any cached artifacts.
    Returns the video_id, which is used to look up the vectorstore for Q&A later.
    """
    cb = progress_cb or _noop
    vid = video_id_for(url)
    paths = _paths(vid)

    cb("downloading", 5)
    download_audio(url, paths["audio"])

    cb("transcribing", 20)
    text = transcribe_audio(paths["audio"], paths["transcript"], progress_cb=cb)

    cb("chunking", 85)
    chunks = chunk_text(text)

    cb("embedding", 90)
    build_or_load_vectorstore(vid, paths["chroma"], chunks)

    paths["meta"].write_text(url, encoding="utf-8")
    cb("done", 100)
    return vid


def load_vectorstore(vid: str) -> Chroma:
    paths = _paths(vid)
    if not (paths["chroma"].exists() and any(paths["chroma"].iterdir())):
        raise FileNotFoundError(f"No processed video found for id {vid}. Process it first.")
    return Chroma(persist_directory=str(paths["chroma"]), embedding_function=get_embedding_model())
