"""
Persistent chat history, per video.

Uses plain sqlite3 (stdlib, no new dependency) — one row per question/answer
turn, so returning to a video you've already asked questions about shows the
prior conversation instead of starting blank.
"""
import sqlite3
from datetime import datetime, timezone

from . import config

DB_PATH = config.DATA_DIR / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_history_video_id ON qa_history(video_id)"
        )


def save_turn(video_id: str, question: str, answer: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO qa_history (video_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (video_id, question, answer, datetime.now(timezone.utc).isoformat()),
        )


def get_history(video_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT question, answer, created_at FROM qa_history WHERE video_id = ? ORDER BY id ASC",
            (video_id,),
        ).fetchall()
    return [dict(row) for row in rows]
