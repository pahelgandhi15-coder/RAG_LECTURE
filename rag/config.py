"""Central configuration. Everything is overridable via .env — nothing is hardcoded."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Silence chromadb's anonymous telemetry (noisy, harmless warnings otherwise)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")          # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 is fast+light on CPU

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# If YouTube asks yt-dlp to "sign in", set one of these so yt-dlp reuses cookies
# from a browser where you're already logged into YouTube.
#   YTDLP_COOKIES_FROM_BROWSER=chrome   (or: firefox, edge, brave, opera, vivaldi)
#   YTDLP_COOKIES_FILE=path/to/cookies.txt   (exported via a "Get cookies.txt" extension)
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip() or None
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "").strip() or None

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your Gemini API key."
    )
