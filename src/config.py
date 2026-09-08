from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT_DIR / "data" / "papers"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "auto").strip().lower()

TOP_K = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "220"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.22"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "420"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

ENABLE_SEMANTIC = os.getenv("ENABLE_SEMANTIC", "true").lower() in {
    "1", "true", "yes", "on"
}
ALLOW_MODEL_DOWNLOAD = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() in {
    "1", "true", "yes", "on"
}

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
