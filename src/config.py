from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

PAPERS_DIR = ROOT_DIR / "data" / "papers"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"

METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
INDEX_CONFIG_PATH = VECTORSTORE_DIR / "index_config.json"


def _setting(name: str, default: str) -> str:
    value = os.getenv(name)

    if value is not None and str(value).strip():
        return str(value)

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default


def _bool(name: str, default: str = "false") -> bool:
    return _setting(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


EMBEDDING_MODEL = _setting(
    "EMBEDDING_MODEL",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
)

RERANKER_MODEL = _setting(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)

GENERATION_MODEL = _setting(
    "GENERATION_MODEL",
    "google/flan-t5-base",
)

GENERATION_BACKEND = _setting(
    "GENERATION_BACKEND",
    "extractive",
).strip().lower()


TOP_K = int(
    _setting("TOP_K", "6")
)

CHUNK_SIZE = int(
    _setting("CHUNK_SIZE", "1400")
)

CHUNK_OVERLAP = int(
    _setting("CHUNK_OVERLAP", "250")
)


LEXICAL_CANDIDATES = int(
    _setting("LEXICAL_CANDIDATES", "80")
)

SEMANTIC_CANDIDATES = int(
    _setting("SEMANTIC_CANDIDATES", "80")
)

RERANK_CANDIDATES = int(
    _setting("RERANK_CANDIDATES", "60")
)


RELEVANCE_THRESHOLD = float(
    _setting("RELEVANCE_THRESHOLD", "0.20")
)

ANSWER_WINDOW_COUNT = int(
    _setting("ANSWER_WINDOW_COUNT", "3")
)

ANSWER_WINDOW_THRESHOLD = float(
    _setting("ANSWER_WINDOW_THRESHOLD", "0.12")
)


MAX_NEW_TOKENS = int(
    _setting("MAX_NEW_TOKENS", "420")
)

MAX_CONTEXT_CHARS = int(
    _setting("MAX_CONTEXT_CHARS", "12000")
)


ENABLE_SEMANTIC = _bool(
    "ENABLE_SEMANTIC",
    "true",
)

ENABLE_RERANKER = _bool(
    "ENABLE_RERANKER",
    "true",
)

ALLOW_MODEL_DOWNLOAD = _bool(
    "ALLOW_MODEL_DOWNLOAD",
    "true",
)


PAPERS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)