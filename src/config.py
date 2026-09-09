from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str) -> str:
    # Streamlit Cloud secrets have priority.
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return os.getenv(name, default)


def _bool(name: str, default: str = "false") -> bool:
    return _get(
        name,
        default,
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


ROOT_DIR = Path(__file__).resolve().parents[1]

PAPERS_DIR = (
    ROOT_DIR
    / "data"
    / "papers"
)

VECTORSTORE_DIR = (
    ROOT_DIR
    / "vectorstore"
)

METADATA_PATH = (
    VECTORSTORE_DIR
    / "metadata.json"
)

FAISS_INDEX_PATH = (
    VECTORSTORE_DIR
    / "index.faiss"
)


EMBEDDING_MODEL = _get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GENERATION_MODEL = _get(
    "GENERATION_MODEL",
    "google/flan-t5-base",
)

GENERATION_BACKEND = _get(
    "GENERATION_BACKEND",
    "extractive",
).strip().lower()

RERANKER_MODEL = _get(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


TOP_K = int(
    _get(
        "TOP_K",
        "6",
    )
)

CHUNK_SIZE = int(
    _get(
        "CHUNK_SIZE",
        "1200",
    )
)

CHUNK_OVERLAP = int(
    _get(
        "CHUNK_OVERLAP",
        "220",
    )
)

RELEVANCE_THRESHOLD = float(
    _get(
        "RELEVANCE_THRESHOLD",
        "0.22",
    )
)

MAX_NEW_TOKENS = int(
    _get(
        "MAX_NEW_TOKENS",
        "420",
    )
)

MAX_CONTEXT_CHARS = int(
    _get(
        "MAX_CONTEXT_CHARS",
        "12000",
    )
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