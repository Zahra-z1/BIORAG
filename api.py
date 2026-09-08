from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent import ask
from src.config import FAISS_INDEX_PATH, METADATA_PATH, PAPERS_DIR, TOP_K

app = FastAPI(title="BIO RAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=12)


@app.get("/health")
def health():
    chunks = 0
    if METADATA_PATH.exists():
        try:
            chunks = len(json.loads(METADATA_PATH.read_text(encoding="utf-8")))
        except Exception:
            chunks = 0
    return {
        "status": "ok" if chunks else "index_missing",
        "pdf_count": len(list(PAPERS_DIR.glob("*.pdf"))),
        "chunk_count": chunks,
        "semantic_index": FAISS_INDEX_PATH.exists(),
    }


@app.post("/ask")
def ask_endpoint(payload: AskRequest):
    try:
        return ask(payload.question, payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BIO RAG backend error: {exc}") from exc
