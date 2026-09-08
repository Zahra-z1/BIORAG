from __future__ import annotations

import json
import numpy as np

from .chunking import create_chunks
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    PAPERS_DIR,
    VECTORSTORE_DIR,
)
from .ingestion import load_documents


def build_index(build_semantic: bool = True):
    docs = load_documents(PAPERS_DIR)
    if not docs:
        raise RuntimeError("No readable PDF text found. Put PDFs in data/papers/.")

    chunks = create_chunks(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote lexical metadata: {len(chunks)} chunks from {len(docs)} readable PDF pages.")

    if not build_semantic:
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        print("Skipped semantic FAISS index (--lexical-only).")
        return {"chunks": len(chunks), "semantic": False}

    try:
        import faiss
        from .embeddings import EmbeddingModel

        vectors = np.asarray(
            EmbeddingModel().encode([x["text"] for x in chunks]), dtype="float32"
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        print(f"Wrote semantic FAISS index with {index.ntotal} vectors.")
        return {"chunks": len(chunks), "semantic": True}
    except Exception as exc:
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        print(f"Semantic index was not built: {exc}")
        print("The app will still work with lexical retrieval. Re-run later to add semantic retrieval.")
        return {"chunks": len(chunks), "semantic": False, "error": str(exc)}
