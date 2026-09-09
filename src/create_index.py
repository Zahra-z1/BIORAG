from __future__ import annotations

import json

import numpy as np

from .chunking import create_chunks

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    INDEX_CONFIG_PATH,
    METADATA_PATH,
    PAPERS_DIR,
    VECTORSTORE_DIR,
)

from .ingestion import (
    load_documents,
)


def build_index(
    build_semantic: bool = True,
):
    docs = load_documents(
        PAPERS_DIR
    )

    if not docs:
        raise RuntimeError(
            "No readable PDF text found. "
            "Put PDFs in data/papers/."
        )

    chunks = create_chunks(
        docs,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_PATH.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    index_config = {
        "embedding_model":
            EMBEDDING_MODEL,

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            CHUNK_OVERLAP,

        "pdf_pages":
            len(docs),

        "chunks":
            len(chunks),
    }

    INDEX_CONFIG_PATH.write_text(
        json.dumps(
            index_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote metadata: "
        f"{len(chunks)} chunks from "
        f"{len(docs)} PDF pages."
    )

    if not build_semantic:

        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()

        print(
            "Skipped semantic index."
        )

        return {
            "chunks": len(chunks),
            "semantic": False,
        }

    try:
        import faiss

        from .embeddings import (
            EmbeddingModel,
        )

        embedder = EmbeddingModel()

        vectors = (
            embedder.encode_documents(
                [
                    chunk["text"]
                    for chunk in chunks
                ]
            )
        )

        vectors = np.asarray(
            vectors,
            dtype="float32",
        )

        index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        index.add(
            vectors
        )

        faiss.write_index(
            index,
            str(
                FAISS_INDEX_PATH
            ),
        )

        print(
            f"Wrote FAISS index "
            f"with {index.ntotal} vectors."
        )

        return {
            "chunks": len(chunks),
            "semantic": True,
        }

    except Exception as exc:

        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()

        print(
            f"Semantic index failed: {exc}"
        )

        return {
            "chunks": len(chunks),
            "semantic": False,
            "error": str(exc),
        }