from __future__ import annotations

import numpy as np

from .config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(
        self,
        model_name=EMBEDDING_MODEL,
    ):
        from sentence_transformers import (
            SentenceTransformer,
        )

        self.model = SentenceTransformer(
            model_name
        )

    def encode_documents(
        self,
        texts,
    ):
        texts = list(texts)

        if hasattr(
            self.model,
            "encode_document",
        ):
            vectors = (
                self.model.encode_document(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        else:
            vectors = (
                self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        return np.asarray(
            vectors,
            dtype="float32",
        )

    def encode_query(
        self,
        text,
    ):
        if hasattr(
            self.model,
            "encode_query",
        ):
            vector = (
                self.model.encode_query(
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        else:
            vector = (
                self.model.encode(
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        return np.asarray(
            vector,
            dtype="float32",
        )

    # Compatibility with old project code.
    def encode(
        self,
        texts,
    ):
        return self.encode_documents(
            texts
        )