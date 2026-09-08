from .config import EMBEDDING_MODEL


class EmbeddingModel:
    """Lazy SentenceTransformer wrapper so lexical retrieval still works offline."""

    def __init__(self, model_name=EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
