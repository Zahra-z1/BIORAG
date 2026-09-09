from __future__ import annotations

import json
import re

from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from .config import (
    ALLOW_MODEL_DOWNLOAD,
    EMBEDDING_MODEL,
    ENABLE_RERANKER,
    ENABLE_SEMANTIC,
    FAISS_INDEX_PATH,
    INDEX_CONFIG_PATH,
    LEXICAL_CANDIDATES,
    METADATA_PATH,
    RELEVANCE_THRESHOLD,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
    SEMANTIC_CANDIDATES,
    TOP_K,
)


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9]+",
            " ",
            (text or "").lower(),
        ),
    ).strip()


def _safe_score(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def _noise_penalty(
    text: str,
) -> float:

    text = text or ""

    lower = text.lower()

    penalty = 0.0

    if "table of contents" in lower:
        penalty += 0.75

    if re.search(
        r"\b(?:references|bibliography)\b",
        lower,
    ):
        penalty += 0.55

    if text.count("....") >= 2:
        penalty += 0.35

    # Review/question pages often contain
    # the exact user question but no answer.
    question_signals = len(
        re.findall(
            r"\b(?:"
            r"what|which|who|where|when|why|how|"
            r"define|describe|explain|discuss|compare"
            r")\b",
            lower,
        )
    )

    if question_signals >= 5:
        penalty += 0.55

    if lower.count("?") >= 4:
        penalty += 0.45

    urls = len(
        re.findall(
            r"(?:https?://|www\.)",
            lower,
        )
    )

    if urls >= 3:
        penalty += 0.25

    years = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )
    )

    if years >= 7:
        penalty += 0.20

    citations = len(
        re.findall(
            r"\[[0-9,\-– ]+\]",
            text,
        )
    )

    if citations >= 8:
        penalty += 0.15

    return min(
        penalty,
        0.85,
    )


class Retriever:

    def __init__(
        self,
        enable_semantic=None,
    ):

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Knowledge-base index missing. "
                "Run: python -m scripts.build_index"
            )

        self.metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8",
            )
        )

        if not self.metadata:
            raise RuntimeError(
                "Knowledge-base metadata is empty."
            )

        self.texts = [
            item.get(
                "text",
                "",
            )
            for item in self.metadata
        ]

        # Normal word retrieval.
        self.word_vectorizer = (
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.995,
                sublinear_tf=True,
                strip_accents="unicode",
            )
        )

        self.word_matrix = (
            self.word_vectorizer
            .fit_transform(
                self.texts
            )
        )

        # Helps with bad PDF extraction,
        # spelling variation and split words.
        self.char_vectorizer = (
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            )
        )

        self.char_matrix = (
            self.char_vectorizer
            .fit_transform(
                self.texts
            )
        )

        self.enable_semantic = (
            ENABLE_SEMANTIC
            if enable_semantic is None
            else enable_semantic
        )

        self._semantic_ready = False
        self._semantic_note = None

        self._index = None
        self._embedder = None

        self._reranker = None
        self._reranker_note = None


    def _ensure_semantic(self):

        if (
            not self.enable_semantic
            or self._semantic_ready
            or self._semantic_note
        ):
            return

        if not FAISS_INDEX_PATH.exists():
            self._semantic_note = (
                "FAISS index not present"
            )

            return

        # Do not accidentally use an old
        # FAISS index built using another model.
        if INDEX_CONFIG_PATH.exists():

            try:
                config = json.loads(
                    INDEX_CONFIG_PATH.read_text(
                        encoding="utf-8",
                    )
                )

                built_model = config.get(
                    "embedding_model"
                )

                if (
                    built_model
                    and built_model
                    != EMBEDDING_MODEL
                ):
                    self._semantic_note = (
                        "Embedding model changed. "
                        "Rebuild index."
                    )

                    return

            except Exception:
                pass

        try:
            import faiss

            from .embeddings import (
                EmbeddingModel,
            )

            index = faiss.read_index(
                str(
                    FAISS_INDEX_PATH
                )
            )

            if (
                index.ntotal
                != len(self.metadata)
            ):
                self._semantic_note = (
                    "FAISS and metadata mismatch. "
                    "Rebuild index."
                )

                return

            self._index = index

            self._embedder = (
                EmbeddingModel(
                    EMBEDDING_MODEL
                )
            )

            self._semantic_ready = True

        except Exception as exc:
            self._semantic_note = str(
                exc
            )


    def _ensure_reranker(self):

        if (
            not ENABLE_RERANKER
            or self._reranker is not None
            or self._reranker_note
        ):
            return

        if not ALLOW_MODEL_DOWNLOAD:
            self._reranker_note = (
                "Reranker download disabled"
            )

            return

        try:
            import torch

            from sentence_transformers import (
                CrossEncoder,
            )

            self._reranker = (
                CrossEncoder(
                    RERANKER_MODEL,
                    activation_fn=(
                        torch.nn.Sigmoid()
                    ),
                    max_length=512,
                )
            )

        except Exception as exc:
            self._reranker_note = str(
                exc
            )


    def _lexical_scores(
        self,
        query: str,
    ):

        word_query = (
            self.word_vectorizer
            .transform(
                [query]
            )
        )

        char_query = (
            self.char_vectorizer
            .transform(
                [query]
            )
        )

        if word_query.nnz:

            word_scores = (
                cosine_similarity(
                    word_query,
                    self.word_matrix,
                )[0]
            )

        else:
            word_scores = np.zeros(
                len(self.metadata)
            )

        if char_query.nnz:

            char_scores = (
                cosine_similarity(
                    char_query,
                    self.char_matrix,
                )[0]
            )

        else:
            char_scores = np.zeros(
                len(self.metadata)
            )

        combined = (
            0.72 * word_scores
            + 0.28 * char_scores
        )

        return (
            combined,
            word_scores,
            char_scores,
        )


    def _semantic_scores(
        self,
        query: str,
    ):

        self._ensure_semantic()

        if not self._semantic_ready:
            return {}

        query_vector = (
            self._embedder
            .encode_query(
                query
            )
        )

        scores, ids = (
            self._index.search(
                np.asarray(
                    query_vector,
                    dtype="float32",
                ),
                min(
                    SEMANTIC_CANDIDATES,
                    len(self.metadata),
                ),
            )
        )

        result = {}

        for score, index in zip(
            scores[0],
            ids[0],
        ):

            if index >= 0:
                result[
                    int(index)
                ] = float(score)

        return result


    def _reranker_text(
        self,
        item,
    ):

        source_title = (
            Path(
                item.get(
                    "source",
                    "",
                )
            )
            .stem
            .replace(
                "_",
                " ",
            )
        )

        section = (
            item.get(
                "section",
                "",
            )
            or ""
        ).strip()

        parts = []

        if source_title:
            parts.append(
                f"Document: {source_title}"
            )

        if section:
            parts.append(
                f"Section: {section}"
            )

        parts.append(
            item.get(
                "text",
                "",
            )
        )

        return "\n".join(
            parts
        )


    def score_texts(
        self,
        query,
        texts,
    ):

        if not texts:
            return []

        self._ensure_reranker()

        if self._reranker is not None:

            try:
                scores = (
                    self._reranker.predict(
                        [
                            (
                                query,
                                text,
                            )
                            for text in texts
                        ],
                        show_progress_bar=False,
                    )
                )

                scores = np.asarray(
                    scores
                ).reshape(-1)

                return [
                    _safe_score(score)
                    for score in scores
                ]

            except Exception as exc:
                self._reranker_note = str(
                    exc
                )

        # Fallback if reranker cannot load.
        word_query = (
            self.word_vectorizer
            .transform(
                [query]
            )
        )

        char_query = (
            self.char_vectorizer
            .transform(
                [query]
            )
        )

        word_texts = (
            self.word_vectorizer
            .transform(
                texts
            )
        )

        char_texts = (
            self.char_vectorizer
            .transform(
                texts
            )
        )

        if word_query.nnz:

            word_scores = (
                cosine_similarity(
                    word_query,
                    word_texts,
                )[0]
            )

        else:
            word_scores = np.zeros(
                len(texts)
            )

        if char_query.nnz:

            char_scores = (
                cosine_similarity(
                    char_query,
                    char_texts,
                )[0]
            )

        else:
            char_scores = np.zeros(
                len(texts)
            )

        return [
            _safe_score(
                0.72 * word
                + 0.28 * char
            )
            for word, char in zip(
                word_scores,
                char_scores,
            )
        ]


    def search(
        self,
        query: str,
        top_k=TOP_K,
    ):

        query = (
            query
            or ""
        ).strip()

        if not query:
            return []

        (
            lexical,
            word_scores,
            char_scores,
        ) = self._lexical_scores(
            query
        )

        semantic = (
            self._semantic_scores(
                query
            )
        )

        lexical_ids = (
            np.argsort(
                lexical
            )[::-1][
                :min(
                    LEXICAL_CANDIDATES,
                    len(self.metadata),
                )
            ]
        )

        candidate_ids = {
            int(index)
            for index in lexical_ids
            if lexical[index] > 0
        }

        candidate_ids.update(
            semantic.keys()
        )

        if not candidate_ids:
            return []

        max_lexical = max(
            (
                float(
                    lexical[index]
                )
                for index
                in candidate_ids
            ),
            default=1.0,
        )

        if max_lexical <= 0:
            max_lexical = 1.0

        candidates = []

        for index in candidate_ids:

            lexical_norm = (
                float(
                    lexical[index]
                )
                / max_lexical
            )

            lexical_norm = (
                _safe_score(
                    lexical_norm
                )
            )

            semantic_norm = (
                _safe_score(
                    max(
                        0.0,
                        semantic.get(
                            index,
                            0.0,
                        ),
                    )
                )
            )

            if semantic:

                preliminary = (
                    0.48
                    * lexical_norm

                    + 0.52
                    * semantic_norm
                )

            else:

                preliminary = (
                    lexical_norm
                )

            candidates.append(
                (
                    preliminary,
                    index,
                    lexical_norm,
                    semantic_norm,
                )
            )

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        candidates = candidates[
            :min(
                RERANK_CANDIDATES,
                len(candidates),
            )
        ]

        reranker_passages = [
            self._reranker_text(
                self.metadata[index]
            )
            for (
                _,
                index,
                _,
                _,
            ) in candidates
        ]

        reranker_scores = (
            self.score_texts(
                query,
                reranker_passages,
            )
        )

        ranked = []

        for (
            preliminary,
            index,
            lexical_norm,
            semantic_norm,
        ), reranker_score in zip(
            candidates,
            reranker_scores,
        ):

            item = dict(
                self.metadata[index]
            )

            penalty = (
                _noise_penalty(
                    item.get(
                        "text",
                        "",
                    )
                )
            )

            if self._reranker is not None:

                score = (
                    0.70
                    * reranker_score

                    + 0.18
                    * semantic_norm

                    + 0.12
                    * lexical_norm
                )

            elif semantic:

                score = (
                    0.54
                    * semantic_norm

                    + 0.46
                    * lexical_norm
                )

            else:

                score = (
                    lexical_norm
                )

            score *= (
                1.0
                - 0.72
                * penalty
            )

            score = _safe_score(
                score
            )

            item.update(
                {
                    "score":
                        round(
                            score,
                            6,
                        ),

                    "reranker_score":
                        round(
                            float(
                                reranker_score
                            ),
                            6,
                        ),

                    "lexical_score":
                        round(
                            float(
                                lexical[index]
                            ),
                            6,
                        ),

                    "word_score":
                        round(
                            float(
                                word_scores[
                                    index
                                ]
                            ),
                            6,
                        ),

                    "char_score":
                        round(
                            float(
                                char_scores[
                                    index
                                ]
                            ),
                            6,
                        ),

                    "semantic_score":
                        round(
                            float(
                                semantic.get(
                                    index,
                                    0.0,
                                )
                            ),
                            6,
                        ),

                    "content_penalty":
                        round(
                            penalty,
                            6,
                        ),
                }
            )

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        selected = []

        seen = set()

        per_page = Counter()

        for item in ranked:

            key = _normalize(
                item.get(
                    "text",
                    "",
                )
            )[:700]

            if not key:
                continue

            if key in seen:
                continue

            page_key = (
                item.get(
                    "source"
                ),
                item.get(
                    "page"
                ),
            )

            if (
                per_page[
                    page_key
                ]
                >= 2
            ):
                continue

            selected.append(
                item
            )

            seen.add(
                key
            )

            per_page[
                page_key
            ] += 1

            if (
                len(selected)
                >= top_k
            ):
                break

        return selected


    def assess(
        self,
        question,
        results,
    ):

        best_score = max(
            (
                float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                for item in results
            ),
            default=0.0,
        )

        best_reranker = max(
            (
                float(
                    item.get(
                        "reranker_score",
                        0.0,
                    )
                )
                for item in results
            ),
            default=0.0,
        )

        answerable = bool(
            results
            and best_score
            >= RELEVANCE_THRESHOLD
        )

        return {
            "answerable":
                answerable,

            "confidence":
                round(
                    _safe_score(
                        best_score
                    ),
                    4,
                ),

            "reranker_confidence":
                round(
                    _safe_score(
                        best_reranker
                    ),
                    4,
                ),

            "semantic_enabled":
                self._semantic_ready,

            "semantic_note":
                self._semantic_note,

            "reranker_enabled":
                self._reranker
                is not None,

            "reranker_note":
                self._reranker_note,
        }


_retriever = None


def get_retriever():

    global _retriever

    if _retriever is None:
        _retriever = Retriever()

    return _retriever


def retrieve(
    query,
    top_k=TOP_K,
):

    return (
        get_retriever()
        .search(
            query,
            top_k,
        )
    )