from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    ALLOW_MODEL_DOWNLOAD,
    ENABLE_RERANKER,
    ENABLE_SEMANTIC,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RELEVANCE_THRESHOLD,
    RERANKER_MODEL,
    TOP_K,
)


STOP = {
    "what", "which", "who", "where",
    "when", "why", "how",
    "does", "do", "did",
    "is", "are", "was", "were",
    "be", "being", "been",
    "a", "an", "the",
    "of", "to", "for", "from",
    "in", "on", "with",
    "and", "or", "about",
    "please", "tell", "me",
    "explain", "describe",
    "define", "give",
}


def normalize(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9]+",
            " ",
            (text or "").lower(),
        ),
    ).strip()


def words(text: str):

    return re.findall(
        r"[A-Za-z][A-Za-z0-9_-]{1,}",
        (text or "").lower(),
    )


def canon(token: str) -> str:

    token = token.lower()

    if (
        token.endswith("ies")
        and len(token) > 4
    ):
        return token[:-3] + "y"

    if (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith("ss")
    ):
        return token[:-1]

    return token


def query_terms(text: str):

    result = []

    for token in words(text):

        if (
            token in STOP
            or len(token) <= 2
        ):
            continue

        token = canon(token)

        if token not in result:
            result.append(token)

    return result


def intent(question: str) -> str:

    q = (
        question
        or ""
    ).strip().lower()

    if re.match(
        r"^(?:"
        r"what\s+(?:is|are)|"
        r"define"
        r")\b",
        q,
    ):
        return "definition"

    if re.search(
        r"\b(?:"
        r"phases?|stages?|steps?|"
        r"types?|levels?|"
        r"components?|parts?"
        r")\b",
        q,
    ):
        return "list"

    if re.match(
        r"^(?:"
        r"explain|describe|how"
        r")\b",
        q,
    ):
        return "explanation"

    return "general"


def subject(question: str) -> str:

    q = (
        question
        or ""
    ).strip().rstrip("?.!")

    patterns = [
        r"^(?:what|which)\s+(?:is|are)\s+(.+)$",
        r"^define\s+(.+)$",
        r"^(?:explain|describe)\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            q,
            re.I,
        )

        if match:

            return re.sub(
                r"^(?:a|an|the)\s+",
                "",
                match.group(1).strip(),
                flags=re.I,
            )

    # Example:
    # "cell cycle phases"
    # becomes
    # "cell cycle"

    cleaned = re.sub(
        r"\b(?:"
        r"phases?|stages?|steps?|"
        r"types?|levels?|"
        r"components?|parts?"
        r")\b",
        " ",
        q,
        flags=re.I,
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()


def subject_tokens(text: str):

    return [
        canon(token)
        for token in words(text)
        if (
            token not in STOP
            and len(token) > 2
        )
    ]


def contains_subject(
    text: str,
    requested_subject: str,
):

    wanted = set(
        subject_tokens(
            requested_subject
        )
    )

    if not wanted:
        return False

    available = {
        canon(token)
        for token in words(text)
    }

    return wanted.issubset(
        available
    )


def definition_score(
    text: str,
    requested_subject: str,
):

    terms = subject_tokens(
        requested_subject
    )

    if (
        not terms
        or not contains_subject(
            text,
            requested_subject,
        )
    ):
        return 0.0

    pattern = r"\s+".join(
        re.escape(term)
        + r"s?"
        for term in terms
    )

    normalized = normalize(
        text
    )

    best = 0.0

    patterns = [
        (
            rf"\b{pattern}\s+"
            rf"(?:means|refers\s+to)\b",
            1.0,
        ),

        (
            rf"\b{pattern}\s+"
            rf"(?:can\s+be\s+)?"
            rf"defined\s+as\b",
            1.0,
        ),

        (
            rf"\b(?:the\s+)?"
            rf"{pattern}\s+"
            rf"(?:is|are)\s+"
            rf"(?:a|an|the)\b",
            0.98,
        ),

        (
            rf"\b(?:the\s+)?"
            rf"{pattern}\s+"
            rf"(?:"
            rf"consists\s+of|"
            rf"comprises|includes|contains"
            rf")\b",
            0.82,
        ),

        # Reverse definition:
        # "A segment ... is called a gene."
        (
            rf"\b(?:is|are)\s+"
            rf"(?:"
            rf"called|known\s+as|termed"
            rf")\s+"
            rf"(?:a\s+|an\s+|the\s+)?"
            rf"{pattern}\b",
            1.0,
        ),
    ]

    for regex, value in patterns:

        if re.search(
            regex,
            normalized,
            re.I,
        ):

            best = max(
                best,
                value,
            )

    # Prevent false definition:
    #
    # "Transcription is initiated..."
    #
    # does NOT define transcription.

    bad = re.search(
        rf"\b{pattern}\s+"
        rf"(?:is|are)\s+"
        rf"(?:"
        rf"initiated|regulated|"
        rf"inhibited|activated|"
        rf"performed|started|terminated"
        rf")\b",
        normalized,
        re.I,
    )

    if (
        bad
        and best < 0.80
    ):
        return 0.05

    return best


def list_score(
    text: str,
    requested_subject: str,
):

    if not contains_subject(
        text,
        requested_subject,
    ):
        return 0.0

    normalized = normalize(
        text
    )

    markers = [
        "phase",
        "stage",
        "step",
        "first",
        "second",
        "third",
        "then",
        "next",
        "finally",

        "g0",
        "g1",
        "g2",
        "s phase",
        "m phase",

        "prophase",
        "metaphase",
        "anaphase",
        "telophase",

        "primary",
        "secondary",
        "tertiary",
        "quaternary",
    ]

    hits = sum(
        1
        for marker in markers
        if marker in normalized
    )

    return min(
        1.0,
        0.25
        + 0.10 * hits,
    )


def title_score(
    source: str,
    requested_subject: str,
):

    title = {
        canon(token)
        for token in words(
            re.sub(
                r"\.pdf$",
                "",
                source or "",
                flags=re.I,
            )
        )
        if token not in STOP
    }

    wanted = set(
        subject_tokens(
            requested_subject
        )
    )

    if (
        not title
        or not wanted
    ):
        return 0.0

    overlap = (
        len(
            title & wanted
        )
        / len(wanted)
    )

    return min(
        1.0,
        overlap,
    )


def content_penalty(
    text: str,
):

    lower = (
        text
        or ""
    ).lower()

    penalty = 0.0

    if "table of contents" in lower:
        penalty += 0.70

    if (
        "references" in lower
        or "bibliography" in lower
    ):
        penalty += 0.55

    if len(
        re.findall(
            r"(?:https?://|www\.)",
            lower,
        )
    ) >= 3:
        penalty += 0.20

    if len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text or "",
        )
    ) >= 6:
        penalty += 0.20

    return min(
        0.85,
        penalty,
    )


def sigmoid(value: float):

    value = max(
        -20.0,
        min(
            20.0,
            float(value),
        ),
    )

    return (
        1.0
        / (
            1.0
            + math.exp(
                -value
            )
        )
    )


class Retriever:

    def __init__(
        self,
        enable_semantic: bool | None = None,
    ):

        if not METADATA_PATH.exists():

            raise FileNotFoundError(
                "Knowledge-base metadata "
                "is missing. Run: "
                "python -m scripts.build_index"
            )

        self.metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8",
            )
        )

        if not self.metadata:

            raise RuntimeError(
                "Knowledge-base metadata "
                "is empty."
            )

        self.texts = [
            item["text"]
            for item in self.metadata
        ]

        self.vectorizer = (
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            )
        )

        self.matrix = (
            self.vectorizer
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

        try:

            import faiss

            from .embeddings import (
                EmbeddingModel
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
                    "FAISS metadata mismatch"
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

            from sentence_transformers import (
                CrossEncoder
            )

            self._reranker = (
                CrossEncoder(
                    RERANKER_MODEL,
                    max_length=512,
                )
            )

        except Exception as exc:

            self._reranker_note = str(
                exc
            )


    def _lexical(
        self,
        question: str,
    ):

        requested_subject = (
            subject(
                question
            )
        )

        query_intent = (
            intent(
                question
            )
        )

        variants = [
            question,
        ]

        if requested_subject:

            variants.append(
                requested_subject
            )

            if query_intent == "definition":

                variants.extend(
                    [
                        f"{requested_subject} definition",
                        f"{requested_subject} means",
                        f"called {requested_subject}",
                    ]
                )

            elif query_intent == "list":

                variants.append(
                    f"{requested_subject} "
                    f"phases stages steps"
                )

            else:

                variants.append(
                    f"{requested_subject} "
                    f"mechanism process explanation"
                )

        scores = np.zeros(
            len(self.metadata),
            dtype=np.float32,
        )

        for variant in variants:

            vector = (
                self.vectorizer
                .transform(
                    [variant]
                )
            )

            if vector.nnz == 0:
                continue

            current = (
                cosine_similarity(
                    vector,
                    self.matrix,
                )[0]
            )

            scores = np.maximum(
                scores,
                current,
            )

        return scores


    def _semantic(
        self,
        question: str,
        count: int,
    ):

        self._ensure_semantic()

        if not self._semantic_ready:
            return {}

        query_vector = np.asarray(
            self._embedder.encode(
                [question]
            ),
            dtype="float32",
        )

        scores, ids = (
            self._index.search(
                query_vector,
                min(
                    count,
                    len(
                        self.metadata
                    ),
                ),
            )
        )

        return {
            int(index): float(score)

            for score, index
            in zip(
                scores[0],
                ids[0],
            )

            if index >= 0
        }


    def _neighbors(
        self,
        ids,
        radius=2,
    ):

        output = set(
            ids
        )

        for index in list(ids):

            source_name = (
                self.metadata[index]
                .get("source")
            )

            page = int(
                self.metadata[index]
                .get(
                    "page",
                    0,
                )
                or 0
            )

            for nearby in range(
                max(
                    0,
                    index - radius,
                ),
                min(
                    len(self.metadata),
                    index + radius + 1,
                ),
            ):

                item = (
                    self.metadata[
                        nearby
                    ]
                )

                if (
                    item.get("source")
                    != source_name
                ):
                    continue

                nearby_page = int(
                    item.get(
                        "page",
                        0,
                    )
                    or 0
                )

                if abs(
                    nearby_page
                    - page
                ) <= 2:

                    output.add(
                        nearby
                    )

        return output


    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        query = (
            query
            or ""
        ).strip()

        if not query:
            return []

        query_intent = (
            intent(
                query
            )
        )

        requested_subject = (
            subject(
                query
            )
        )

        lexical = (
            self._lexical(
                query
            )
        )

        semantic = (
            self._semantic(
                query,
                max(
                    100,
                    top_k * 20,
                ),
            )
        )

        lexical_ids = (
            np.argsort(
                lexical
            )[::-1][
                :max(
                    100,
                    top_k * 20,
                )
            ]
        )

        candidate_ids = {
            int(index)

            for index
            in lexical_ids

            if lexical[index] > 0
        }

        candidate_ids.update(
            semantic.keys()
        )

        # IMPORTANT:
        #
        # For definition/list questions,
        # search the entire small corpus for
        # actual answer passages.
        #
        # This is why:
        #
        # "What is translation?"
        # should find an actual definition,
        #
        # not "comparison of translation".

        if query_intent in {
            "definition",
            "list",
        }:

            for index, item in enumerate(
                self.metadata
            ):

                text = item.get(
                    "text",
                    "",
                )

                if not contains_subject(
                    text,
                    requested_subject,
                ):
                    continue

                if (
                    query_intent
                    == "definition"
                ):

                    if definition_score(
                        text,
                        requested_subject,
                    ) >= 0.75:

                        candidate_ids.add(
                            index
                        )

                else:

                    if list_score(
                        text,
                        requested_subject,
                    ) >= 0.45:

                        candidate_ids.add(
                            index
                        )

        candidate_ids = (
            self._neighbors(
                candidate_ids
            )
        )

        candidates = list(
            candidate_ids
        )

        # REAL second-stage reranking

        self._ensure_reranker()

        reranker_scores = {}

        if (
            self._reranker is not None
            and candidates
        ):

            pairs = [
                (
                    query,
                    self.metadata[index]
                    .get(
                        "text",
                        "",
                    )[:3500],
                )

                for index
                in candidates
            ]

            predictions = (
                self._reranker.predict(
                    pairs,
                    show_progress_bar=False,
                )
            )

            values = np.asarray(
                predictions
            ).reshape(-1)

            for index, value in zip(
                candidates,
                values,
            ):

                reranker_scores[
                    index
                ] = sigmoid(
                    float(value)
                )

        ranked = []

        for index in candidates:

            item = dict(
                self.metadata[
                    index
                ]
            )

            text = item.get(
                "text",
                "",
            )

            lexical_score = min(
                1.0,
                float(
                    lexical[
                        index
                    ]
                )
                * 4.0,
            )

            semantic_raw = max(
                0.0,
                float(
                    semantic.get(
                        index,
                        0.0,
                    )
                ),
            )

            semantic_score = max(
                0.0,
                min(
                    1.0,
                    (
                        semantic_raw
                        - 0.15
                    )
                    / 0.55,
                ),
            )

            rerank_score = (
                reranker_scores.get(
                    index,
                    0.0,
                )
            )

            definition_value = (
                definition_score(
                    text,
                    requested_subject,
                )
                if (
                    query_intent
                    == "definition"
                )
                else 0.0
            )

            list_value = (
                list_score(
                    text,
                    requested_subject,
                )
                if (
                    query_intent
                    == "list"
                )
                else 0.0
            )

            document_score = (
                title_score(
                    item.get(
                        "source",
                        "",
                    ),
                    requested_subject,
                )
            )

            penalty = (
                content_penalty(
                    text
                )
            )

            if self._reranker is not None:

                final_score = (
                    0.58
                    * rerank_score

                    + 0.12
                    * lexical_score

                    + 0.10
                    * semantic_score

                    + 0.08
                    * document_score
                )

            else:

                final_score = (
                    0.46
                    * lexical_score

                    + 0.28
                    * semantic_score

                    + 0.12
                    * document_score
                )

            if (
                query_intent
                == "definition"
            ):

                final_score += (
                    0.34
                    * definition_value
                )

            elif (
                query_intent
                == "list"
            ):

                final_score += (
                    0.30
                    * list_value
                )

            final_score *= (
                1.0
                - 0.70
                * penalty
            )

            item.update(
                {
                    "score":
                        round(
                            float(
                                final_score
                            ),
                            6,
                        ),

                    "lexical_score":
                        round(
                            float(
                                lexical[
                                    index
                                ]
                            ),
                            6,
                        ),

                    "semantic_score":
                        round(
                            semantic_raw,
                            6,
                        ),

                    "reranker_score":
                        round(
                            float(
                                rerank_score
                            ),
                            6,
                        ),

                    "definition_score":
                        round(
                            float(
                                definition_value
                            ),
                            6,
                        ),

                    "list_score":
                        round(
                            float(
                                list_value
                            ),
                            6,
                        ),

                    "source_match":
                        round(
                            float(
                                document_score
                            ),
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

        per_page = Counter()

        seen = set()

        for item in ranked:

            key = normalize(
                item.get(
                    "text",
                    "",
                )
            )[:500]

            if key in seen:
                continue

            seen.add(
                key
            )

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
        question: str,
        results: list[dict],
    ):

        query_intent = (
            intent(
                question
            )
        )

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

        best_definition = max(
            (
                float(
                    item.get(
                        "definition_score",
                        0.0,
                    )
                )
                for item in results
            ),
            default=0.0,
        )

        best_list = max(
            (
                float(
                    item.get(
                        "list_score",
                        0.0,
                    )
                )
                for item in results
            ),
            default=0.0,
        )

        if (
            query_intent
            == "definition"
        ):

            # Better to return "not found"
            # than random sentences about
            # the same word.

            answerable = bool(
                results
                and best_definition
                >= 0.75
            )

        elif (
            query_intent
            == "list"
        ):

            answerable = bool(
                results
                and (
                    best_list
                    >= 0.45

                    or best_score
                    >= RELEVANCE_THRESHOLD
                )
            )

        else:

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
                    best_score,
                    4,
                ),

            "answerability":
                round(
                    (
                        best_definition

                        if query_intent
                        == "definition"

                        else best_list
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