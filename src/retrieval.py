from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    ENABLE_SEMANTIC,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RELEVANCE_THRESHOLD,
    TOP_K,
)


GENERIC_WORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "does", "do", "did", "is", "are", "was", "were",
    "be", "being", "been",
    "a", "an", "the",
    "of", "to", "for", "from", "in", "on", "with",
    "and", "or", "about",
    "please", "tell", "me",
    "explain", "describe", "define",
    "give", "difference", "different", "differ",
    "between", "compare", "comparison",
    "process", "work", "works", "working",
    "function", "functions", "role",
}


DEFINITION_NOUNS = {
    "process",
    "field",
    "branch",
    "science",
    "discipline",
    "study",
    "set",
    "sequence",
    "segment",
    "unit",
    "molecule",
    "structure",
    "cycle",
    "stage",
    "phase",
    "system",
    "mechanism",
    "information",
    "material",
    "collection",
    "region",
    "type",
    "form",
    "method",
    "technique",
    "procedure",
    "activity",
    "event",
    "series",
    "synthesis",
    "production",
    "formation",
    "copying",
    "transfer",
}


PROCESS_MARKERS = [
    "first",
    "second",
    "third",
    "then",
    "next",
    "finally",
    "begins",
    "starts",
    "occurs",
    "takes place",
    "followed by",
    "results in",
    "during",
    "stage",
    "phase",
    "step",
    "steps",
    "mechanism",
    "polymerase",
    "helicase",
    "strand",
    "template",
    "synthesis",
    "elongation",
    "termination",
    "initiation",
]


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


def tokens(text: str):
    return re.findall(
        r"[a-zA-Z][a-zA-Z0-9_-]{1,}",
        (text or "").lower(),
    )


def canonical(token: str) -> str:
    token = token.lower().strip()

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith("ss")
    ):
        return token[:-1]

    return token


def content_terms(question: str):
    result = []

    for token in tokens(question):
        if token in GENERIC_WORDS:
            continue

        if len(token) <= 2:
            continue

        token = canonical(token)

        if token not in result:
            result.append(token)

    return result


def get_subject(question: str) -> str:
    q = (question or "").strip().rstrip("?.!")

    patterns = [
        r"^(?:what|which)\s+(?:is|are)\s+(.+)$",
        r"^define\s+(.+)$",
        r"^describe\s+(.+)$",
        r"^explain\s+(.+)$",
        r"^how\s+(?:does|do)\s+(.+?)\s+(?:work|occur|happen)$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            q,
            re.I,
        )

        if match:
            subject = match.group(1).strip()

            subject = re.sub(
                r"^(?:a|an|the)\s+",
                "",
                subject,
                flags=re.I,
            )

            return subject

    return q


def is_definition_question(question: str):
    return bool(
        re.match(
            r"^\s*(?:"
            r"what\s+(?:is|are)|"
            r"which\s+(?:is|are)|"
            r"define"
            r")\b",
            question or "",
            re.I,
        )
    )


def is_explanation_question(question: str):
    return bool(
        re.match(
            r"^\s*(?:"
            r"explain|describe|how"
            r")\b",
            question or "",
            re.I,
        )
    )


def split_sentences(text: str):
    clean = re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            clean,
        )
        if len(sentence.strip()) >= 20
    ]


def subject_terms(subject: str):
    return {
        canonical(word)
        for word in tokens(subject)
        if word not in GENERIC_WORDS
        and len(word) > 2
    }


def ordered_subject_terms(subject: str):
    result = []

    for word in tokens(subject):
        if word in GENERIC_WORDS:
            continue

        if len(word) <= 2:
            continue

        word = canonical(word)

        if word not in result:
            result.append(word)

    return result


def contains_subject(text: str, subject: str):
    wanted = subject_terms(subject)

    if not wanted:
        return False

    available = {
        canonical(word)
        for word in tokens(text)
    }

    return wanted.issubset(
        available
    )


def subject_regex(subject: str):
    terms = ordered_subject_terms(
        subject
    )

    if not terms:
        return ""

    pieces = [
        re.escape(term) + r"s?"
        for term in terms
    ]

    pattern = pieces[0]

    for piece in pieces[1:]:
        pattern += (
            r"(?:\s+\w+){0,2}"
            r"\s+"
            + piece
        )

    return pattern


def exact_subject_match(
    text: str,
    subject: str,
):
    pattern = subject_regex(
        subject
    )

    if not pattern:
        return False

    return bool(
        re.search(
            rf"\b{pattern}\b",
            normalize(text),
            re.I,
        )
    )


def source_title_score(
    source: str,
    subject: str,
    query_terms: list[str],
):
    """
    Give a strong bonus to dedicated PDFs.

    Example:
    Explain protein structure
    -> protein-structure.pdf
    """

    filename = re.sub(
        r"\.pdf$",
        "",
        source or "",
        flags=re.I,
    )

    title_terms = {
        canonical(word)
        for word in tokens(filename)
        if word not in GENERIC_WORDS
    }

    wanted = (
        subject_terms(subject)
        | {
            canonical(word)
            for word in query_terms
        }
    )

    if (
        not title_terms
        or not wanted
    ):
        return 0.0

    common = (
        title_terms
        & wanted
    )

    overlap = (
        len(common)
        / max(
            1,
            len(wanted),
        )
    )

    subject_set = subject_terms(
        subject
    )

    if (
        subject_set
        and subject_set.issubset(
            title_terms
        )
    ):
        return 1.0

    if len(common) >= 2:
        return min(
            1.0,
            overlap + 0.25,
        )

    return min(
        1.0,
        overlap,
    )


def looks_like_toc_or_references(
    text: str,
):
    lower = (
        text or ""
    ).lower()

    penalty = 0.0

    if "table of contents" in lower:
        penalty += 0.75

    if (text or "").count("....") >= 2:
        penalty += 0.45

    if (
        "references" in lower
        or "bibliography" in lower
    ):
        penalty += 0.60

    years = re.findall(
        r"\b(?:19|20)\d{2}\b",
        text or "",
    )

    if len(years) >= 5:
        penalty += 0.25

    urls = re.findall(
        r"(?:https?://|www\.)",
        lower,
    )

    if len(urls) >= 3:
        penalty += 0.25

    question_count = len(
        re.findall(
            r"\b(?:"
            r"what|which|how|why|"
            r"describe|explain|define"
            r")\b",
            lower,
        )
    )

    if question_count >= 8:
        penalty += 0.25

    return min(
        penalty,
        0.85,
    )


def definition_sentence_score(
    sentence: str,
    subject: str,
):
    """
    Detect REAL definitions.

    Good:
    The genome is the entirety...

    Good:
    A segment of DNA ... is called a gene.

    Bad:
    Transcription is initiated from...
    """

    if not contains_subject(
        sentence,
        subject,
    ):
        return 0.0

    pattern = subject_regex(
        subject
    )

    if not pattern:
        return 0.0

    low = normalize(
        sentence
    )

    score = 0.0

    # Reverse definitions:
    # "... is called a gene"
    reverse_patterns = [
        (
            rf"\b(?:is|are)\s+"
            rf"called\s+"
            rf"(?:a\s+|an\s+|the\s+)?"
            rf"{pattern}\b"
        ),
        (
            rf"\b(?:is|are)\s+"
            rf"known\s+as\s+"
            rf"(?:a\s+|an\s+|the\s+)?"
            rf"{pattern}\b"
        ),
        (
            rf"\b(?:is|are)\s+"
            rf"termed\s+"
            rf"(?:a\s+|an\s+|the\s+)?"
            rf"{pattern}\b"
        ),
    ]

    for regex in reverse_patterns:
        if re.search(
            regex,
            low,
            re.I,
        ):
            score = max(
                score,
                1.0,
            )

    # Explicit definition patterns.
    direct_patterns = [
        (
            rf"(?:^|\bthe\s+)"
            rf"{pattern}\s+"
            rf"(?:means|refers\s+to)\b"
        ),
        (
            rf"{pattern}\s+"
            rf"(?:can\s+be\s+)?"
            rf"defined\s+as\b"
        ),
        (
            rf"\bdefinition\s+"
            rf"(?:of\s+)?"
            rf"{pattern}\b"
        ),
    ]

    for regex in direct_patterns:
        if re.search(
            regex,
            low,
            re.I,
        ):
            score = max(
                score,
                1.0,
            )

    # X is ...
    copula = re.search(
        rf"(?:^|\bthe\s+)"
        rf"{pattern}\s+"
        rf"(?:is|are)\s+(.+)",
        low,
        re.I,
    )

    if copula:
        tail = copula.group(1).strip()

        words = tail.split()

        if words:
            first = words[0]

            # X is a field/process/structure...
            if first in {
                "a",
                "an",
                "the",
            }:
                score = max(
                    score,
                    0.98,
                )

            elif canonical(
                first
            ) in DEFINITION_NOUNS:
                score = max(
                    score,
                    0.92,
                )

            elif tail.startswith(
                (
                    "one of ",
                    "part of ",
                    "a type of ",
                    "a form of ",
                )
            ):
                score = max(
                    score,
                    0.85,
                )

            # IMPORTANT:
            # "transcription is initiated"
            # is NOT a definition.
            elif first.endswith(
                (
                    "ed",
                    "ing",
                )
            ):
                score = max(
                    score,
                    0.10,
                )

            else:
                score = max(
                    score,
                    0.30,
                )

    # X consists/includes...
    if re.search(
        rf"(?:^|\bthe\s+)"
        rf"{pattern}\s+"
        rf"(?:"
        rf"consists\s+of|"
        rf"comprises|"
        rf"includes|"
        rf"contains"
        rf")\b",
        low,
        re.I,
    ):
        score = max(
            score,
            0.78,
        )

    # Heading-like fallback.
    # Useful for "cell cycle".
    if (
        score < 0.30
        and len(
            subject_terms(
                subject
            )
        ) >= 2
    ):
        first_part = (
            sentence[:160]
        )

        if contains_subject(
            first_part,
            subject,
        ):
            score = max(
                score,
                0.32,
            )

    if (
        35
        <= len(sentence)
        <= 450
    ):
        score += 0.04

    return min(
        score,
        1.0,
    )


def explanation_sentence_score(
    sentence: str,
    subject: str,
):
    if not contains_subject(
        sentence,
        subject,
    ):
        return 0.0

    lower = sentence.lower()

    score = 0.30

    marker_hits = 0

    for marker in PROCESS_MARKERS:
        if marker in lower:
            marker_hits += 1

    score += min(
        0.45,
        marker_hits * 0.06,
    )

    if re.search(
        r"\b(?:"
        r"primary|secondary|tertiary|"
        r"quaternary|level|levels|"
        r"consists|comprises|"
        r"includes|involves"
        r")\b",
        lower,
        re.I,
    ):
        score += 0.15

    pattern = subject_regex(
        subject
    )

    if pattern:
        match = re.search(
            rf"\b{pattern}\b",
            normalize(sentence),
            re.I,
        )

        if (
            match
            and match.start() <= 45
        ):
            score += 0.08

    return min(
        score,
        1.0,
    )


class Retriever:

    def __init__(
        self,
        enable_semantic: bool | None = None,
    ):

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Knowledge-base metadata is missing. "
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
            item["text"]
            for item in self.metadata
        ]

        self.vectorizer = (
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.995,
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

        self.vocabulary = set(
            self.vectorizer.vocabulary_
        )

        self.enable_semantic = (
            ENABLE_SEMANTIC
            if enable_semantic is None
            else enable_semantic
        )

        self._semantic_ready = False

        self._semantic_disabled_reason = (
            None
        )

        self._index = None
        self._embedder = None


    def _ensure_semantic(self):

        if (
            not self.enable_semantic
            or self._semantic_ready
            or self._semantic_disabled_reason
        ):
            return

        if not FAISS_INDEX_PATH.exists():
            self._semantic_disabled_reason = (
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
                self._semantic_disabled_reason = (
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

            self._semantic_disabled_reason = (
                str(exc)
            )


    def _lexical_scores(
        self,
        question: str,
    ):

        subject = get_subject(
            question
        )

        terms = content_terms(
            question
        )

        queries = [
            question,
        ]

        if terms:
            queries.append(
                " ".join(terms)
            )

        if subject:

            if is_definition_question(
                question
            ):

                queries.extend(
                    [
                        subject,
                        f"{subject} definition",
                        f"{subject} is a",
                        f"{subject} is the",
                        f"{subject} refers to",
                        f"called {subject}",
                        f"known as {subject}",
                    ]
                )

            else:

                queries.extend(
                    [
                        subject,
                        f"{subject} process",
                        f"{subject} mechanism",
                        f"{subject} stages",
                        f"{subject} phases",
                        f"{subject} structure",
                    ]
                )

        scores = np.zeros(
            len(self.metadata),
            dtype=np.float32,
        )

        for query in queries:

            vector = (
                self.vectorizer
                .transform(
                    [query]
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


    def _semantic_scores(
        self,
        question: str,
        candidate_count=140,
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
                    candidate_count,
                    len(
                        self.metadata
                    ),
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


    def _corpus_coverage(
        self,
        question: str,
    ):

        terms = content_terms(
            question
        )

        if not terms:
            return 0.0

        corpus = normalize(
            " ".join(
                self.texts
            )
        )

        matched = 0

        for term in terms:

            if re.search(
                rf"\b"
                rf"{re.escape(term)}"
                rf"s?\b",
                corpus,
            ):
                matched += 1

        return (
            matched
            / len(terms)
        )


    def _add_neighbors(
        self,
        candidate_ids,
        radius=2,
    ):
        """
        Include neighboring chunks.

        Important when a heading like
        "Cell Cycle" is separate from the
        explanatory paragraph.
        """

        expanded = set(
            candidate_ids
        )

        for index in list(
            candidate_ids
        ):

            item = self.metadata[
                index
            ]

            source = item.get(
                "source"
            )

            page = int(
                item.get(
                    "page",
                    0,
                )
                or 0
            )

            start = max(
                0,
                index - radius,
            )

            end = min(
                len(self.metadata),
                index + radius + 1,
            )

            for nearby_index in range(
                start,
                end,
            ):

                neighbor = (
                    self.metadata[
                        nearby_index
                    ]
                )

                if (
                    neighbor.get(
                        "source"
                    )
                    != source
                ):
                    continue

                neighbor_page = int(
                    neighbor.get(
                        "page",
                        0,
                    )
                    or 0
                )

                if abs(
                    neighbor_page
                    - page
                ) <= 2:

                    expanded.add(
                        nearby_index
                    )

        return expanded


    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ):

        query = (
            query or ""
        ).strip()

        if not query:
            return []

        lexical = (
            self._lexical_scores(
                query
            )
        )

        semantic = (
            self._semantic_scores(
                query,
                candidate_count=max(
                    140,
                    top_k * 24,
                ),
            )
        )

        subject = get_subject(
            query
        )

        terms = content_terms(
            query
        )

        definition_intent = (
            is_definition_question(
                query
            )
        )

        explanation_intent = (
            is_explanation_question(
                query
            )
        )

        lexical_ids = (
            np.argsort(
                lexical
            )[::-1][
                :max(
                    140,
                    top_k * 24,
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

        candidate_ids = (
            self._add_neighbors(
                candidate_ids,
                radius=2,
            )
        )

        ranked = []

        for index in candidate_ids:

            item = dict(
                self.metadata[
                    index
                ]
            )

            text = item.get(
                "text",
                "",
            )

            available_terms = {
                canonical(word)
                for word in tokens(
                    text
                )
            }

            lexical_raw = float(
                lexical[index]
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

            if terms:

                coverage = (
                    sum(
                        1
                        for term
                        in terms
                        if canonical(
                            term
                        )
                        in available_terms
                    )
                    / len(terms)
                )

            else:

                coverage = 0.0

            subject_present = (
                1.0
                if contains_subject(
                    text,
                    subject,
                )
                else 0.0
            )

            exact_match = (
                1.0
                if exact_subject_match(
                    text,
                    subject,
                )
                else 0.0
            )

            source_match = (
                source_title_score(
                    item.get(
                        "source",
                        "",
                    ),
                    subject,
                    terms,
                )
            )

            sentence_scores = []

            for sentence in (
                split_sentences(
                    text
                )
            ):

                if definition_intent:

                    sentence_score = (
                        definition_sentence_score(
                            sentence,
                            subject,
                        )
                    )

                elif explanation_intent:

                    sentence_score = (
                        explanation_sentence_score(
                            sentence,
                            subject,
                        )
                    )

                else:

                    sentence_score = (
                        0.25
                        if contains_subject(
                            sentence,
                            subject,
                        )
                        else 0.0
                    )

                sentence_scores.append(
                    sentence_score
                )

            answerability = max(
                sentence_scores,
                default=0.0,
            )

            content_penalty = (
                looks_like_toc_or_references(
                    text
                )
            )

            lexical_scaled = min(
                1.0,
                lexical_raw * 4.0,
            )

            semantic_scaled = max(
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

            if definition_intent:

                final_score = (
                    0.20
                    * lexical_scaled
                    + 0.12
                    * semantic_scaled
                    + 0.08
                    * coverage
                    + 0.08
                    * subject_present
                    + 0.10
                    * exact_match
                    + 0.34
                    * answerability
                    + 0.20
                    * source_match
                )

                # Real definitions receive
                # a major bonus.
                if answerability >= 0.90:
                    final_score += 0.20

                elif answerability >= 0.70:
                    final_score += 0.10

            elif explanation_intent:

                final_score = (
                    0.20
                    * lexical_scaled
                    + 0.18
                    * semantic_scaled
                    + 0.08
                    * coverage
                    + 0.08
                    * subject_present
                    + 0.10
                    * exact_match
                    + 0.26
                    * answerability
                    + 0.28
                    * source_match
                )

            else:

                final_score = (
                    0.30
                    * lexical_scaled
                    + 0.22
                    * semantic_scaled
                    + 0.12
                    * coverage
                    + 0.10
                    * subject_present
                    + 0.10
                    * exact_match
                    + 0.24
                    * source_match
                )

            final_score *= (
                1.0
                - 0.70
                * content_penalty
            )

            # Mild penalty only.
            # Do NOT hard reject concepts such
            # as cell cycle simply because they
            # lack a neat "X is..." sentence.
            if (
                definition_intent
                and answerability < 0.15
                and source_match < 0.50
            ):
                final_score *= 0.82

            item.update(
                {
                    "score": round(
                        float(
                            final_score
                        ),
                        6,
                    ),

                    "lexical_score": round(
                        lexical_raw,
                        6,
                    ),

                    "semantic_score": round(
                        semantic_raw,
                        6,
                    ),

                    "term_coverage": round(
                        float(
                            coverage
                        ),
                        6,
                    ),

                    "answerability_score": round(
                        float(
                            answerability
                        ),
                        6,
                    ),

                    "source_match": round(
                        float(
                            source_match
                        ),
                        6,
                    ),

                    "exact_phrase_match": round(
                        float(
                            exact_match
                        ),
                        6,
                    ),

                    "content_penalty": round(
                        float(
                            content_penalty
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

        seen_text = set()

        for item in ranked:

            text_key = normalize(
                item.get(
                    "text",
                    "",
                )
            )[:400]

            if text_key in seen_text:
                continue

            seen_text.add(
                text_key
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

        corpus_coverage = (
            self._corpus_coverage(
                question
            )
        )

        best_score = max(
            (
                float(
                    result.get(
                        "score",
                        0.0,
                    )
                )
                for result
                in results
            ),
            default=0.0,
        )

        best_answerability = max(
            (
                float(
                    result.get(
                        "answerability_score",
                        0.0,
                    )
                )
                for result
                in results
            ),
            default=0.0,
        )

        best_source_match = max(
            (
                float(
                    result.get(
                        "source_match",
                        0.0,
                    )
                )
                for result
                in results
            ),
            default=0.0,
        )

        threshold = min(
            RELEVANCE_THRESHOLD,
            0.22,
        )

        answerable = bool(
            results
            and corpus_coverage > 0
            and (
                best_score
                >= threshold

                or best_answerability
                >= 0.45

                or best_source_match
                >= 0.65
            )
        )

        return {
            "answerable":
                answerable,

            "confidence":
                round(
                    best_score,
                    4,
                ),

            "query_corpus_coverage":
                round(
                    corpus_coverage,
                    4,
                ),

            "answerability":
                round(
                    best_answerability,
                    4,
                ),

            "source_match":
                round(
                    best_source_match,
                    4,
                ),

            "semantic_enabled":
                self._semantic_ready,

            "semantic_note":
                self._semantic_disabled_reason,
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