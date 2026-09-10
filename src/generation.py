from __future__ import annotations

import re
from typing import Optional

from .config import (
    ANSWER_WINDOW_COUNT,
    ENABLE_QA_EXTRACTOR,
    GENERATION_BACKEND,
    QA_MODEL,
    QA_TOP_CONTEXTS,
)

from .retrieval import get_retriever


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


# =========================================================
# TEXT HELPERS
# =========================================================

def _clean(text: str) -> str:
    text = re.sub(
        r"[ \t]+",
        " ",
        text or "",
    )

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _sentences(text: str) -> list[str]:
    return [
        _clean(part)
        for part in _SENTENCE_SPLIT.split(
            text or ""
        )
        if _clean(part)
    ]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(
            r"[A-Za-z0-9]+",
            text or "",
        )
        if len(token) > 2
    }


def _jaccard(
    first: str,
    second: str,
) -> float:

    a = _tokens(first)
    b = _tokens(second)

    if not a or not b:
        return 0.0

    return (
        len(a & b)
        / len(a | b)
    )


def _noise(text: str) -> bool:
    text = _clean(text)

    lower = text.lower()

    if len(text) < 35:
        return True

    if len(
        re.findall(
            r"[A-Za-z]+",
            text,
        )
    ) < 6:
        return True

    if "table of contents" in lower:
        return True

    if re.match(
        r"^(?:references|bibliography)\b",
        lower,
    ):
        return True

    if len(
        re.findall(
            r"(?:https?://|www\.)",
            lower,
        )
    ) >= 2:
        return True

    # Do not return textbook questions.
    if text.endswith("?"):
        return True

    if re.search(
        r"\b(?:"
        r"worksheet|"
        r"exercise|"
        r"questions and answers"
        r")\b",
        lower,
    ):
        return True

    return False


# =========================================================
# GENERIC QUESTION TYPE DETECTION
# =========================================================

def _comparison_parts(
    question: str,
) -> Optional[tuple[str, str]]:

    q = (
        question
        or ""
    ).strip().rstrip("?.!")

    patterns = [
        (
            r"^(?:what\s+(?:is|are)\s+the\s+)?"
            r"(?:difference|differences)\s+between\s+"
            r"(.+?)\s+and\s+(.+)$"
        ),
        (
            r"^compare\s+"
            r"(.+?)\s+"
            r"(?:and|with|to)\s+"
            r"(.+)$"
        ),
        (
            r"^(.+?)\s+"
            r"(?:vs\.?|versus)\s+"
            r"(.+)$"
        ),
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            q,
            re.I,
        )

        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()

            if left and right:
                return left, right

    return None


def _is_multi_evidence_question(
    question: str,
) -> bool:
    """
    Generic detection only.

    No biology-specific terminology.
    """

    q = (
        question
        or ""
    ).lower()

    if re.match(
        r"^\s*(?:"
        r"explain|"
        r"describe|"
        r"discuss|"
        r"compare|"
        r"summarize|"
        r"summarise|"
        r"why|"
        r"how"
        r")\b",
        q,
    ):
        return True

    if re.search(
        r"\b(?:"
        r"difference|differences|"
        r"stages|phases|steps|"
        r"types|kinds|"
        r"functions|roles|"
        r"components|parts|"
        r"advantages|disadvantages|"
        r"causes|effects|"
        r"reasons|methods|ways|"
        r"mechanisms|processes"
        r")\b",
        q,
    ):
        return True

    return False


def _is_direct_question(
    question: str,
) -> bool:

    if _is_multi_evidence_question(
        question
    ):
        return False

    q = (
        question
        or ""
    ).strip()

    return bool(
        re.match(
            r"^(?:"
            r"define\b|"
            r"what\s+(?:is|are|was|were)\b|"
            r"who\s+(?:is|are|was|were)\b|"
            r"where\s+(?:is|are|was|were)\b|"
            r"when\s+(?:is|was|did|does)\b|"
            r"which\s+(?:is|are)\b|"
            r"how\s+many\b|"
            r"how\s+much\b"
            r")",
            q,
            re.I,
        )
    )


# =========================================================
# EXTRACTIVE QA MODEL
# =========================================================

_qa_pipeline = None
_qa_error = None


def _get_qa_pipeline():

    global _qa_pipeline
    global _qa_error

    if not ENABLE_QA_EXTRACTOR:
        return None

    if _qa_pipeline is not None:
        return _qa_pipeline

    if _qa_error is not None:
        return None

    try:
        from transformers import pipeline

        _qa_pipeline = pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
        )

        return _qa_pipeline

    except Exception as exc:
        _qa_error = str(exc)
        return None


def _sentence_containing_answer(
    context: str,
    start: int,
    end: int,
) -> str:

    for match in re.finditer(
        r"[^.!?\n]+(?:[.!?]|$)",
        context or "",
    ):

        if (
            match.start() <= start
            and match.end() >= end
        ):
            return _clean(
                match.group(0)
            )

    return ""


def _qa_answer(
    question: str,
    evidence: list[dict],
) -> Optional[str]:

    qa = _get_qa_pipeline()

    if qa is None or not evidence:
        return None

    top_retrieval = max(
        float(
            item.get(
                "score",
                0.0,
            )
        )
        for item in evidence
    )

    # Only allow QA to inspect passages
    # reasonably close to the top result.
    #
    # Example:
    # genome page 97 = high score
    # unrelated chromosome pages = much lower
    #
    # Therefore chromosome cannot steal the answer.
    cutoff = max(
        0.12,
        top_retrieval * 0.72,
    )

    contexts = [
        item
        for item in evidence
        if float(
            item.get(
                "score",
                0.0,
            )
        ) >= cutoff
    ][
        :QA_TOP_CONTEXTS
    ]

    if not contexts:
        contexts = evidence[:1]

    candidates = []

    for item in contexts:

        context = _clean(
            item.get(
                "text",
                "",
            )
        )

        if len(context) < 60:
            continue

        try:
            result = qa(
                question=question,
                context=context,
                handle_impossible_answer=True,
            )

        except Exception:
            continue

        answer = _clean(
            result.get(
                "answer",
                "",
            )
        )

        if not answer:
            continue

        qa_score = float(
            result.get(
                "score",
                0.0,
            )
        )

        start = int(
            result.get(
                "start",
                0,
            )
        )

        end = int(
            result.get(
                "end",
                0,
            )
        )

        sentence = (
            _sentence_containing_answer(
                context,
                start,
                end,
            )
        )

        if (
            not sentence
            or _noise(sentence)
        ):
            sentence = answer

        retrieval_score = float(
            item.get(
                "score",
                0.0,
            )
        )

        final_score = (
            0.82 * qa_score
            + 0.18 * retrieval_score
        )

        candidates.append(
            {
                "text":
                    sentence,

                "answer":
                    answer,

                "score":
                    final_score,

                "source":
                    item.get(
                        "source",
                        "Unknown source",
                    ),

                "page":
                    item.get(
                        "page",
                        "?",
                    ),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    best = candidates[0]

    return (
        f"{best['text']} "
        f"[Source: "
        f"{best['source']}, "
        f"p. {best['page']}]"
    )


# =========================================================
# PASSAGE WINDOWS FOR EXPLANATION QUESTIONS
# =========================================================

def _window_candidates(
    evidence: list[dict],
):

    if not evidence:
        return []

    best_retrieval = max(
        float(
            item.get(
                "score",
                0.0,
            )
        )
        for item in evidence
    )

    # Remove distant low-quality retrieval results.
    useful = [
        item
        for item in evidence
        if float(
            item.get(
                "score",
                0.0,
            )
        ) >= max(
            0.10,
            best_retrieval * 0.58,
        )
    ][:6]

    if not useful:
        useful = evidence[:3]

    candidates = []

    seen = set()

    for evidence_rank, item in enumerate(
        useful
    ):

        sentences = [
            sentence
            for sentence
            in _sentences(
                item.get(
                    "text",
                    "",
                )
            )
            if not _noise(
                sentence
            )
        ]

        if not sentences:
            continue

        # One or two sentence windows.
        for window_size in (
            1,
            2,
        ):

            for index in range(
                len(sentences)
                - window_size
                + 1
            ):

                window = _clean(
                    " ".join(
                        sentences[
                            index:
                            index + window_size
                        ]
                    )
                )

                if (
                    len(window) < 45
                    or len(window) > 850
                ):
                    continue

                key = re.sub(
                    r"\W+",
                    " ",
                    window.lower(),
                ).strip()

                if (
                    not key
                    or key in seen
                ):
                    continue

                seen.add(key)

                candidates.append(
                    {
                        "text":
                            window,

                        "source":
                            item.get(
                                "source",
                                "Unknown source",
                            ),

                        "page":
                            item.get(
                                "page",
                                "?",
                            ),

                        "retrieval_score":
                            float(
                                item.get(
                                    "score",
                                    0.0,
                                )
                            ),

                        "evidence_rank":
                            evidence_rank,
                    }
                )

    return candidates


def _rank_windows(
    question: str,
    evidence: list[dict],
    limit: int = 4,
    force_top_anchor: bool = True,
):

    candidates = (
        _window_candidates(
            evidence
        )
    )

    if not candidates:
        return []

    retriever = (
        get_retriever()
    )

    model_scores = (
        retriever.score_texts(
            question,
            [
                item["text"]
                for item
                in candidates
            ],
        )
    )

    for item, model_score in zip(
        candidates,
        model_scores,
    ):

        item[
            "window_score"
        ] = float(
            model_score
        )

        # IMPORTANT:
        #
        # Passage retrieval gets the majority
        # of the score.
        #
        # This prevents a random sentence from
        # a lower-ranked page from winning
        # simply because it repeats query words.
        item[
            "final_score"
        ] = (
            0.68
            * item[
                "retrieval_score"
            ]

            + 0.32
            * item[
                "window_score"
            ]

            - 0.008
            * item[
                "evidence_rank"
            ]
        )

    candidates.sort(
        key=lambda item:
            item[
                "final_score"
            ],
        reverse=True,
    )

    selected = []

    # -----------------------------------------------------
    # ANCHOR THE ANSWER IN THE TOP RETRIEVED PASSAGE
    # -----------------------------------------------------

    if force_top_anchor:

        top_passage_candidates = [
            item
            for item in candidates
            if item[
                "evidence_rank"
            ] == 0
        ]

        if top_passage_candidates:

            anchor = max(
                top_passage_candidates,
                key=lambda item:
                    item[
                        "window_score"
                    ],
            )

        else:
            anchor = candidates[0]

        selected.append(
            anchor
        )

    else:

        selected.append(
            candidates[0]
        )

    anchor_score = float(
        selected[0][
            "final_score"
        ]
    )

    pool = [
        item
        for item in candidates
        if (
            item is not selected[0]
            and item[
                "final_score"
            ] >= anchor_score * 0.68
        )
    ]

    page_counts = {
        (
            selected[0][
                "source"
            ],
            selected[0][
                "page"
            ],
        ): 1
    }

    while (
        pool
        and len(selected) < limit
    ):

        best_item = None

        best_value = float(
            "-inf"
        )

        for candidate in pool:

            redundancy = max(
                (
                    _jaccard(
                        candidate[
                            "text"
                        ],
                        chosen[
                            "text"
                        ],
                    )
                    for chosen
                    in selected
                ),
                default=0.0,
            )

            page_key = (
                candidate[
                    "source"
                ],
                candidate[
                    "page"
                ],
            )

            same_page_penalty = (
                0.06
                * page_counts.get(
                    page_key,
                    0,
                )
            )

            value = (
                candidate[
                    "final_score"
                ]

                - 0.18
                * redundancy

                - same_page_penalty
            )

            if value > best_value:
                best_value = value
                best_item = candidate

        if best_item is None:
            break

        pool.remove(
            best_item
        )

        duplicate = any(
            _jaccard(
                best_item[
                    "text"
                ],
                chosen[
                    "text"
                ],
            ) >= 0.68

            for chosen in selected
        )

        if duplicate:
            continue

        selected.append(
            best_item
        )

        page_key = (
            best_item[
                "source"
            ],
            best_item[
                "page"
            ],
        )

        page_counts[
            page_key
        ] = (
            page_counts.get(
                page_key,
                0,
            )
            + 1
        )

    return selected


# =========================================================
# GENERIC EXPLANATION ANSWER
# =========================================================

def _explanation_answer(
    question: str,
    evidence: list[dict],
):

    selected = _rank_windows(
        question,
        evidence,
        limit=max(
            2,
            min(
                ANSWER_WINDOW_COUNT,
                5,
            ),
        ),
        force_top_anchor=True,
    )

    if not selected:

        best = evidence[0]

        return (
            f"{_clean(best.get('text', ''))} "
            f"[Source: "
            f"{best.get('source', 'Unknown source')}, "
            f"p. {best.get('page', '?')}]"
        )

    lines = []

    for index, item in enumerate(
        selected
    ):

        cited = (
            f"{item['text']} "
            f"[Source: "
            f"{item['source']}, "
            f"p. {item['page']}]"
        )

        if index == 0:
            lines.append(cited)

        else:
            lines.append(
                f"- {cited}"
            )

    return "\n".join(
        lines
    )


# =========================================================
# COMPARISON
# =========================================================

def _merge_evidence(
    *groups,
):

    merged = []

    seen = set()

    for group in groups:

        for item in group:

            key = (
                item.get(
                    "source"
                ),
                item.get(
                    "page"
                ),
                re.sub(
                    r"\W+",
                    " ",
                    item.get(
                        "text",
                        "",
                    ).lower(),
                )[:500],
            )

            if key in seen:
                continue

            seen.add(key)

            merged.append(item)

    return merged


def _comparison_answer(
    question: str,
    evidence: list[dict],
):

    parts = _comparison_parts(
        question
    )

    if not parts:
        return None

    left, right = parts

    retriever = (
        get_retriever()
    )

    # Retrieve each concept independently.
    left_evidence = (
        retriever.search(
            f"What is {left}?",
            top_k=3,
        )
    )

    right_evidence = (
        retriever.search(
            f"What is {right}?",
            top_k=3,
        )
    )

    left_answer = _qa_answer(
        f"What is {left}?",
        left_evidence,
    )

    right_answer = _qa_answer(
        f"What is {right}?",
        right_evidence,
    )

    combined = _merge_evidence(
        evidence,
        left_evidence,
        right_evidence,
    )

    comparison_windows = (
        _rank_windows(
            question,
            combined,
            limit=2,
            force_top_anchor=False,
        )
    )

    lines = []

    if left_answer:
        lines.append(
            f"**{left}:** "
            f"{left_answer}"
        )

    if right_answer:
        lines.append(
            f"**{right}:** "
            f"{right_answer}"
        )

    if comparison_windows:

        lines.append(
            "**Relevant differences:**"
        )

        for item in comparison_windows:

            lines.append(
                f"- {item['text']} "
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]"
            )

    if not lines:
        return None

    return "\n".join(
        lines
    )


# =========================================================
# PUBLIC API
# =========================================================

def generate_answer(
    question: str,
    evidence: list[dict],
):

    if not evidence:

        return (
            "I could not find enough evidence "
            "in the indexed PDFs to answer "
            "this question.",
            "none",
        )

    # Comparison BEFORE direct QA because:
    #
    # "What is the difference between X and Y?"
    #
    # begins with "what is".
    comparison = (
        _comparison_answer(
            question,
            evidence,
        )
    )

    if comparison:

        return (
            comparison,
            "grounded-extractive",
        )

    # Definitions and simple factual questions.
    if _is_direct_question(
        question
    ):

        direct = _qa_answer(
            question,
            evidence,
        )

        if direct:

            return (
                direct,
                "extractive-qa",
            )

    # Explanations, processes, lists, why/how,
    # stages, functions, etc.
    return (
        _explanation_answer(
            question,
            evidence,
        ),
        "grounded-extractive",
    )


def generator_status():

    return {
        "backend":
            GENERATION_BACKEND,

        "qa_model":
            QA_MODEL,

        "qa_enabled":
            ENABLE_QA_EXTRACTOR,

        "qa_error":
            _qa_error,
    }