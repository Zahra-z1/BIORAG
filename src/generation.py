from __future__ import annotations

import re

from .config import (
    ALLOW_MODEL_DOWNLOAD,
    ANSWER_WINDOW_COUNT,
    ANSWER_WINDOW_THRESHOLD,
    GENERATION_BACKEND,
    GENERATION_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
)

from .prompts import GENERATION_PROMPT
from .retrieval import get_retriever


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


def _clean_text(text: str) -> str:
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

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _sentences(text: str):
    if not text:
        return []

    return [
        _clean_text(part)
        for part in _SENTENCE_SPLIT.split(text)
        if _clean_text(part)
    ]


def _noise_sentence(text: str) -> bool:
    text = (
        text
        or ""
    ).strip()

    lower = text.lower()

    if len(text) < 25:
        return True

    if "table of contents" in lower:
        return True

    if re.search(
        r"^\s*(?:references|bibliography)\s*$",
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

    # A question appearing inside a textbook
    # should not itself become the answer.
    if text.endswith("?"):
        return True

    if len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )
    ) >= 3:
        return True

    if len(
        re.findall(
            r"\[[0-9,\-– ]+\]",
            text,
        )
    ) >= 5:
        return True

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if letters:
        upper_ratio = (
            sum(
                char.isupper()
                for char in letters
            )
            / len(letters)
        )

        if (
            upper_ratio > 0.78
            and len(text) > 100
        ):
            return True

    return False


def _tokens(text: str):
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

    first_tokens = _tokens(
        first
    )

    second_tokens = _tokens(
        second
    )

    if (
        not first_tokens
        or not second_tokens
    ):
        return 0.0

    return (
        len(
            first_tokens
            & second_tokens
        )
        /
        len(
            first_tokens
            | second_tokens
        )
    )


def _normalized_key(
    text: str,
) -> str:

    return re.sub(
        r"\W+",
        " ",
        (
            text
            or ""
        ).lower(),
    ).strip()


def _sentence_candidates(
    evidence,
):
    """
    Create ONE answer sentence per candidate.

    The reranker sees nearby context, but only
    the central sentence is returned.

    This prevents duplicate nested windows such as:
    - sentence A
    - sentence A + sentence B
    - sentence A + sentence B + sentence C
    """

    candidates = []

    seen = set()

    for evidence_rank, item in enumerate(
        evidence
    ):

        raw_sentences = _sentences(
            item.get(
                "text",
                "",
            )
        )

        if not raw_sentences:
            continue

        for index, sentence in enumerate(
            raw_sentences
        ):

            if _noise_sentence(
                sentence
            ):
                continue

            key = _normalized_key(
                sentence
            )

            if (
                not key
                or key in seen
            ):
                continue

            seen.add(
                key
            )

            # The CrossEncoder gets the previous
            # and next sentence for context.
            start = max(
                0,
                index - 1,
            )

            end = min(
                len(
                    raw_sentences
                ),
                index + 2,
            )

            context_parts = [
                value
                for value
                in raw_sentences[
                    start:end
                ]
                if not _noise_sentence(
                    value
                )
            ]

            context = _clean_text(
                " ".join(
                    context_parts
                )
            )

            section = (
                item.get(
                    "section",
                    "",
                )
                or ""
            ).strip()

            score_parts = []

            if section:
                score_parts.append(
                    f"Section: {section}"
                )

            score_parts.append(
                context
                or sentence
            )

            candidates.append(
                {
                    "text":
                        sentence,

                    "score_text":
                        "\n".join(
                            score_parts
                        ),

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

                    "section":
                        section,

                    "evidence_rank":
                        evidence_rank,

                    "retrieval_score":
                        float(
                            item.get(
                                "score",
                                0.0,
                            )
                        ),
                }
            )

    return candidates


def _select_diverse_candidates(
    question: str,
    evidence,
):
    candidates = (
        _sentence_candidates(
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
                item[
                    "score_text"
                ]
                for item
                in candidates
            ],
        )
    )

    for candidate, model_score in zip(
        candidates,
        model_scores,
    ):

        candidate[
            "model_score"
        ] = float(
            model_score
        )

        candidate[
            "answer_score"
        ] = (
            0.86
            * float(
                model_score
            )

            + 0.14
            * candidate[
                "retrieval_score"
            ]
        )

    candidates.sort(
        key=lambda item:
            item[
                "answer_score"
            ],
        reverse=True,
    )

    best_score = float(
        candidates[0][
            "answer_score"
        ]
    )

    # Keep enough evidence for explanation/list
    # questions, but reject weak unrelated text.
    minimum_score = max(
        ANSWER_WINDOW_THRESHOLD,
        best_score * 0.62,
    )

    pool = [
        item
        for item
        in candidates[:40]
        if float(
            item[
                "answer_score"
            ]
        ) >= minimum_score
    ]

    if not pool:
        pool = [
            candidates[0]
        ]

    selected = []

    per_page = {}

    while (
        pool
        and len(selected)
        < ANSWER_WINDOW_COUNT
    ):

        best_item = None

        best_value = float(
            "-inf"
        )

        for candidate in pool:

            relevance = float(
                candidate[
                    "answer_score"
                ]
            )

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

            page_count = (
                per_page.get(
                    page_key,
                    0,
                )
            )

            # Prefer different useful information
            # rather than five similar sentences
            # from one page.
            diversity_penalty = (
                0.10
                * page_count
            )

            # Maximal Marginal Relevance.
            value = (
                0.78
                * relevance

                - 0.22
                * redundancy

                - diversity_penalty
            )

            if value > best_value:

                best_value = (
                    value
                )

                best_item = (
                    candidate
                )

        if best_item is None:
            break

        best_key = (
            _normalized_key(
                best_item[
                    "text"
                ]
            )
        )

        duplicate = False

        for chosen in selected:

            chosen_key = (
                _normalized_key(
                    chosen[
                        "text"
                    ]
                )
            )

            if (
                best_key
                == chosen_key

                or best_key
                in chosen_key

                or chosen_key
                in best_key

                or _jaccard(
                    best_item[
                        "text"
                    ],
                    chosen[
                        "text"
                    ],
                ) >= 0.66
            ):

                duplicate = True

                break

        pool.remove(
            best_item
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

        per_page[
            page_key
        ] = (
            per_page.get(
                page_key,
                0,
            )
            + 1
        )

    return selected


def extractive_answer(
    question,
    evidence,
):

    if not evidence:

        return (
            "I could not find enough evidence "
            "in the indexed PDFs to answer "
            "this question."
        )

    selected = (
        _select_diverse_candidates(
            question,
            evidence,
        )
    )

    if not selected:

        best = evidence[0]

        return (
            f"{_clean_text(best.get('text', ''))} "
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
            lines.append(
                cited
            )

        else:
            lines.append(
                f"- {cited}"
            )

    return "\n".join(
        lines
    )


class LocalGenerator:

    def __init__(
        self,
        model_name=GENERATION_MODEL,
    ):

        import torch

        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        self.torch = torch

        local_only = (
            not ALLOW_MODEL_DOWNLOAD
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name,
                local_files_only=local_only,
            )
        )

        config = (
            AutoConfig
            .from_pretrained(
                model_name,
                local_files_only=local_only,
            )
        )

        self.is_encoder_decoder = bool(
            getattr(
                config,
                "is_encoder_decoder",
                False,
            )
        )

        model_class = (
            AutoModelForSeq2SeqLM
            if self.is_encoder_decoder
            else AutoModelForCausalLM
        )

        self.model = (
            model_class
            .from_pretrained(
                model_name,
                local_files_only=local_only,
            )
        )

        self.device = (
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()


    def _max_input_tokens(
        self,
    ):

        values = []

        for value in [
            getattr(
                self.tokenizer,
                "model_max_length",
                None,
            ),

            getattr(
                self.model.config,
                "max_position_embeddings",
                None,
            ),

            getattr(
                self.model.config,
                "n_positions",
                None,
            ),
        ]:

            if (
                isinstance(
                    value,
                    int,
                )
                and 128
                <= value
                < 100000
            ):

                values.append(
                    value
                )

        return (
            min(values)
            if values
            else 2048
        )


    def generate(
        self,
        question,
        evidence,
    ):

        context_parts = []

        used = 0

        for item in evidence:

            block = (
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]\n"
                f"{item['text']}"
            )

            if (
                used
                + len(block)
                > MAX_CONTEXT_CHARS
            ):

                remaining = (
                    MAX_CONTEXT_CHARS
                    - used
                )

                if remaining > 250:

                    context_parts.append(
                        block[
                            :remaining
                        ]
                    )

                break

            context_parts.append(
                block
            )

            used += len(
                block
            )

        prompt = (
            GENERATION_PROMPT
            .format(
                question=question,
                context="\n\n".join(
                    context_parts
                ),
            )
        )

        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self._max_input_tokens()
                ),
            )
        )

        inputs = {
            key:
                value.to(
                    self.device
                )

            for key, value
            in inputs.items()
        }

        kwargs = {
            "max_new_tokens":
                MAX_NEW_TOKENS,

            "do_sample":
                False,

            "num_beams":
                (
                    3
                    if self.is_encoder_decoder
                    else 1
                ),

            "no_repeat_ngram_size":
                3,

            "use_cache":
                True,
        }

        if (
            self.tokenizer.pad_token_id
            is not None
        ):

            kwargs[
                "pad_token_id"
            ] = (
                self.tokenizer
                .pad_token_id
            )

        with self.torch.inference_mode():

            output = (
                self.model.generate(
                    **inputs,
                    **kwargs,
                )
            )

        if self.is_encoder_decoder:

            tokens = (
                output[0]
            )

        else:

            input_length = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            tokens = (
                output[0][
                    input_length:
                ]
            )

        return (
            self.tokenizer.decode(
                tokens,
                skip_special_tokens=True,
            )
            .strip()
        )


_generator = None
_generator_error = None


def generate_answer(
    question,
    evidence,
):

    global _generator
    global _generator_error

    if not evidence:

        return (
            "I could not find enough evidence "
            "in the indexed PDFs.",
            "none",
        )

    if GENERATION_BACKEND in {
        "extractive",
        "reranked-extractive",
    }:

        return (
            extractive_answer(
                question,
                evidence,
            ),
            "extractive",
        )

    if GENERATION_BACKEND in {
        "auto",
        "transformers",
    }:

        try:

            if (
                _generator is None
                and _generator_error
                is None
            ):

                _generator = (
                    LocalGenerator()
                )

            if (
                _generator
                is not None
            ):

                answer = (
                    _generator.generate(
                        question,
                        evidence,
                    )
                )

                if answer:

                    return (
                        answer,
                        "transformers",
                    )

        except Exception as exc:

            _generator_error = str(
                exc
            )

    return (
        extractive_answer(
            question,
            evidence,
        ),
        "extractive",
    )


def generator_status():

    return {
        "backend":
            GENERATION_BACKEND,

        "model":
            GENERATION_MODEL,

        "error":
            _generator_error,
    }