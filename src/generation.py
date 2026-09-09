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

from .prompts import (
    GENERATION_PROMPT,
)

from .retrieval import (
    get_retriever,
)


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


def _clean_text(
    text: str,
) -> str:

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

    return text.strip()


def _sentences(
    text: str,
):

    if not text:
        return []

    return [
        _clean_text(part)
        for part in _SENTENCE_SPLIT.split(
            text
        )
        if _clean_text(part)
    ]


def _noise_text(
    text: str,
) -> bool:

    text = (
        text
        or ""
    ).strip()

    lower = text.lower()

    if len(text) < 25:
        return True

    if "table of contents" in lower:
        return True

    if re.match(
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

    # Do not return a review/exercise
    # question as the answer.
    if (
        text.endswith("?")
        and re.match(
            r"^\s*(?:"
            r"what|which|who|where|when|why|how|"
            r"define|describe|explain|discuss|compare"
            r")\b",
            lower,
        )
    ):
        return True

    if len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )
    ) >= 3:
        return True

    return False


def _tokens(
    text: str,
):

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
):

    a = _tokens(first)
    b = _tokens(second)

    if not a or not b:
        return 0.0

    return (
        len(a & b)
        / len(a | b)
    )


def _window_candidates(
    evidence,
):

    candidates = []

    seen = set()

    for evidence_rank, item in enumerate(
        evidence
    ):

        sentences = [
            sentence
            for sentence in _sentences(
                item.get(
                    "text",
                    "",
                )
            )
            if not _noise_text(
                sentence
            )
        ]

        if not sentences:

            text = _clean_text(
                item.get(
                    "text",
                    "",
                )
            )

            if text:
                sentences = [text]

        # Score one-, two- and three-sentence
        # answer windows.
        for size in (
            1,
            2,
            3,
        ):

            for start in range(
                len(sentences)
            ):

                end = (
                    start
                    + size
                )

                if end > len(sentences):
                    break

                window = _clean_text(
                    " ".join(
                        sentences[
                            start:end
                        ]
                    )
                )

                if (
                    len(window) < 35
                    or len(window) > 1100
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

                seen.add(
                    key
                )

                section = (
                    item.get(
                        "section",
                        "",
                    )
                    or ""
                ).strip()

                if section:

                    score_text = (
                        f"Section: {section}\n"
                        f"{window}"
                    )

                else:

                    score_text = (
                        window
                    )

                candidates.append(
                    {
                        "text":
                            window,

                        "score_text":
                            score_text,

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

    retriever = (
        get_retriever()
    )

    candidates = (
        _window_candidates(
            evidence
        )
    )

    if not candidates:

        best = evidence[0]

        return (
            f"{_clean_text(best.get('text', ''))} "
            f"[Source: "
            f"{best.get('source', 'Unknown source')}, "
            f"p. {best.get('page', '?')}]"
        )

    scores = (
        retriever.score_texts(
            question,
            [
                item["score_text"]
                for item in candidates
            ],
        )
    )

    for candidate, model_score in zip(
        candidates,
        scores,
    ):

        # Sentence/window relevance dominates.
        candidate[
            "answer_score"
        ] = (
            0.84
            * float(
                model_score
            )

            + 0.16
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

    selected = []

    for candidate in candidates:

        score = float(
            candidate[
                "answer_score"
            ]
        )

        if selected:

            minimum = max(
                ANSWER_WINDOW_THRESHOLD,
                best_score * 0.72,
            )

            if score < minimum:
                continue

        duplicate = False

        for chosen in selected:

            if (
                _jaccard(
                    candidate["text"],
                    chosen["text"],
                )
                >= 0.72
            ):
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(
            candidate
        )

        if (
            len(selected)
            >= ANSWER_WINDOW_COUNT
        ):
            break

    if not selected:
        selected = [
            candidates[0]
        ]

    first = selected[0]

    lines = [
        (
            f"{first['text']} "
            f"[Source: "
            f"{first['source']}, "
            f"p. {first['page']}]"
        )
    ]

    for item in selected[1:]:

        lines.append(
            (
                f"- {item['text']} "
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]"
            )
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

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
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
                used + len(block)
                > MAX_CONTEXT_CHARS
            ):

                remaining = (
                    MAX_CONTEXT_CHARS
                    - used
                )

                if remaining > 250:
                    context_parts.append(
                        block[:remaining]
                    )

                break

            context_parts.append(
                block
            )

            used += len(block)

        prompt = (
            GENERATION_PROMPT.format(
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

            tokens = output[0]

        else:

            input_length = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            tokens = output[0][
                input_length:
            ]

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

    # Recommended mode.
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

            if _generator is not None:

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