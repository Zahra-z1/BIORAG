from __future__ import annotations

import re
import torch

from .config import (
    ALLOW_MODEL_DOWNLOAD,
    GENERATION_BACKEND,
    GENERATION_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
)

from .prompts import (
    GENERATION_PROMPT,
)


_GENERIC = {
    "what", "which",
    "how", "why",
    "does", "do",
    "is", "are",
    "the", "a", "an",
    "of", "to", "in",
    "and", "or",
    "explain",
    "describe",
    "define",
    "about",
    "between",
}


_DEFINITION_NOUNS = {
    "process",
    "field",
    "branch",
    "science",
    "discipline",
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
    "series",
    "synthesis",
    "production",
    "formation",
    "copying",
    "transfer",
}


def _canonical(
    token: str,
):
    token = (
        token.lower()
        .strip()
    )

    if (
        token.endswith("ies")
        and len(token) > 4
    ):
        return (
            token[:-3]
            + "y"
        )

    if (
        token.endswith("s")
        and len(token) > 3
        and not token.endswith(
            "ss"
        )
    ):
        return token[:-1]

    return token


def _ordered_terms(
    text: str,
):

    generic = (
        _GENERIC
        | {
            "from",
            "difference",
            "different",
            "differ",
            "compare",
            "comparison",
            "versus",
            "vs",
            "please",
            "tell",
            "me",
            "work",
            "works",
            "working",
            "process",
            "function",
            "functions",
            "role",
        }
    )

    result = []

    for token in re.findall(
        r"[a-zA-Z]"
        r"[a-zA-Z0-9_-]{2,}",
        (text or "").lower(),
    ):

        if token in generic:
            continue

        token = _canonical(
            token
        )

        if token not in result:
            result.append(
                token
            )

    return result


def _terms(
    text: str,
):
    return set(
        _ordered_terms(
            text
        )
    )


def _sentences(
    text: str,
):

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
        if len(
            sentence.strip()
        ) >= 25
    ]


def _subject(
    question: str,
):

    q = (
        question
        .strip()
        .rstrip("?.!")
    )

    patterns = [
        (
            r"^(?:what|which)"
            r"\s+(?:is|are)"
            r"\s+(.+)$"
        ),
        (
            r"^(?:define|"
            r"describe|explain)"
            r"\s+(.+)$"
        ),
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            q,
            re.I,
        )

        if match:

            subject = (
                match.group(1)
                .strip()
            )

            return re.sub(
                r"^(?:a|an|the)\s+",
                "",
                subject,
                flags=re.I,
            )

    return q


def _subject_terms(
    subject: str,
):

    result = []

    for word in re.findall(
        r"[A-Za-z]"
        r"[A-Za-z0-9_-]{1,}",
        subject or "",
    ):

        if word.lower() in {
            "a",
            "an",
            "the",
            "of",
            "for",
            "in",
            "on",
        }:
            continue

        word = _canonical(
            word
        )

        if word not in result:
            result.append(
                word
            )

    return result


def _subject_regex(
    subject: str,
):

    terms = _subject_terms(
        subject
    )

    if not terms:
        return ""

    pieces = [
        re.escape(term)
        + r"s?"
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


def _contains_subject(
    sentence: str,
    subject: str,
):

    wanted = set(
        _subject_terms(
            subject
        )
    )

    available = {
        _canonical(word)
        for word in re.findall(
            r"[A-Za-z]"
            r"[A-Za-z0-9_-]{1,}",
            sentence or "",
        )
    }

    return bool(
        wanted
        and wanted.issubset(
            available
        )
    )


def _noise_sentence(
    sentence: str,
):

    sentence = (
        sentence.strip()
    )

    lower = (
        sentence.lower()
    )

    if (
        "www." in lower
        or "http://" in lower
        or "https://" in lower
    ):
        return True

    if lower.startswith(
        "keywords:"
    ):
        return True

    if lower.startswith(
        "keyword:"
    ):
        return True

    if (
        "questions and answers below"
        in lower
    ):
        return True

    if re.match(
        r"^\s*(?:"
        r"what|which|why|how"
        r")\b.*\?$",
        sentence,
        re.I,
    ):
        return True

    if re.match(
        r"^\s*first of all\s*:"
        r"\s*(?:what|which|why|how)",
        sentence,
        re.I,
    ):
        return True

    if len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            sentence,
        )
    ) >= 3:
        return True

    if len(
        re.findall(
            r"\[[0-9,\- ]+\]",
            sentence,
        )
    ) >= 4:
        return True

    letters = [
        char
        for char in sentence
        if char.isalpha()
    ]

    if letters:

        upper_ratio = (
            sum(
                char.isupper()
                for char
                in letters
            )
            / len(letters)
        )

        if (
            upper_ratio > 0.72
            and len(sentence) > 70
        ):
            return True

    return False


def _clean_sentence(
    sentence: str,
):

    sentence = re.sub(
        r"\s+",
        " ",
        sentence,
    ).strip()

    sentence = sentence.replace(
        " - ",
        " ",
    )

    if len(sentence) > 520:

        sentence = (
            sentence[:517]
            .rstrip()
            + "..."
        )

    return sentence


def _definition_strength(
    sentence: str,
    subject: str,
):
    """
    Detect real definitions.

    This fixes:
    - gene
    - genome
    - bioinformatics
    - transcription

    while rejecting:
    "Transcription is initiated..."
    """

    if not _contains_subject(
        sentence,
        subject,
    ):
        return 0.0

    pattern = _subject_regex(
        subject
    )

    if not pattern:
        return 0.0

    lower = re.sub(
        r"\s+",
        " ",
        sentence.lower(),
    )

    score = 0.0

    # Reverse definition:
    # "... is called a gene."
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
            lower,
            re.I,
        ):
            score = max(
                score,
                1.0,
            )

    # Explicit definition.
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
            lower,
            re.I,
        ):
            score = max(
                score,
                1.0,
            )

    copula = re.search(
        rf"(?:^|\bthe\s+)"
        rf"{pattern}\s+"
        rf"(?:is|are)\s+(.+)",
        lower,
        re.I,
    )

    if copula:

        tail = (
            copula.group(1)
            .strip()
        )

        words = tail.split()

        if words:

            first = words[0]

            # "Genome is the entirety..."
            # "Bioinformatics is a field..."
            if first in {
                "a",
                "an",
                "the",
            }:

                score = max(
                    score,
                    0.98,
                )

            elif (
                _canonical(
                    first
                )
                in _DEFINITION_NOUNS
            ):

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

            # Critical:
            # transcription is initiated
            # != definition
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

    if re.search(
        rf"(?:^|\bthe\s+)"
        rf"{pattern}\s+"
        rf"(?:"
        rf"consists\s+of|"
        rf"comprises|"
        rf"includes|"
        rf"contains"
        rf")\b",
        lower,
        re.I,
    ):

        score = max(
            score,
            0.78,
        )

    return min(
        score,
        1.0,
    )


def _definition_answer(
    question: str,
    evidence: list[dict],
):

    subject = _subject(
        question
    )

    if not subject:
        return None

    candidates = []

    for result_rank, item in enumerate(
        evidence
    ):

        base_score = float(
            item.get(
                "score",
                0.0,
            )
        )

        source_match = float(
            item.get(
                "source_match",
                0.0,
            )
        )

        for (
            sentence_rank,
            sentence,
        ) in enumerate(
            _sentences(
                item.get(
                    "text",
                    "",
                )
            )
        ):

            if _noise_sentence(
                sentence
            ):
                continue

            strength = (
                _definition_strength(
                    sentence,
                    subject,
                )
            )

            # Do NOT accept weak sentences like
            # "Transcription is initiated..."
            if strength < 0.60:
                continue

            score = (
                0.36
                * base_score
                + 0.48
                * strength
                + 0.16
                * source_match
                - 0.010
                * result_rank
                - 0.002
                * sentence_rank
            )

            if (
                35
                <= len(sentence)
                <= 420
            ):
                score += 0.04

            candidates.append(
                (
                    score,
                    sentence,
                    item,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    (
        _,
        direct_sentence,
        direct_item,
    ) = candidates[0]

    result = [
        (
            f"{_clean_sentence(direct_sentence)} "
            f"[Source: "
            f"{direct_item['source']}, "
            f"p. {direct_item['page']}]"
        )
    ]

    direct_key = re.sub(
        r"\W+",
        " ",
        direct_sentence.lower(),
    ).strip()

    support_candidates = []

    for item in evidence:

        for sentence in _sentences(
            item.get(
                "text",
                "",
            )
        ):

            if _noise_sentence(
                sentence
            ):
                continue

            key = re.sub(
                r"\W+",
                " ",
                sentence.lower(),
            ).strip()

            if key == direct_key:
                continue

            if not _contains_subject(
                sentence,
                subject,
            ):
                continue

            score = (
                0.45
                * float(
                    item.get(
                        "score",
                        0.0,
                    )
                )
                + 0.15
                * float(
                    item.get(
                        "source_match",
                        0.0,
                    )
                )
            )

            if (
                item.get(
                    "source"
                )
                == direct_item.get(
                    "source"
                )
            ):
                score += 0.20

            if (
                item.get(
                    "source"
                )
                == direct_item.get(
                    "source"
                )
                and item.get(
                    "page"
                )
                == direct_item.get(
                    "page"
                )
            ):
                score += 0.15

            if re.search(
                r"\b(?:"
                r"includes|contains|"
                r"consists|comprises|"
                r"stage|phase|function"
                r")\b",
                sentence,
                re.I,
            ):
                score += 0.10

            support_candidates.append(
                (
                    score,
                    sentence,
                    item,
                )
            )

    support_candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    seen = {
        direct_key
    }

    added = 0

    for (
        _,
        sentence,
        item,
    ) in support_candidates:

        key = re.sub(
            r"\W+",
            " ",
            sentence.lower(),
        ).strip()

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            (
                f"- {_clean_sentence(sentence)} "
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]"
            )
        )

        added += 1

        if added >= 2:
            break

    return "\n".join(
        result
    )


def _comparison_answer(
    question: str,
    evidence: list[dict],
):

    if not re.search(
        r"\b(?:"
        r"differ|difference|"
        r"compare|comparison|"
        r"versus|vs\.?"
        r")\b",
        question,
        re.I,
    ):
        return None

    terms = _ordered_terms(
        question
    )

    if len(terms) < 2:
        return None

    concepts = terms[:2]

    chosen = []

    for concept in concepts:

        candidates = []

        for item in evidence:

            for sentence in _sentences(
                item.get(
                    "text",
                    "",
                )
            ):

                if _noise_sentence(
                    sentence
                ):
                    continue

                sentence_terms = (
                    _terms(
                        sentence
                    )
                )

                if concept not in sentence_terms:
                    continue

                score = (
                    0.55
                    * float(
                        item.get(
                            "score",
                            0.0,
                        )
                    )
                    + 0.20
                    * float(
                        item.get(
                            "source_match",
                            0.0,
                        )
                    )
                )

                candidates.append(
                    (
                        score,
                        sentence,
                        item,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        chosen.append(
            (
                concept,
                candidates[0][1],
                candidates[0][2],
            )
        )

    output = [
        (
            "The indexed sources "
            "distinguish them as follows:"
        )
    ]

    for (
        concept,
        sentence,
        item,
    ) in chosen:

        label = (
            concept.upper()
            if len(concept) <= 4
            else concept.capitalize()
        )

        output.append(
            (
                f"- **{label}:** "
                f"{_clean_sentence(sentence)} "
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]"
            )
        )

    return "\n".join(
        output
    )


def _general_extractive_answer(
    question: str,
    evidence: list[dict],
):

    question_terms = (
        _terms(
            question
        )
    )

    subject = _subject(
        question
    )

    subject_terms = set(
        _subject_terms(
            subject
        )
    )

    definition_intent = bool(
        re.match(
            r"^\s*(?:"
            r"what|which"
            r")\s+(?:is|are)\b"
            r"|^\s*define\b",
            question,
            re.I,
        )
    )

    candidates = []

    seen = set()

    for result_rank, item in enumerate(
        evidence
    ):

        base_score = float(
            item.get(
                "score",
                0.0,
            )
        )

        source_match = float(
            item.get(
                "source_match",
                0.0,
            )
        )

        for (
            sentence_rank,
            sentence,
        ) in enumerate(
            _sentences(
                item.get(
                    "text",
                    "",
                )
            )
        ):

            if _noise_sentence(
                sentence
            ):
                continue

            key = re.sub(
                r"\W+",
                " ",
                sentence.lower(),
            ).strip()

            if key in seen:
                continue

            seen.add(
                key
            )

            sentence_terms = (
                _terms(
                    sentence
                )
            )

            question_overlap = (
                len(
                    question_terms
                    & sentence_terms
                )
                / max(
                    1,
                    len(
                        question_terms
                    ),
                )
            )

            subject_overlap = (
                len(
                    subject_terms
                    & sentence_terms
                )
                / max(
                    1,
                    len(
                        subject_terms
                    ),
                )
            )

            if (
                question_overlap <= 0
                and subject_overlap <= 0
            ):
                continue

            score = (
                0.32
                * base_score
                + 0.24
                * question_overlap
                + 0.18
                * subject_overlap
                + 0.26
                * source_match
                - 0.010
                * result_rank
                - 0.002
                * sentence_rank
            )

            if (
                45
                <= len(sentence)
                <= 430
            ):
                score += 0.04

            if re.search(
                r"\b(?:"
                r"primary|secondary|"
                r"tertiary|quaternary|"
                r"phase|stage|first|"
                r"then|next|finally|"
                r"begins|starts|"
                r"followed|consists|"
                r"comprises|includes|"
                r"involves|occurs|"
                r"formed|structure|"
                r"synthesis|polymerase|"
                r"strand|template"
                r")\b",
                sentence,
                re.I,
            ):
                score += 0.12

            # Definition fallback:
            # even if there is no perfect
            # "X is..." sentence, prefer
            # explanatory sentences.
            if definition_intent:

                if re.search(
                    r"\b(?:"
                    r"process|series|cycle|"
                    r"consists|includes|"
                    r"comprises|involves|"
                    r"stage|phase"
                    r")\b",
                    sentence,
                    re.I,
                ):
                    score += 0.10

            candidates.append(
                (
                    score,
                    sentence,
                    item,
                )
            )

    candidates.sort(
        key=lambda item:
            item[0],
        reverse=True,
    )

    if not candidates:

        return (
            "I found relevant documents, "
            "but could not extract a reliable "
            "answer passage from them."
        )

    top_source = (
        candidates[0][2]
        .get("source")
    )

    selected = []

    selected_keys = set()

    # Keep the answer coherent:
    # prefer one strong source.
    for (
        _,
        sentence,
        item,
    ) in candidates:

        if (
            item.get(
                "source"
            )
            != top_source
        ):
            continue

        key = re.sub(
            r"\W+",
            " ",
            sentence.lower(),
        ).strip()

        if key in selected_keys:
            continue

        selected.append(
            (
                sentence,
                item,
            )
        )

        selected_keys.add(
            key
        )

        if len(selected) >= 4:
            break

    # If strongest source has too little,
    # allow other evidence.
    if len(selected) < 2:

        for (
            _,
            sentence,
            item,
        ) in candidates:

            key = re.sub(
                r"\W+",
                " ",
                sentence.lower(),
            ).strip()

            if key in selected_keys:
                continue

            selected.append(
                (
                    sentence,
                    item,
                )
            )

            selected_keys.add(
                key
            )

            if len(selected) >= 4:
                break

    output = [
        "Based on the indexed literature:"
    ]

    for sentence, item in selected:

        output.append(
            (
                f"- {_clean_sentence(sentence)} "
                f"[Source: "
                f"{item['source']}, "
                f"p. {item['page']}]"
            )
        )

    return "\n".join(
        output
    )


def extractive_answer(
    question: str,
    evidence: list[dict],
):

    definition_question = bool(
        re.match(
            r"^\s*(?:"
            r"what|which"
            r")\s+(?:is|are)\b"
            r"|^\s*define\b",
            question,
            re.I,
        )
    )

    if definition_question:

        answer = (
            _definition_answer(
                question,
                evidence,
            )
        )

        if answer:
            return answer

    comparison = (
        _comparison_answer(
            question,
            evidence,
        )
    )

    if comparison:
        return comparison

    return (
        _general_extractive_answer(
            question,
            evidence,
        )
    )


class LocalGenerator:

    def __init__(
        self,
        model_name=GENERATION_MODEL,
    ):

        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

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
                < 100_000
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

        if getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        ) is not None:

            kwargs[
                "pad_token_id"
            ] = (
                self.tokenizer
                .pad_token_id
            )

        elif getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        ) is not None:

            kwargs[
                "pad_token_id"
            ] = (
                self.tokenizer
                .eos_token_id
            )

        with torch.inference_mode():

            output = (
                self.model.generate(
                    **inputs,
                    **kwargs,
                )
            )

        if self.is_encoder_decoder:

            generated_tokens = (
                output[0]
            )

        else:

            input_length = (
                inputs[
                    "input_ids"
                ].shape[1]
            )

            generated_tokens = (
                output[0][
                    input_length:
                ]
            )

        return (
            self.tokenizer
            .decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            .strip()
        )


_generator = None

_generator_error = None


def generate_answer(
    question: str,
    evidence: list[dict],
):

    global _generator
    global _generator_error

    if not evidence:

        return (
            (
                "I could not find enough "
                "evidence in the indexed "
                "documents to answer "
                "this question."
            ),
            "none",
        )

    if (
        GENERATION_BACKEND
        == "extractive"
    ):

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

                bad_answer = (
                    not answer
                    or len(
                        answer.split()
                    ) < 8
                    or (
                        "the relevant mechanism "
                        "or explanation"
                        in answer.lower()
                    )
                )

                if not bad_answer:

                    return (
                        answer,
                        "transformers",
                    )

        except Exception as exc:

            _generator_error = (
                str(exc)
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