from __future__ import annotations

import re


_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9(\[•\-])"
)


def _clean(text: str) -> str:
    return re.sub(
        r"[ \t]+",
        " ",
        text or "",
    ).strip()


def _looks_like_heading(line: str) -> bool:
    line = _clean(line)

    if not line:
        return False

    if len(line) > 120:
        return False

    words = re.findall(
        r"[A-Za-z0-9]+",
        line,
    )

    if not words:
        return False

    if len(words) > 14:
        return False

    if line.endswith(
        (
            ".",
            "?",
            "!",
            ";",
            ",",
        )
    ):
        return False

    if re.match(
        r"^\s*(?:chapter|section|part)\s+\w+",
        line,
        re.I,
    ):
        return True

    if re.match(
        r"^\s*\d+(?:\.\d+){0,4}\s+\S+",
        line,
    ):
        return True

    letters = [
        char
        for char in line
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

        if upper_ratio >= 0.72:
            return True

    alpha_words = re.findall(
        r"[A-Za-z]+",
        line,
    )

    if alpha_words:
        title_ratio = (
            sum(
                word[0].isupper()
                for word in alpha_words
            )
            / len(alpha_words)
        )

        if (
            title_ratio >= 0.80
            and len(alpha_words) <= 10
        ):
            return True

    return False


def _split_units(text: str):
    lines = [
        _clean(line)
        for line in (text or "").splitlines()
        if _clean(line)
    ]

    units = []

    current_section = ""

    for line in lines:

        if _looks_like_heading(line):
            current_section = line

            units.append(
                {
                    "text": line,
                    "section": current_section,
                    "heading": True,
                }
            )

            continue

        pieces = [
            _clean(piece)
            for piece in _SENTENCE_SPLIT.split(
                line
            )
            if _clean(piece)
        ]

        if not pieces:
            pieces = [line]

        for piece in pieces:
            units.append(
                {
                    "text": piece,
                    "section": current_section,
                    "heading": False,
                }
            )

    return units


def _split_long_text(
    text: str,
    chunk_size: int,
):
    if len(text) <= chunk_size:
        return [text]

    words = text.split()

    parts = []

    current = []
    length = 0

    for word in words:
        extra = (
            len(word)
            + (1 if current else 0)
        )

        if (
            current
            and length + extra > chunk_size
        ):
            parts.append(
                " ".join(current)
            )

            current = []
            length = 0

        current.append(
            word
        )

        length += extra

    if current:
        parts.append(
            " ".join(current)
        )

    return parts


def create_chunks(
    documents,
    chunk_size=1400,
    overlap=250,
):
    if chunk_size < 400:
        raise ValueError(
            "chunk_size must be at least 400"
        )

    if (
        overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError(
            "Invalid chunk overlap"
        )

    chunks = []

    for doc in documents:

        units = _split_units(
            doc.get(
                "text",
                "",
            )
        )

        current = []

        page_chunk = 0

        def emit(items):
            nonlocal page_chunk

            if not items:
                return

            text = " ".join(
                item["text"]
                for item in items
            ).strip()

            if not text:
                return

            section = ""

            for item in reversed(items):
                if item.get("section"):
                    section = item["section"]
                    break

            chunks.append(
                {
                    "text": text,
                    "source": doc["source"],
                    "page": doc["page"],
                    "section": section,
                    "page_chunk": page_chunk,
                    "chunk_id": len(chunks),
                }
            )

            page_chunk += 1

        for unit in units:

            pieces = _split_long_text(
                unit["text"],
                chunk_size,
            )

            for piece in pieces:

                new_item = {
                    "text": piece,
                    "section": unit.get(
                        "section",
                        "",
                    ),
                    "heading": unit.get(
                        "heading",
                        False,
                    ),
                }

                current_text = " ".join(
                    item["text"]
                    for item in current
                )

                # Strong section boundary.
                if (
                    new_item["heading"]
                    and current
                    and len(current_text)
                    >= int(
                        chunk_size * 0.45
                    )
                ):
                    emit(current)

                    current = []

                prospective = " ".join(
                    [
                        *[
                            item["text"]
                            for item in current
                        ],
                        new_item["text"],
                    ]
                )

                if (
                    current
                    and len(prospective)
                    > chunk_size
                ):
                    emit(current)

                    tail = []

                    tail_len = 0

                    for old in reversed(
                        current
                    ):
                        if old.get("heading"):
                            break

                        add = (
                            len(old["text"])
                            + (
                                1
                                if tail
                                else 0
                            )
                        )

                        if (
                            tail
                            and tail_len + add
                            > overlap
                        ):
                            break

                        tail.insert(
                            0,
                            old,
                        )

                        tail_len += add

                    current = tail

                current.append(
                    new_item
                )

        emit(current)

    return chunks