import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[•])")


def _sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def create_chunks(documents, chunk_size=1200, overlap=220):
    """Create sentence-aware chunks and keep every chunk on a single PDF page."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    for doc in documents:
        sentences = _sentences(doc["text"])
        current = []
        current_len = 0

        def emit(items):
            text = " ".join(items).strip()
            if text:
                chunks.append(
                    {
                        "text": text,
                        "source": doc["source"],
                        "page": doc["page"],
                        "chunk_id": len(chunks),
                    }
                )

        for sentence in sentences:
            # Handle unusually long extracted lines without cutting through words.
            if len(sentence) > chunk_size:
                words = sentence.split()
                pieces = []
                part = []
                length = 0
                for word in words:
                    if part and length + len(word) + 1 > chunk_size:
                        pieces.append(" ".join(part))
                        part, length = [], 0
                    part.append(word)
                    length += len(word) + 1
                if part:
                    pieces.append(" ".join(part))
            else:
                pieces = [sentence]

            for piece in pieces:
                extra = len(piece) + (1 if current else 0)
                if current and current_len + extra > chunk_size:
                    emit(current)
                    # Keep complete trailing sentences as overlap.
                    tail = []
                    tail_len = 0
                    for old in reversed(current):
                        add = len(old) + (1 if tail else 0)
                        if tail and tail_len + add > overlap:
                            break
                        tail.insert(0, old)
                        tail_len += add
                    current = tail
                    current_len = len(" ".join(current))

                current.append(piece)
                current_len = len(" ".join(current))

        emit(current)

    return chunks
