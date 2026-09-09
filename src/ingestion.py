from __future__ import annotations

from pathlib import Path
import re


def normalize_pdf_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00ad", "")
    text = text.replace("\x00", "")

    # Fix words broken by line-break hyphenation.
    text = re.sub(
        r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])",
        r"\1\2",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return text.strip()


def _load_with_pymupdf(path: Path):
    try:
        import fitz
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyMuPDF is not installed. Install it with 'pip install pymupdf' or use a PDF engine with pypdf."
        ) from exc

    pages = []

    document = fitz.open(
        str(path)
    )

    try:
        for page_no, page in enumerate(
            document,
            1,
        ):
            text = page.get_text(
                "text",
                sort=True,
            )

            text = normalize_pdf_text(
                text
            )

            if text:
                pages.append(
                    {
                        "text": text,
                        "source": path.name,
                        "page": page_no,
                    }
                )

    finally:
        document.close()

    return pages


def _load_with_pypdf(path: Path):
    from pypdf import PdfReader

    pages = []

    reader = PdfReader(
        str(path)
    )

    for page_no, page in enumerate(
        reader.pages,
        1,
    ):
        text = normalize_pdf_text(
            page.extract_text() or ""
        )

        if text:
            pages.append(
                {
                    "text": text,
                    "source": path.name,
                    "page": page_no,
                }
            )

    return pages


def load_documents(pdf_dir):
    docs = []
    errors = []

    for path in sorted(
        Path(pdf_dir).glob("*.pdf")
    ):
        try:
            try:
                pages = _load_with_pymupdf(
                    path
                )

            except Exception:
                pages = _load_with_pypdf(
                    path
                )

            docs.extend(
                pages
            )

        except Exception as exc:
            errors.append(
                f"{path.name}: {exc}"
            )

    for error in errors:
        print(
            f"[ingestion] skipped: {error}"
        )

    return docs