from __future__ import annotations

from pathlib import Path
import re


def normalize_pdf_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\u00ad",
        "",
    )

    text = text.replace(
        "\x00",
        "",
    )

    # Repair words broken across lines.
    #
    # Example:
    # trans-
    # cription
    #
    # becomes:
    # transcription
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


def _block_text(
    block,
) -> str:

    # PyMuPDF block format:
    #
    # x0, y0, x1, y1,
    # text, block_number, block_type

    if (
        len(block) >= 7
        and block[6] != 0
    ):
        return ""

    return normalize_pdf_text(
        block[4]
        if len(block) >= 5
        else ""
    )


def _ordered_block_text(
    page,
) -> str:
    """
    Layout-aware extraction.

    Single-column PDF:
        normal top-to-bottom order

    Two-column PDF:
        left column top-to-bottom
        then right column top-to-bottom

    This prevents academic journal columns from
    being mixed sentence-by-sentence.
    """

    blocks = []

    for block in page.get_text(
        "blocks"
    ):

        text = _block_text(
            block
        )

        if not text:
            continue

        blocks.append(
            {
                "x0":
                    float(
                        block[0]
                    ),

                "y0":
                    float(
                        block[1]
                    ),

                "x1":
                    float(
                        block[2]
                    ),

                "y1":
                    float(
                        block[3]
                    ),

                "text":
                    text,
            }
        )

    if not blocks:
        return ""

    page_width = float(
        page.rect.width
    )

    center = (
        page_width
        / 2.0
    )

    tolerance = (
        page_width
        * 0.045
    )

    left = []

    right = []

    wide = []

    for block in blocks:

        width = (
            block["x1"]
            - block["x0"]
        )

        if (
            width
            >= page_width
            * 0.62
        ):

            wide.append(
                block
            )

        elif (
            block["x1"]
            <= center
            + tolerance
        ):

            left.append(
                block
            )

        elif (
            block["x0"]
            >= center
            - tolerance
        ):

            right.append(
                block
            )

        else:

            wide.append(
                block
            )

    # Don't assume every page has columns.
    # Require actual evidence of both sides.
    two_column = (
        len(left) >= 2
        and len(right) >= 2
    )

    if not two_column:

        ordered = sorted(
            blocks,
            key=lambda item: (
                round(
                    item["y0"],
                    1,
                ),
                item["x0"],
            ),
        )

        return normalize_pdf_text(
            "\n".join(
                item["text"]
                for item
                in ordered
            )
        )

    first_column_y = min(
        item["y0"]
        for item
        in (
            left
            + right
        )
    )

    prefix = [
        item
        for item
        in wide
        if (
            item["y1"]
            <= first_column_y
            + 35
        )
    ]

    suffix = [
        item
        for item
        in wide
        if item
        not in prefix
    ]

    prefix.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    left.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    right.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    suffix.sort(
        key=lambda item: (
            item["y0"],
            item["x0"],
        )
    )

    ordered = [
        *prefix,
        *left,
        *right,
        *suffix,
    ]

    return normalize_pdf_text(
        "\n".join(
            item["text"]
            for item
            in ordered
        )
    )


def _load_with_pymupdf(
    path: Path,
):

    import fitz

    pages = []

    document = fitz.open(
        str(path)
    )

    try:

        for page_no, page in enumerate(
            document,
            1,
        ):

            text = (
                _ordered_block_text(
                    page
                )
            )

            if text:

                pages.append(
                    {
                        "text":
                            text,

                        "source":
                            path.name,

                        "page":
                            page_no,
                    }
                )

    finally:

        document.close()

    return pages


def _load_with_pypdf(
    path: Path,
):

    from pypdf import (
        PdfReader,
    )

    pages = []

    reader = PdfReader(
        str(path)
    )

    for page_no, page in enumerate(
        reader.pages,
        1,
    ):

        text = normalize_pdf_text(
            page.extract_text()
            or ""
        )

        if text:

            pages.append(
                {
                    "text":
                        text,

                    "source":
                        path.name,

                    "page":
                        page_no,
                }
            )

    return pages


def load_documents(
    pdf_dir,
):

    docs = []

    errors = []

    for path in sorted(
        Path(
            pdf_dir
        ).glob(
            "*.pdf"
        )
    ):

        try:

            try:

                pages = (
                    _load_with_pymupdf(
                        path
                    )
                )

            except Exception:

                pages = (
                    _load_with_pypdf(
                        path
                    )
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
            f"[ingestion] skipped: "
            f"{error}"
        )

    return docs