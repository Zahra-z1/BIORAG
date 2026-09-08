from pathlib import Path
import re
from pypdf import PdfReader


def normalize_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts without rewriting the source meaning."""
    if not text:
        return ""
    text = text.replace("\u00ad", "")
    # Join words split only because of a line-break hyphen.
    text = re.sub(r"([A-Za-z])-[ \t]*\n[ \t]*([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def load_documents(pdf_dir):
    docs = []
    errors = []
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, 1):
                try:
                    text = normalize_pdf_text(page.extract_text() or "")
                except Exception as exc:  # corrupted single page should not kill the KB
                    errors.append(f"{path.name} page {page_no}: {exc}")
                    continue
                if text:
                    docs.append(
                        {"text": text, "source": path.name, "page": page_no}
                    )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    for err in errors:
        print(f"[ingestion] skipped: {err}")
    return docs
