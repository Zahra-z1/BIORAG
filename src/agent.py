from __future__ import annotations

from .config import TOP_K
from .generation import generate_answer
from .retrieval import get_retriever


def ask(question: str, top_k: int = TOP_K):
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please enter a biomedical or bioinformatics question.",
            "results": [],
            "confidence": 0.0,
            "answerable": False,
            "generation_backend": "none",
        }
    if len(question) > 2000:
        return {
            "answer": "Please shorten the question to under 2,000 characters.",
            "results": [],
            "confidence": 0.0,
            "answerable": False,
            "generation_backend": "none",
        }

    retriever = get_retriever()
    results = retriever.search(question, top_k=top_k)
    assessment = retriever.assess(question, results)

    if not assessment["answerable"]:
        return {
            "answer": (
                "I could not find sufficiently relevant evidence in the indexed PDFs to answer this reliably. "
                "Try rephrasing the question or add a source document that covers the topic."
            ),
            "results": results[:3],
            "confidence": assessment["confidence"],
            "answerable": False,
            "generation_backend": "none",
            "retrieval": assessment,
        }

    answer, backend = generate_answer(question, results)
    return {
        "answer": answer,
        "results": results,
        "confidence": assessment["confidence"],
        "answerable": True,
        "generation_backend": backend,
        "retrieval": assessment,
    }
