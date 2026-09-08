from __future__ import annotations

from datetime import datetime
import hashlib
import json
import uuid

import streamlit as st

from src.agent import ask
from src.config import FAISS_INDEX_PATH, METADATA_PATH, PAPERS_DIR

st.set_page_config(page_title="BIO RAG", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
html, body, .stApp { background:#0E1117; color:#E6EDE9; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] { background:#0E1117; }
header[data-testid="stHeader"] { background:#0E1117; }
section[data-testid="stSidebar"] { background:#0E1117; border-right:1px solid #252B2A; }
section[data-testid="stSidebar"] * { color:#DCE5E0; }
section[data-testid="stSidebar"] p, .stCaption { color:#8F9C96 !important; }
.stButton > button, .stDownloadButton > button {
  background:#161B1A; color:#DCE5E0; border:1px solid #292F2D; border-radius:10px; box-shadow:none;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#4FAF78; color:white; background:#1A211F; }
[data-testid="stChatInput"] > div { background:#161B1A; border:1px solid #292F2D; border-radius:16px; }
[data-testid="stChatInput"] textarea { color:#F2F6F4; caret-color:#4FAF78; }
[data-testid="stExpander"] { background:#161B1A; border:1px solid #292F2D; border-radius:10px; }
.hero { text-align:center; padding:55px 10px 28px; }
.hero h1 { font-size:58px; margin:0; letter-spacing:-2px; color:#F2F6F4; }
.hero h1 span { color:#4FAF78; }
.hero p { color:#8F9C96; font-size:16px; }
.brand { font-size:25px; font-weight:800; color:#F2F6F4; letter-spacing:-.6px; }
.confidence { color:#8F9C96; font-size:12px; margin-top:4px; }
#MainMenu, footer { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def _new_id():
    return uuid.uuid4().hex


def _unique_chat_name(base: str):
    base = base.strip() or "New conversation"
    if base not in st.session_state.chats:
        return base
    i = 2
    while f"{base} ({i})" in st.session_state.chats:
        i += 1
    return f"{base} ({i})"


def _pdf_bytes(source: str):
    path = PAPERS_DIR / source
    return path.read_bytes() if path.exists() else None


def _source_key(message_id: str, index: int, source: str, page):
    raw = f"{message_id}|{index}|{source}|{page}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def source_block(results, message_id: str):
    if not results:
        return
    st.markdown("#### 📚 Sources")
    for i, item in enumerate(results, 1):
        source = item.get("source", "Unknown source")
        page = item.get("page", "?")
        score = float(item.get("score", 0.0))
        with st.expander(f"📄 {source} · Page {page} · Relevance {score:.3f}"):
            st.caption(
                f"Combined relevance {score:.4f} · lexical {float(item.get('lexical_score', 0)):.4f} · "
                f"semantic {float(item.get('semantic_score', 0)):.4f}"
            )
            st.markdown(item.get("text", ""))
            data = _pdf_bytes(source)
            if data:
                st.download_button(
                    "⬇ Download PDF",
                    data=data,
                    file_name=source,
                    mime="application/pdf",
                    key=f"pdf_{_source_key(message_id, i, source, page)}",
                )


if "chats" not in st.session_state:
    st.session_state.chats = {"New conversation": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New conversation"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def current_messages():
    if st.session_state.current_chat not in st.session_state.chats:
        st.session_state.current_chat = next(iter(st.session_state.chats))
    return st.session_state.chats[st.session_state.current_chat]


def new_chat():
    name = _unique_chat_name("New conversation")
    st.session_state.chats[name] = []
    st.session_state.current_chat = name


def delete_current_chat():
    st.session_state.chats.pop(st.session_state.current_chat, None)
    if not st.session_state.chats:
        st.session_state.chats["New conversation"] = []
    st.session_state.current_chat = next(iter(st.session_state.chats))


with st.sidebar:
    st.markdown('<div class="brand">🧬 BIO RAG</div>', unsafe_allow_html=True)
    st.caption("Biomedical & bioinformatics research assistant")
    st.divider()

    if st.button("＋ New chat", use_container_width=True):
        new_chat(); st.rerun()

    st.markdown("### Conversations")
    for chat_name in list(st.session_state.chats):
        prefix = "● " if chat_name == st.session_state.current_chat else ""
        if st.button(prefix + chat_name, key=f"chat_{hashlib.sha1(chat_name.encode()).hexdigest()[:10]}", use_container_width=True):
            st.session_state.current_chat = chat_name; st.rerun()

    st.divider()
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    st.markdown("### Knowledge base")
    st.caption(f"📚 {len(pdf_files)} PDF documents")
    if METADATA_PATH.exists():
        st.success("Retrieval index ready", icon="🟢")
        st.caption("Semantic FAISS: ready" if FAISS_INDEX_PATH.exists() else "Semantic FAISS: optional / not built")
    else:
        st.warning("Index missing. Run `python -m scripts.build_index --lexical-only`.")

    st.divider()
    show_sources = st.toggle("Show sources", value=True)
    show_retrieved = st.toggle("Show retrieved content", value=False)

    st.divider()
    if st.button("🗑 Delete current chat", use_container_width=True):
        delete_current_chat(); st.rerun()

    if current_messages():
        export_data = json.dumps(current_messages(), indent=2, ensure_ascii=False)
        st.download_button(
            "⬇ Export conversation",
            data=export_data,
            file_name=f"bio_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown(f"**{st.session_state.current_chat}**")
st.caption("Answers are restricted to evidence in the indexed PDFs.")
st.divider()

messages = current_messages()
if not messages:
    st.markdown(
        '<div class="hero"><div style="font-size:50px">🧬</div><h1>BIO <span>RAG</span></h1>'
        '<p>Ask evidence-grounded questions about your biomedical and bioinformatics literature.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    suggestions = [
        (c1, "🧬 What is a gene?", "What is a gene?"),
        (c2, "🔬 What is bioinformatics?", "What is bioinformatics?"),
        (c1, "🧪 Explain protein structure", "Explain the structure of proteins."),
        (c2, "🧬 Explain DNA replication", "Explain DNA replication."),
    ]
    for col, label, q in suggestions:
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.pending_question = q; st.rerun()
else:
    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                if message.get("confidence") is not None:
                    st.caption(
                        f"Retrieval confidence: {float(message.get('confidence', 0)):.2f} · "
                        f"Answer mode: {message.get('generation_backend', 'unknown')}"
                    )
                results = message.get("results", [])
                message_id = message.get("id", _new_id())
                if show_sources and results:
                    source_block(results, message_id)
                if show_retrieved and results:
                    st.markdown("#### 🔎 Retrieved content")
                    for item in results:
                        st.markdown(f"**{item.get('source')} — page {item.get('page')}**")
                        st.markdown(item.get("text", ""))
                        st.divider()

question = st.chat_input("Ask BIO RAG a biomedical question...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question and question.strip():
    question = question.strip()
    if st.session_state.current_chat.startswith("New conversation") and not current_messages():
        base = question[:35] + ("..." if len(question) > 35 else "")
        new_name = _unique_chat_name(base)
        old = st.session_state.current_chat
        st.session_state.chats[new_name] = st.session_state.chats.pop(old)
        st.session_state.current_chat = new_name

    st.session_state.chats[st.session_state.current_chat].append(
        {"id": _new_id(), "role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the biomedical knowledge base..."):
                result = ask(question)
            answer = result.get("answer", "I could not generate an answer.")
        except Exception as exc:
            result = {"results": [], "confidence": 0.0, "generation_backend": "error"}
            answer = f"The BIO RAG backend encountered an error: `{exc}`"
        st.markdown(answer)
        st.caption(
            f"Retrieval confidence: {float(result.get('confidence', 0)):.2f} · "
            f"Answer mode: {result.get('generation_backend', 'unknown')}"
        )
        message_id = _new_id()
        if show_sources and result.get("results"):
            source_block(result["results"], message_id)

    st.session_state.chats[st.session_state.current_chat].append(
        {
            "id": message_id,
            "role": "assistant",
            "content": answer,
            "results": result.get("results", []),
            "confidence": result.get("confidence", 0.0),
            "generation_backend": result.get("generation_backend", "unknown"),
        }
    )
    st.rerun()
