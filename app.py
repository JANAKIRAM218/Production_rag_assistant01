"""
app.py
------
Streamlit front-end for the Zyro Dynamics HR Help Desk RAG pipeline.
All RAG logic lives in rag_backend.py — this module only handles UI.
"""

from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Backend imports — no RAG logic duplicated here
# ---------------------------------------------------------------------------
from rag_backend import (
    EMBEDDING_MODEL,
    ENSEMBLE_WEIGHTS,
    FAISS_PATH,
    LLM_MODEL,
    REFUSAL_MESSAGE,
    RERANKER_MODEL,
    RERANK_THRESHOLD,
    TOP_K_CONTEXT,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
    _build_chunks,
    _build_llm,
    _build_reranker,
    answer_question,
    build_retrievers,
    build_vectorstore,
    load_documents,
    load_vectorstore,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Zyro HR Assistant", page_icon="🏢", layout="wide")

# ---------------------------------------------------------------------------
# Quick-action prompts shown on the welcome screen
# ---------------------------------------------------------------------------
QUICK_ACTIONS: list[tuple[str, str, str, str]] = [
    ("📑", "Leave Policy", "View leave entitlements", "What is the leave policy?"),
    ("🏠", "Work From Home", "WFH eligibility & rules", "How does the work-from-home policy work?"),
    ("👶", "Maternity Leave", "Maternity benefits", "What is the maternity leave policy?"),
]

# ---------------------------------------------------------------------------
# Theme — modern dark, ChatGPT / Claude style
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #030712;
            --card: #111827;
            --border: #1F2937;
            --text: #FFFFFF;
            --text-secondary: #9CA3AF;
        }

        html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

        .stApp { background-color: var(--bg); color: var(--text); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        #MainMenu, footer { visibility: hidden; }

        .block-container {
            max-width: 760px;
            margin: 0 auto;
            padding-top: 1.2rem;
            padding-bottom: 7rem;
        }

        /* ── Hero ─────────────────────────────────────────────────────── */
        .hero { text-align: center; margin: 1.5rem 0 2rem 0; }
        .hero h1 {
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0 0 0.6rem 0;
            color: var(--text);
            letter-spacing: -0.01em;
        }
        .hero p {
            color: var(--text-secondary);
            font-size: 0.98rem;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.5;
        }

        .compact-header {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text);
            padding: 0.3rem 0 1.1rem 0;
        }

        /* ── Quick-action SaaS cards ──────────────────────────────────── */
        .stButton > button {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 16px;
            padding: 1.15rem 1rem;
            font-size: 0.98rem;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #374151;
            background: #161f2e;
            transform: translateY(-3px);
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
            color: var(--text);
        }
        .stButton > button:active { transform: translateY(-1px); }

        .qa-caption {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.78rem;
            margin: 0.45rem 0 0.2rem 0;
        }

        /* ── Welcome copy ─────────────────────────────────────────────── */
        .welcome { text-align: center; margin: 2.2rem 0 0.5rem 0; color: var(--text-secondary); }
        .welcome h3 { color: var(--text); margin-bottom: 0.5rem; font-size: 1.05rem; }
        .welcome p { font-size: 0.9rem; line-height: 1.6; max-width: 540px; margin: 0 auto; }

        /* ── Chat bubbles ─────────────────────────────────────────────── */
        .chat-row { display: flex; margin: 0.45rem 0; }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.assistant { justify-content: flex-start; }

        .bubble {
            max-width: 85%;
            padding: 0.8rem 1.15rem;
            border-radius: 18px;
            line-height: 1.55;
            font-size: 0.95rem;
        }
        .bubble p { margin: 0 0 0.5rem 0; }
        .bubble p:last-child { margin-bottom: 0; }
        .bubble ul { margin: 0.2rem 0 0.5rem 1.1rem; padding: 0; }

        .bubble.user {
            background: #1F2937;
            color: var(--text);
            border-bottom-right-radius: 4px;
        }
        .bubble.assistant {
            background: var(--card);
            border: 1px solid var(--border);
            color: var(--text);
            border-bottom-left-radius: 4px;
        }
        .bubble.refusal {
            background: #1c1410;
            border: 1px solid #92400e;
            color: #fcd34d;
            border-bottom-left-radius: 4px;
        }
        .bubble.error {
            background: #1c1112;
            border: 1px solid #7f1d1d;
            color: #fca5a5;
            border-bottom-left-radius: 4px;
        }

        /* ── Compact source cards ────────────────────────────────────── */
        .sources-wrap { display: flex; justify-content: flex-start; margin: 0.1rem 0 0.7rem 0; }
        .sources-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; max-width: 85%; }
        .source-card {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #0b0f17;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.35rem 0.7rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        .source-card b { color: var(--text); font-weight: 600; }
        .source-card .dot { color: var(--border); }

        /* ── Alerts ───────────────────────────────────────────────────── */
        [data-testid="stAlert"] {
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }

        /* ── Chat input — large rounded, ChatGPT style ───────────────── */
        [data-testid="stBottomBlockContainer"] {
            background: linear-gradient(180deg, rgba(3,7,18,0) 0%, var(--bg) 35%);
        }
        [data-testid="stChatInput"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 26px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
        }
        [data-testid="stChatInput"] textarea {
            color: var(--text);
            font-size: 0.97rem;
            padding: 0.5rem 0.3rem;
        }

        .stSpinner > div { color: var(--text-secondary); }

        /* ── Responsive ───────────────────────────────────────────────── */
        @media (max-width: 640px) {
            .bubble, .sources-grid { max-width: 92%; }
            .hero h1 { font-size: 1.6rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Helper — environment check
# ---------------------------------------------------------------------------

def _env_ok() -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_missing_var_names)."""
    required = {"GROQ_API_KEY": os.getenv("GROQ_API_KEY")}
    missing = [k for k, v in required.items() if not v]
    return (len(missing) == 0), missing

# ---------------------------------------------------------------------------
# Cached pipeline loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_pipeline() -> dict[str, Any] | str:
    """Load or build the full RAG pipeline. Cached for the session lifetime.

    Returns:
        A dict with keys ``ensemble_retriever``, ``reranker``, ``llm``
        on success, or an error string on failure.
    """
    try:
        faiss_path = Path(FAISS_PATH)
        if faiss_path.exists():
            vectorstore = load_vectorstore(str(faiss_path))
            docs = load_documents()
            chunks = _build_chunks(docs)
        else:
            vectorstore, chunks = build_vectorstore()

        ensemble_retriever = build_retrievers(vectorstore, chunks)
        reranker = _build_reranker()
        llm = _build_llm()

        return {
            "ensemble_retriever": ensemble_retriever,
            "reranker": reranker,
            "llm": llm,
        }
    except FileNotFoundError as exc:
        return f"FileNotFoundError: {exc}"
    except Exception as exc:
        return f"Unexpected error during pipeline initialisation: {exc}"

# ---------------------------------------------------------------------------
# Chat bubble rendering
# ---------------------------------------------------------------------------

def _format_bubble_html(text: str) -> str:
    """Lightweight, dependency-free markdown → HTML for chat bubbles.

    Supports paragraphs, ``**bold**`` and simple ``- `` bullet lists, which
    covers the vast majority of LLM-generated answer formatting.
    """
    escaped = html.escape(text.strip())
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

    out_lines: list[str] = []
    in_list = False
    for raw_line in escaped.split("\n"):
        line = raw_line.strip()
        if line.startswith("- ") or line.startswith("• "):
            if not in_list:
                out_lines.append("<ul>")
                in_list = True
            out_lines.append(f"<li>{line[2:].strip()}</li>")
            continue
        if in_list:
            out_lines.append("</ul>")
            in_list = False
        if line:
            out_lines.append(f"<p>{line}</p>")
    if in_list:
        out_lines.append("</ul>")

    return "".join(out_lines) or f"<p>{escaped}</p>"


def _render_bubble(role: str, text: str, variant: str = "default") -> None:
    css_classes = f"bubble {role}"
    if variant != "default":
        css_classes += f" {variant}"
    st.markdown(
        f'<div class="chat-row {role}"><div class="{css_classes}">'
        f"{_format_bubble_html(text)}</div></div>",
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict[str, str]]) -> None:
    """Render source citations as compact cards — no HTML tables, no expander."""
    if not sources:
        return
    cards = "".join(
        f'<div class="source-card">📄 <b>{html.escape(str(s.get("policy_name", "Unknown Policy")))}</b>'
        f'<span class="dot">&bull;</span>Page {html.escape(str(s.get("page_number", "—")))}</div>'
        for s in sources
    )
    st.markdown(
        f'<div class="sources-wrap"><div class="sources-grid">{cards}</div></div>',
        unsafe_allow_html=True,
    )


def _render_history() -> None:
    for message in st.session_state.messages:
        role = message["role"]
        variant = "default"
        if role == "assistant":
            if message.get("is_refusal"):
                variant = "refusal"
            elif message.get("is_error"):
                variant = "error"
        _render_bubble(role, message["content"], variant)
        if role == "assistant":
            _render_sources(message.get("sources", []))

# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------

def _handle_query(query: str, pipeline: dict[str, Any]) -> None:
    """Run *query* through the RAG pipeline and append results to chat history."""
    st.session_state.messages.append({"role": "user", "content": query})
    _render_bubble("user", query)

    with st.spinner("Searching HR policies..."):
        try:
            result: dict[str, Any] = answer_question(
                query=query,
                ensemble_retriever=pipeline["ensemble_retriever"],
                reranker=pipeline["reranker"],
                llm=pipeline["llm"],
            )
        except Exception as exc:
            error_msg = (
                f"An error occurred while processing your request: {exc}\n\n"
                "Please try again or contact your system administrator."
            )
            _render_bubble("assistant", error_msg, "error")
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": [], "is_error": True}
            )
            return

    answer: str = result.get("answer", REFUSAL_MESSAGE)
    sources: list[dict[str, str]] = result.get("sources", [])
    is_refusal = answer.strip() == REFUSAL_MESSAGE.strip()

    _render_bubble("assistant", answer, "refusal" if is_refusal else "default")
    if not is_refusal:
        _render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [] if is_refusal else sources,
            "is_refusal": is_refusal,
        }
    )

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Environment check ──────────────────────────────────────────────────
    env_ok, missing_vars = _env_ok()
    if not env_ok:
        st.error(
            f"Missing required environment variable(s): **{', '.join(missing_vars)}**\n\n"
            "Please set them in your `.env` file or shell environment and restart the app."
        )
        return

    # ── Pipeline initialisation ────────────────────────────────────────────
    pipeline_or_error = _load_pipeline()
    if isinstance(pipeline_or_error, str):
        error_msg = pipeline_or_error
        if "FileNotFoundError" in error_msg and "faiss_index" in error_msg.lower():
            st.error(
                "**FAISS index not found.**\n\n"
                "The vector store has not been built yet. "
                "Run `python rag_backend.py` from the project root to index your documents, "
                "then restart this app."
            )
        elif "FileNotFoundError" in error_msg and "docs" in error_msg.lower():
            st.error(
                "**No PDF documents found.**\n\n"
                "Place your HR policy PDF files in the `docs/` directory, "
                "then restart this app."
            )
        else:
            st.error(f"**Pipeline initialisation failed.**\n\n{error_msg}")
        return

    pipeline: dict[str, Any] = pipeline_or_error
    has_history = bool(st.session_state.messages)

    # ── Header ─────────────────────────────────────────────────────────────
    header_col, clear_col = st.columns([9, 1])
    with clear_col:
        if st.button("🗑️", help="Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    chip_query: str | None = None

    if not has_history:
        st.markdown(
            """
            <div class="hero">
                <h1>🏢 Zyro HR Assistant</h1>
                <p>Ask questions about leave, benefits, work from home, onboarding
                and company policies.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(3, gap="medium")
        for col, (icon, label, caption, question) in zip(cols, QUICK_ACTIONS):
            with col:
                if st.button(f"{icon}  {label}", use_container_width=True, key=f"qa_{label}"):
                    chip_query = question
                st.markdown(f'<div class="qa-caption">{caption}</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="welcome">
                <h3>Welcome to the Zyro Dynamics HR Help Desk</h3>
                <p>Ask any question about company HR policies, including leave policy,
                WFH policy, performance reviews, benefits, notice period, and travel policy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with header_col:
            st.markdown('<div class="compact-header">🏢 Zyro HR Assistant</div>', unsafe_allow_html=True)
        _render_history()

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about Zyro Dynamics HR policies...")

    if chip_query:
        _handle_query(chip_query, pipeline)
        st.rerun()

    if user_input and user_input.strip():
        _handle_query(user_input.strip(), pipeline)
        st.rerun()


if __name__ == "__main__":
    main()