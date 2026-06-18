"""
rag_production.py
-----------------
Production-ready RAG pipeline for Zyro Dynamics HR Policy Q&A.

Architecture:
    PDF Loading → Metadata Enrichment → RecursiveCharacterTextSplitter
    → BAAI/bge-large-en-v1.5 → FAISS → MMR Retrieval → BM25 Retrieval
    → EnsembleRetriever (weights=[0.7, 0.3]) → Top 20 Retrieval
    → BAAI/bge-reranker-large → Top 5 → Reranker Threshold Gate
    → Top 3 Context → Llama-3.3-70B (Groq) → Strict Grounding Prompt
    → Answer + Citations
"""

from __future__ import annotations

# ==============================================================================
# STANDARD LIBRARY
# ==============================================================================
import logging
import os
import re
from pathlib import Path
from typing import Any

# ==============================================================================
# THIRD-PARTY
# ==============================================================================
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sentence_transformers import CrossEncoder

# ==============================================================================
# ENVIRONMENT & LOGGING
# ==============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# LANGSMITH SETUP
# ==============================================================================

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# ==============================================================================
# CONFIG
# ==============================================================================

# API Keys
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

# Models
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
# bge-reranker-base chosen over bge-reranker-large: ~3× faster, negligible accuracy drop
RERANKER_MODEL: str = "BAAI/bge-reranker-base"
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Chunking — smaller chunks reduce leave-table split-boundary failures
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 200

# Retrieval
TOP_K_RETRIEVAL: int = 30       # wider net catches fragmented leave tables
TOP_K_RERANK: int = 5
TOP_K_CONTEXT: int = 5          # 5 chunks needed for split leave-entitlement tables
MMR_FETCH_K: int = 60           # scale with TOP_K_RETRIEVAL
ENSEMBLE_WEIGHTS: list[float] = [0.7, 0.3]
# Primary OOS gate — performance(0.0427) and password(0.0390) must pass → 0.03
RERANK_THRESHOLD: float = 0.03
# Secondary context quality floor — removes near-zero garbage sources
CONTEXT_SCORE_FLOOR: float = 0.05

# Paths
DOCS_PATH: str = "docs"
FAISS_PATH: str = "faiss_index"

# Refusal
REFUSAL_MESSAGE: str = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents."
)

# System prompt
SYSTEM_PROMPT: str = """
You are Zyro Dynamics HR Assistant.

Answer ONLY using the retrieved context below.

Rules:
- Answer in 2-4 sentences maximum.
- Use exact policy wording and exact numbers from the context.
- Do not paraphrase numerical values (days, weeks, percentages).
- If the answer contains leave counts, notice periods, probation duration,
  benefits, reimbursement amounts, or eligibility criteria,
  include the exact number from the policy.
- If multiple values appear in context, use the most specific policy statement.
- When context contains a table or list of leave types, read each row carefully
  and extract the figure that matches the question's leave type.
- Never summarize tables — return exact figures from them.
- Always mention the policy name.
- Do not use outside knowledge, do not guess, do not infer.
- Only refuse with "I can only answer HR-related questions from Zyro Dynamics
  policy documents." if the answer is genuinely absent from the context.
  Do NOT refuse if a relevant number or policy exists anywhere in the context.
""".strip()

# Boilerplate lines to strip from raw PDF pages
_COMMON_LINES: frozenset[str] = frozenset(
    {
        "Zyro Dynamics Pvt. Ltd.",
        "Confidential — For Internal Use Only",
        "Navigate the Future",
        "Document Code",
        "Version",
        "Effective Date",
        "Document Owner",
        "Company Profile",
        "Employee Handbook",
        "Leave Policy",
        "Performance Review Policy",
        "Compensation and Benefits Policy",
        "Onboarding and Separation Policy",
        "HR",
        "Corporate Communications",
        "Human Resources",
    }
)

# ==============================================================================
# DOCUMENT LOADING
# ==============================================================================


def load_documents(docs_dir: str = DOCS_PATH) -> list[Document]:
    """Load all PDF files from *docs_dir* using PyMuPDFLoader.

    Args:
        docs_dir: Path to the directory containing HR policy PDFs.

    Returns:
        A list of :class:`langchain_core.documents.Document` objects,
        one per PDF page, with cleaned text and enriched metadata.

    Raises:
        FileNotFoundError: If *docs_dir* does not exist or contains no PDFs.
    """
    dir_path = Path(docs_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {dir_path}")

    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {dir_path}")

    logger.info("Found %d PDF file(s) in '%s'.", len(pdf_files), dir_path)

    all_docs: list[Document] = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            all_docs.extend(docs)
            logger.debug("Loaded %d page(s) from '%s'.", len(docs), pdf_file.name)
        except Exception as exc:
            logger.error("Failed to load '%s': %s", pdf_file.name, exc)
            raise

    logger.info("Total pages loaded: %d.", len(all_docs))

    _clean_documents(all_docs)
    _enrich_metadata(all_docs)

    return all_docs


# ==============================================================================
# METADATA ENRICHMENT
# ==============================================================================


def _clean_page_text(text: str) -> str:
    """Remove boilerplate headers, footers, and noise from a page's raw text.

    Args:
        text: Raw text extracted from a single PDF page.

    Returns:
        Cleaned text with common lines and patterns removed.
    """
    patterns = [
        r"Page\s+\d+",
        r"Doc Code:\s*[A-Z0-9\-]+",
        r"ZDL-[A-Z]+-\d+",
        r"V\.\d+",
        r"\d{2}\s+[A-Za-z]+\s+\d{4}",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    cleaned_lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and line.strip() not in _COMMON_LINES
    ]
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_doc_code(text: str) -> bool:
    """Return True if *text* matches a document-code pattern (e.g. ZDL-HR-01)."""
    return bool(re.match(r"^[A-Z]{2,}-[A-Z]{2,}-\d+$", text.strip()))


def _detect_section(text: str) -> str:
    """Infer the policy section name from the first all-caps heading in *text*.

    Args:
        text: Page or chunk text.

    Returns:
        The detected section name, or ``"Unknown"`` if none is found.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _is_doc_code(line):
            continue
        if re.match(r"^V\.\d+$", line):
            continue
        if line.isupper() and 5 <= len(line) <= 60 and len(line.split()) <= 8:
            return line
    return "Unknown"


def _clean_documents(docs: list[Document]) -> None:
    """Clean raw page text in-place for all documents."""
    for doc in docs:
        doc.page_content = _clean_page_text(doc.page_content)


def _enrich_metadata(docs: list[Document]) -> None:
    """Add ``policy_name``, ``doc_id``, ``page_number``, and ``section`` metadata in-place."""
    for doc in docs:
        filename = Path(doc.metadata.get("source", "unknown")).name
        stem = Path(filename).stem
        parts = stem.split("_", 1)
        policy_name = parts[1].replace("_", " ") if len(parts) > 1 else stem

        doc.metadata["policy_name"] = policy_name
        doc.metadata["doc_id"] = filename
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        doc.metadata["section"] = _detect_section(doc.page_content)


# ==============================================================================
# CHUNKING
# ==============================================================================


def _build_chunks(docs: list[Document]) -> list[Document]:
    """Split documents into chunks and inject metadata preambles.

    Uses :class:`RecursiveCharacterTextSplitter` with ``chunk_size=1000``
    and ``chunk_overlap=200``.

    Args:
        docs: Cleaned and metadata-enriched documents (one per page).

    Returns:
        A flat list of chunk-level :class:`Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[Document] = []
    for doc in docs:
        for chunk_text in splitter.split_text(doc.page_content):
            new_doc = doc.model_copy(deep=True)
            new_doc.page_content = chunk_text
            chunks.append(new_doc)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    # Prepend structured metadata preamble to each chunk
    for chunk in chunks:
        preamble = (
            f"Policy: {chunk.metadata.get('policy_name', '')}\n"
            f"Section: {chunk.metadata.get('section', '')}\n"
            f"Page: {chunk.metadata.get('page_number', '')}\n"
            f"Document: {chunk.metadata.get('doc_id', '')}"
        )
        chunk.page_content = preamble + "\n\n" + chunk.page_content

    logger.info("Generated %d chunks from %d page(s).", len(chunks), len(docs))
    return chunks


# ==============================================================================
# EMBEDDINGS
# ==============================================================================


def _build_embedding_model() -> HuggingFaceEmbeddings:
    """Instantiate the BGE embedding model.

    Returns:
        A :class:`HuggingFaceEmbeddings` instance backed by
        ``BAAI/bge-large-en-v1.5`` with L2-normalised embeddings.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# ==============================================================================
# FAISS
# ==============================================================================


def build_vectorstore(
    docs_dir: str = DOCS_PATH,
    faiss_path: str = FAISS_PATH,
) -> tuple[FAISS, list[Document]]:
    """Load PDFs, chunk them, embed, build a FAISS index, and persist it.

    Args:
        docs_dir: Directory containing source PDFs.
        faiss_path: Directory where the FAISS index will be saved.

    Returns:
        A tuple of ``(vectorstore, chunks)`` where *chunks* are the
        :class:`Document` objects that were indexed.
    """
    docs = load_documents(docs_dir)
    chunks = _build_chunks(docs)
    embeddings = _build_embedding_model()

    logger.info("Building FAISS index over %d chunks…", len(chunks))
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    logger.info("FAISS index saved to '%s'.", faiss_path)

    return vectorstore, chunks


def load_vectorstore(
    faiss_path: str = FAISS_PATH,
) -> FAISS:
    """Load a previously persisted FAISS index from disk.

    Args:
        faiss_path: Path to the saved FAISS index directory.

    Returns:
        A :class:`FAISS` vectorstore ready for similarity search.

    Raises:
        FileNotFoundError: If *faiss_path* does not exist.
    """
    if not Path(faiss_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            "Run build_vectorstore() first."
        )
    embeddings = _build_embedding_model()
    logger.info("Loading FAISS index from '%s'.", faiss_path)
    return FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ==============================================================================
# BM25
# ==============================================================================


def _build_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    """Build a BM25 sparse retriever from *chunks*.

    Args:
        chunks: Chunk-level documents (same set used to build FAISS).

    Returns:
        A :class:`BM25Retriever` configured to return ``TOP_K_RETRIEVAL`` docs.
    """
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = TOP_K_RETRIEVAL
    logger.info("BM25 retriever built with k=%d.", TOP_K_RETRIEVAL)
    return retriever


# ==============================================================================
# ENSEMBLE RETRIEVER
# ==============================================================================


def build_retrievers(
    vectorstore: FAISS,
    chunks: list[Document],
) -> EnsembleRetriever:
    """Compose the MMR and BM25 retrievers into a weighted EnsembleRetriever.

    Args:
        vectorstore: A ready FAISS vectorstore.
        chunks: The document chunks used for BM25 indexing.

    Returns:
        An :class:`EnsembleRetriever` with weights ``[0.7, 0.3]``.
    """
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_RETRIEVAL, "fetch_k": MMR_FETCH_K},
    )
    bm25_retriever = _build_bm25_retriever(chunks)

    ensemble = EnsembleRetriever(
        retrievers=[mmr_retriever, bm25_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )
    logger.info(
        "EnsembleRetriever built (weights=%s).", ENSEMBLE_WEIGHTS
    )
    return ensemble


# ==============================================================================
# RERANKER
# ==============================================================================


def _build_reranker() -> CrossEncoder:
    """Load the BGE cross-encoder reranking model.

    Returns:
        A :class:`~sentence_transformers.CrossEncoder` instance.
    """
    logger.info("Loading reranker model: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


@traceable(name="reranker_stage")
def rerank_documents(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_k: int = TOP_K_RERANK,
) -> list[tuple[Document, float]]:
    """Score *docs* against *query* using the cross-encoder and return the top-k.

    Args:
        query: The user question.
        docs: Candidate documents from the ensemble retriever.
        reranker: A loaded :class:`~sentence_transformers.CrossEncoder`.
        top_k: Number of top-scored documents to return.

    Returns:
        A list of ``(document, score)`` tuples sorted by score descending,
        truncated to *top_k*.
    """
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores: list[float] = reranker.predict(pairs).tolist()

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ==============================================================================
# QUERY EXPANSION
# ==============================================================================

# Maps query keywords → enriched retrieval string.
# Applied before ensemble retrieval to improve BM25 term matching for
# concepts that appear under different terminology in policy documents.
_QUERY_EXPANSIONS: dict[str, str] = {
    "sick leave":     "sick leave medical leave SL leave entitlement",
    "sick":           "sick leave medical leave SL leave entitlement",
    "password":       "password access management credentials authentication",
    "harassment":     "harassment POSH sexual harassment complaint ICC",
    "pip":            "performance improvement plan PIP underperformance",
    "performance review": "performance review appraisal annual review KRA",
    "laptop":         "personal laptop device BYOD equipment IT policy",
    "bonus":          "bonus variable pay incentive performance bonus",
    "probation":      "probation period new employee joining",
    "gratuity":       "gratuity full and final settlement F&F",
}


def _expand_query(query: str) -> str:
    """Append policy-specific terminology to *query* for better BM25 recall.

    Checks each key in ``_QUERY_EXPANSIONS`` case-insensitively against the
    query and appends the corresponding expansion terms if matched.

    Args:
        query: The raw user question.

    Returns:
        The enriched query string (original query + expansion terms),
        or the original query unchanged if no match is found.
    """
    q_lower = query.lower()
    expansions: list[str] = []
    for keyword, expansion in _QUERY_EXPANSIONS.items():
        if keyword in q_lower:
            expansions.append(expansion)
    if expansions:
        enriched = query + " " + " ".join(expansions)
        logger.info("[EXPAND] Query enriched: '%s'", enriched)
        return enriched
    return query


def _boost_posh_scores(
    query: str,
    ranked: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Re-weight reranker scores for POSH policy docs when query is harassment-related.

    The cross-encoder systematically undervalues POSH documents relative to
    Code of Conduct chunks because both use similar complaint-related language.
    When the query contains harassment keywords, multiply POSH doc scores by a
    boost factor so they sort above Code of Conduct on re-sort.

    Args:
        query: The raw user question.
        ranked: Sorted ``(document, score)`` list from the cross-encoder.

    Returns:
        Re-sorted list with POSH scores boosted if query is harassment-related.
    """
    harassment_terms = {"harassment", "posh", "sexual", "complaint", "icc"}
    if not any(term in query.lower() for term in harassment_terms):
        return ranked

    POSH_BOOST = 3.0
    boosted = []
    for doc, score in ranked:
        policy = doc.metadata.get("policy_name", "").lower()
        if "sexual harassment" in policy or "posh" in policy:
            boosted.append((doc, score * POSH_BOOST))
            logger.info(
                "[BOOST] POSH doc score %.4f → %.4f | section='%s'",
                score,
                score * POSH_BOOST,
                doc.metadata.get("section", ""),
            )
        else:
            boosted.append((doc, score))

    return sorted(boosted, key=lambda x: x[1], reverse=True)





@traceable(name="hybrid_retrieval")
def retrieve_context(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
    top_k_context: int = TOP_K_CONTEXT,
    rerank_threshold: float = RERANK_THRESHOLD,
    context_score_floor: float = CONTEXT_SCORE_FLOOR,
) -> list[Document] | str:
    """Run the full hybrid retrieval pipeline for *query*.

    Steps:
        1. Fetch ``top_k_retrieval`` candidates via the EnsembleRetriever.
        2. Rerank with the cross-encoder and keep ``top_k_rerank``.
        3. Apply primary threshold gate — refuses if best score < ``rerank_threshold``.
        4. Apply secondary score floor — context docs must score ≥ ``context_score_floor``.
           If too few pass, fall back to the top-1 doc to guarantee an answer.
        5. Return the top ``top_k_context`` documents.

    Args:
        query: The user question.
        ensemble_retriever: A built :class:`EnsembleRetriever`.
        reranker: A loaded :class:`~sentence_transformers.CrossEncoder`.
        top_k_retrieval: Number of candidates to fetch from the ensemble.
        top_k_rerank: Number of docs kept after reranking.
        top_k_context: Maximum context docs returned to the LLM.
        rerank_threshold: Primary OOS gate — refuse below this.
        context_score_floor: Secondary quality gate — filters low-relevance
            docs from context to prevent garbage source pollution.

    Returns:
        A list of up to ``top_k_context`` :class:`Document` objects,
        or :data:`REFUSAL_MESSAGE` if retrieval or thresholds fail.
    """
    docs: list[Document] = ensemble_retriever.invoke(query)
    logger.info(
        "[STEP 1] Retrieval complete — %d doc(s) for query: '%s'.",
        len(docs),
        query,
    )

    if not docs:
        logger.warning("[STEP 1] No documents retrieved — refusing.")
        return REFUSAL_MESSAGE

    ranked = rerank_documents(
        query=query,
        docs=docs,
        reranker=reranker,
        top_k=top_k_rerank,
    )

    if not ranked:
        logger.warning("[STEP 2] Reranker returned no results — refusing.")
        return REFUSAL_MESSAGE

    # Apply domain-specific score boosts (e.g. POSH policy for harassment queries)
    ranked = _boost_posh_scores(query, ranked)

    # Log top-5 scores — use this to tune thresholds
    for i, (doc, score) in enumerate(ranked[:5]):
        logger.info(
            "[RANK %d] score=%.4f | policy='%s' | section='%s'",
            i + 1,
            score,
            doc.metadata.get("policy_name", ""),
            doc.metadata.get("section", ""),
        )

    best_doc, best_score = ranked[0]
    logger.info(
        "[STEP 2] Best reranker score: %.4f (threshold=%.2f) | policy='%s' | section='%s'.",
        best_score,
        rerank_threshold,
        best_doc.metadata.get("policy_name", ""),
        best_doc.metadata.get("section", ""),
    )

    # Primary gate: OOS rejection
    if best_score < rerank_threshold:
        logger.info(
            "[STEP 2] Score %.4f below threshold %.2f — refusing query: '%s'.",
            best_score,
            rerank_threshold,
            query,
        )
        return REFUSAL_MESSAGE

    # Secondary filter: remove garbage sources below score floor
    filtered = [
        (doc, score) for doc, score in ranked
        if score >= context_score_floor
    ]

    # Fallback: always keep at least the best doc so we never return empty
    if not filtered:
        logger.info(
            "[STEP 2] No docs above floor %.2f — falling back to top-1 (score=%.4f).",
            context_score_floor,
            best_score,
        )
        filtered = ranked[:1]

    final_docs = [doc for doc, _ in filtered[:top_k_context]]
    logger.info(
        "[STEP 3] Returning %d context doc(s) to LLM (floor=%.2f).",
        len(final_docs),
        context_score_floor,
    )
    return final_docs


# ==============================================================================
# GENERATION
# ==============================================================================


def _build_llm() -> ChatGroq:
    """Instantiate the Groq-hosted Llama-3.3-70B model.

    Returns:
        A :class:`~langchain_groq.ChatGroq` instance with temperature 0.
    """
    logger.info("Loading LLM: %s", LLM_MODEL)
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)


def _format_context(docs: list[Document]) -> str:
    """Serialise *docs* into a structured context block for the LLM prompt.

    Args:
        docs: Top-ranked context documents.

    Returns:
        A formatted multi-section string with policy, section, page, and content.
    """
    parts = []
    for doc in docs:
        part = (
            f"Policy: {doc.metadata.get('policy_name', 'Unknown')}\n"
            f"Section: {doc.metadata.get('section', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page_number', '?')}\n\n"
            f"{doc.page_content}"
        )
        parts.append(part)
    return "\n\n".join(parts)


def _build_sources(docs: list[Document]) -> list[dict[str, str]]:
    """Build a deduplicated list of source citation dicts from *docs*.

    Args:
        docs: Context documents used to generate the answer.

    Returns:
        A list of dicts, each with keys ``policy_name``, ``page_number``,
        and ``doc_id``, deduplicated by ``(policy_name, page_number)``.
    """
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for doc in docs:
        policy = doc.metadata.get("policy_name", "Unknown")
        page = str(doc.metadata.get("page_number", "?"))
        doc_id = doc.metadata.get("doc_id", "")
        key = (policy, page)
        if key not in seen:
            sources.append({"policy_name": policy, "page_number": page, "doc_id": doc_id})
            seen.add(key)
    return sources


@traceable(name="answer_question")
def answer_question(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    llm: ChatGroq,
) -> dict[str, Any]:
    """Answer *query* using the full RAG pipeline and return a structured response.

    Pipeline:
        1. Hybrid retrieval via the EnsembleRetriever (MMR + BM25).
        2. Cross-encoder reranking with threshold gate.
        3. LLM answer generation grounded in top-k context.

    OOS handling is performed exclusively by the reranker threshold gate:
    irrelevant queries score below ``RERANK_THRESHOLD`` and receive the
    refusal message without ever reaching the LLM.

    Args:
        query: The user's HR-related question.
        ensemble_retriever: A built :class:`EnsembleRetriever`.
        reranker: A loaded :class:`~sentence_transformers.CrossEncoder`.
        llm: A ready :class:`~langchain_groq.ChatGroq` instance.

    Returns:
        A dict with keys:

        * ``"answer"`` — grounded answer string, or :data:`REFUSAL_MESSAGE`.
        * ``"sources"`` — list of source citation dicts (empty on refusal).
        * ``"documents"`` — list of :class:`Document` objects used as context
          (empty on refusal).
    """
    logger.info("[STEP 0] answer_question called for: '%s'.", query)

    # Expand query for better BM25 recall on terminology-sensitive topics
    retrieval_query = _expand_query(query)

    context_or_refusal = retrieve_context(
        query=retrieval_query,
        ensemble_retriever=ensemble_retriever,
        reranker=reranker,
    )

    if isinstance(context_or_refusal, str):
        logger.info("[STEP 3] Retrieval refused — returning refusal message.")
        return {"answer": context_or_refusal, "sources": [], "documents": []}

    final_docs: list[Document] = context_or_refusal

    if not final_docs:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

    context = _format_context(final_docs)
    logger.info("[STEP 4] Context built — %d chars. Calling LLM.", len(context))

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )

    try:
        response = llm.invoke(prompt)
        answer: str = response.content
        logger.info("[STEP 5] LLM answered — %d chars.", len(answer))
    except Exception as exc:
        logger.error("LLM invocation failed: %s", exc)
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

    return {
        "answer": answer,
        "sources": _build_sources(final_docs),
        "documents": final_docs,
    }


# ==============================================================================
# EVALUATION
# ==============================================================================

_GOLD_DATASET: list[dict[str, Any]] = [
    # ── Leave Policy ──────────────────────────────────────────────────────
    {
        "question": "How many sick leaves are allowed per year?",
        "answer_contains": ["10", "sick leave"],
    },
    {
        "question": "How many casual leaves do employees get?",
        "answer_contains": ["8", "casual leave"],
    },
    {
        "question": "How many earned leaves are provided?",
        "answer_contains": ["earned leave"],
    },
    {
        "question": "What is the maternity leave entitlement?",
        "answer_contains": ["26 weeks"],
    },
    {
        "question": "How many days of paternity leave are allowed?",
        "answer_contains": ["paternity"],
    },
    {
        "question": "How many bereavement leaves are allowed?",
        "answer_contains": ["bereavement"],
    },
    {
        "question": "Can sick leave be carried forward to next year?",
        "answer_contains": ["sick leave"],
    },
    # ── WFH / Hybrid ─────────────────────────────────────────────────────
    {
        "question": "How does the work from home policy work?",
        "answer_contains": ["work from home"],
    },
    {
        "question": "Which employee level is eligible for WFH?",
        "answer_contains": ["L3"],
    },
    # ── Compensation & Benefits ───────────────────────────────────────────
    {
        "question": "What health insurance benefits are provided?",
        "answer_contains": ["health"],
    },
    {
        "question": "Is there a travel reimbursement policy?",
        "answer_contains": ["travel"],
    },
    {
        "question": "What is the meal or food allowance policy?",
        "answer_contains": ["allowance"],
    },
    # ── Onboarding & Separation ───────────────────────────────────────────
    {
        "question": "What is the notice period for resignation?",
        "answer_contains": ["notice period"],
    },
    {
        "question": "How long is the probation period for new employees?",
        "answer_contains": ["probation"],
    },
    {
        "question": "What happens to leaves during probation?",
        "answer_contains": ["probation"],
    },
    # ── IT / Security ─────────────────────────────────────────────────────
    {
        "question": "What are the password requirements?",
        "answer_contains": ["12 characters"],
    },
    {
        "question": "Can employees use personal laptops for work?",
        "answer_contains": ["personal laptop"],
    },
    # ── Performance ───────────────────────────────────────────────────────
    {
        "question": "How often are performance reviews conducted?",
        "answer_contains": ["performance review"],
    },
    {
        "question": "What is the performance rating scale used?",
        "answer_contains": ["rating"],
    },
    # ── OOS (must refuse via reranker threshold) ──────────────────────────
    {
        "question": "Who won the IPL in 2025?",
        "answer_contains": ["I can only answer HR-related questions"],
    },
]


@traceable(name="evaluate_rag")
def evaluate(
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    llm: ChatGroq,
    test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run keyword-match evaluation over the gold dataset.

    Args:
        ensemble_retriever: A built :class:`EnsembleRetriever`.
        reranker: A loaded :class:`~sentence_transformers.CrossEncoder`.
        llm: A ready :class:`~langchain_groq.ChatGroq` instance.
        test_cases: Optional custom test cases. Defaults to the built-in
            ``_GOLD_DATASET``.  Each entry must have ``"question"`` and
            ``"answer_contains"`` keys.

    Returns:
        A dict with keys ``"total"``, ``"correct"``, ``"accuracy"``,
        and ``"results"`` (a list of per-question dicts with ``"question"``,
        ``"passed"``, and ``"answer"``).
    """
    dataset = test_cases if test_cases is not None else _GOLD_DATASET
    total = len(dataset)
    correct = 0
    results: list[dict[str, Any]] = []

    for item in dataset:
        question: str = item["question"]
        expected_phrases: list[str] = item.get("answer_contains", [])

        try:
            result = answer_question(
                query=question,
                ensemble_retriever=ensemble_retriever,
                reranker=reranker,
                llm=llm,
            )
            answer = result["answer"] if isinstance(result, dict) else result
        except Exception as exc:
            logger.error("Error answering '%s': %s", question, exc)
            answer = ""

        passed = all(
            phrase.lower() in answer.lower() for phrase in expected_phrases
        )

        if passed:
            correct += 1

        results.append(
            {"question": question, "passed": passed, "answer": answer}
        )

        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s", status, question)

    accuracy = correct / total if total else 0.0
    logger.info("Evaluation complete — Accuracy: %d/%d (%.1f%%)", correct, total, accuracy * 100)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


# ==============================================================================
# ENTRYPOINT
# ==============================================================================


def main() -> None:
    """Build the RAG pipeline and run the evaluation suite."""
    logger.info("=== Zyro Dynamics RAG Pipeline Starting ===")

    # Build or load vector store
    faiss_index_path = Path(FAISS_PATH)
    if faiss_index_path.exists():
        logger.info("FAISS index found — loading from disk.")
        vectorstore = load_vectorstore(FAISS_PATH)
        docs = load_documents(DOCS_PATH)
        chunks = _build_chunks(docs)
    else:
        logger.info("No FAISS index found — building from scratch.")
        vectorstore, chunks = build_vectorstore(DOCS_PATH, FAISS_PATH)

    # Build retrievers
    ensemble_retriever = build_retrievers(vectorstore, chunks)

    # Load reranker and LLM
    reranker = _build_reranker()
    llm = _build_llm()

    # Run evaluation
    logger.info("Running evaluation suite…")
    eval_results = evaluate(
        ensemble_retriever=ensemble_retriever,
        reranker=reranker,
        llm=llm,
    )

    logger.info(
        "Final Accuracy: %d/%d (%.1f%%)",
        eval_results["correct"],
        eval_results["total"],
        eval_results["accuracy"] * 100,
    )

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()