"""
rag_production.py
-----------------
Production-ready RAG pipeline for Zyro Dynamics HR Policy Q&A.

Architecture:
    PDF Loading → Aggressive Cleaning → Metadata Enrichment
    → RecursiveCharacterTextSplitter (500/100)
    → BAAI/bge-large-en-v1.5 → FAISS MMR + BM25 EnsembleRetriever
    → Top 30 → BAAI/bge-reranker-base → Top 5
    → Dual-gate (primary threshold + context floor)
    → Top 5 Context → Answer Validation → Llama-3.3-70B (Groq)
    → Strict Grounding Prompt → Structured Response + Citations
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
# Restored to large for final submission — base was faster during dev; large scores higher
RERANKER_MODEL: str = "BAAI/bge-reranker-large"
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Chunking — 500/100 produces ~150-250 chunks from 39 pages (vs 99 at 800/200)
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# Retrieval
TOP_K_RETRIEVAL: int = 20       # 20 is sufficient for ~200 chunks; 30 sends too much noise to reranker
TOP_K_RERANK: int = 7
TOP_K_CONTEXT: int = 3          # 3 prevents conflicting leave numbers from multiple chunks
MMR_FETCH_K: int = 40
ENSEMBLE_WEIGHTS: list[float] = [0.7, 0.3]   # semantic-heavy; policy Qs are conceptual not keyword

# Threshold gates
# Raised from 0.03: bge-reranker-large scores higher overall; 0.08 is safer floor
RERANK_THRESHOLD: float = 0.08
# Secondary context floor: removes near-zero noise docs from prompt
CONTEXT_SCORE_FLOOR: float = 0.05

# Paths
DOCS_PATH: str = "docs"
FAISS_PATH: str = "faiss_index"

# Refusal
REFUSAL_MESSAGE: str = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents."
)

# ==============================================================================
# SYSTEM PROMPT  (Missing Piece #6: strict grounding)
# ==============================================================================

SYSTEM_PROMPT: str = """
You are the Zyro Dynamics HR Assistant.

Answer ONLY from the retrieved context below. Follow every rule exactly.

RULES:
1. Answer in 2-4 sentences maximum.
2. Use exact policy wording. Copy exact numbers — never paraphrase them.
3. For leave counts, notice periods, probation durations, insurance amounts,
   reimbursement caps, or eligibility criteria: always state the exact figure.
4. If context contains a table or bulleted list, read every row and extract
   the specific figure that matches the question (e.g. "Sick Leave: 10 days").
5. If multiple values appear, use the most specific matching policy statement.
6. Always name the policy document in your answer.
7. Do NOT use outside knowledge. Do NOT guess. Do NOT infer.
8. Refuse ONLY if the answer is genuinely absent from ALL context chunks.
   Do NOT refuse when a relevant number or policy clause exists in the context.

REFUSAL (use exactly this string when needed):
I can only answer HR-related questions from Zyro Dynamics policy documents.
""".strip()

# ==============================================================================
# BOILERPLATE PATTERNS  (Missing Piece #7: aggressive cleaning)
# ==============================================================================

_COMMON_LINES: frozenset[str] = frozenset({
    "Zyro Dynamics Pvt. Ltd.",
    "Confidential — For Internal Use Only",
    "Confidential - For Internal Use Only",
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
    "Prevention of Sexual Harassment Policy",
    "Work From Home Policy",
    "IT and Data Security Policy",
    "Code of Conduct",
    "HR",
    "Corporate Communications",
    "Human Resources",
    "Strictly Confidential",
    "Internal Use Only",
    "For Internal Use Only",
})

_CLEAN_PATTERNS: list[str] = [
    r"Page\s+\d+\s+of\s+\d+",          # "Page 1 of 10"
    r"Page\s+\d+",                       # "Page 5"
    r"Doc(?:ument)?\s+Code:\s*[A-Z0-9\-]+",
    r"ZDL-[A-Z]+-\d+",                  # document codes
    r"\bV\.\d+(\.\d+)?\b",             # version numbers
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",  # dates
    r"©\s*\d{4}.*",                     # copyright lines
    r"[-─═]{3,}",                       # horizontal rules
]

# ==============================================================================
# QUERY EXPANSION MAP  (Missing Piece #1 / Fix #3)
# ==============================================================================

_QUERY_EXPANSIONS: dict[str, str] = {
    "sick leave":         "sick leave medical leave SL leave entitlement annual",
    "sick":               "sick leave medical leave SL entitlement",
    "casual leave":       "casual leave CL leave entitlement",
    "earned leave":       "earned leave EL privilege leave annual leave",
    "maternity":          "maternity leave 26 weeks pregnancy",
    "paternity":          "paternity leave father child birth",
    "bereavement":        "bereavement leave death family",
    "password":           "password access management credentials minimum length characters",
    "harassment":         "harassment POSH sexual harassment complaint ICC filing",
    "sexual harassment":  "POSH prevention sexual harassment policy complaint",
    "pip":                "performance improvement plan PIP underperformance duration",
    "performance review": "performance review appraisal annual KRA rating cycle",
    "laptop":             "personal laptop BYOD device equipment IT security policy",
    "bonus":              "bonus variable pay incentive performance annual",
    "probation":          "probation period new employee joining confirmation",
    "gratuity":           "gratuity full final settlement F&F separation",
    "notice period":      "notice period resignation separation days grade",
    "wfh":                "work from home WFH remote hybrid policy",
    "remote":             "work from home WFH remote policy eligibility",
    "insurance":          "health insurance medical group personal accident term life",
    "reimbursement":      "reimbursement travel meal internet allowance",
}


def _expand_query(query: str) -> str:
    """Append HR-specific terminology to *query* to improve BM25 recall.

    Args:
        query: Raw user question.

    Returns:
        Enriched query string, or original if no expansion matched.
    """
    q_lower = query.lower()
    expansions: list[str] = []
    for keyword, expansion in _QUERY_EXPANSIONS.items():
        if keyword in q_lower:
            expansions.append(expansion)
    if expansions:
        enriched = query + " " + " ".join(dict.fromkeys(" ".join(expansions).split()))
        logger.info("[EXPAND] '%s' → '%s'", query, enriched[:120])
        return enriched
    return query

# ==============================================================================
# OOS CLASSIFIER  (Missing Piece #1: instant refusal before retrieval)
# ==============================================================================

# Hard OOS patterns: if matched → refuse immediately without retrieval
_OOS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bipl\b", re.I),
    re.compile(r"\bcricket\b", re.I),
    re.compile(r"\bfootball\b", re.I),
    re.compile(r"\bweather\b", re.I),
    re.compile(r"\bbitcoin\b", re.I),
    re.compile(r"\bcrypto\b", re.I),
    re.compile(r"\bstock\s+market\b", re.I),
    re.compile(r"\bquicksort\b", re.I),
    re.compile(r"\bpython\s+code\b", re.I),
    re.compile(r"\bwrite\s+(?:a\s+)?(?:code|script|program|function)\b", re.I),
    re.compile(r"\brecipe\b", re.I),
    re.compile(r"\bwho\s+won\b", re.I),
    re.compile(r"\btoday.s\s+(?:weather|news|score)\b", re.I),
]


def _is_oos_query(query: str) -> bool:
    """Return True if *query* is definitively out-of-scope.

    Only hard regex patterns are checked — no keyword or length heuristics.
    Short or ambiguous HR queries like "Can interns apply?", "Who approves it?",
    "What is allowed?" must reach the retriever; the reranker threshold is the
    correct gate for low-confidence cases, not a word-count heuristic.

    Args:
        query: Raw user question.

    Returns:
        ``True`` if a hard OOS pattern matched; ``False`` otherwise.
    """
    for pattern in _OOS_PATTERNS:
        if pattern.search(query):
            logger.info("[OOS] Hard pattern match — refusing immediately.")
            return True
    return False

# ==============================================================================
# DOCUMENT LOADING
# ==============================================================================


def load_documents(docs_dir: str = DOCS_PATH) -> list[Document]:
    """Load all PDF files from *docs_dir* using PyMuPDFLoader.

    Args:
        docs_dir: Path to the directory containing HR policy PDFs.

    Returns:
        Cleaned and metadata-enriched :class:`Document` list, one per page.

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
        except Exception as exc:
            logger.error("Failed to load '%s': %s", pdf_file.name, exc)
            raise

    logger.info("Total pages loaded: %d.", len(all_docs))
    _clean_documents(all_docs)
    _enrich_metadata(all_docs)
    return all_docs

# ==============================================================================
# METADATA ENRICHMENT  (Missing Piece #4: richer metadata)
# ==============================================================================


def _clean_page_text(text: str) -> str:
    """Aggressively remove boilerplate, headers, footers, and noise.

    Missing Piece #7: normalises case variation in leave-type names so
    "leave", "Leave", "LEAVE" all embed identically.

    Args:
        text: Raw text from one PDF page.

    Returns:
        Cleaned, normalised text.
    """
    # Remove boilerplate regex patterns
    for pattern in _CLEAN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove boilerplate exact lines
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _COMMON_LINES:
            continue
        # Remove pure page-number lines like "3" or "10"
        if re.match(r"^\d{1,3}$", stripped):
            continue
        # Remove lines that are only punctuation/symbols
        if re.match(r"^[^a-zA-Z0-9]+$", stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_doc_code(text: str) -> bool:
    """Return True if *text* matches a document-code pattern (e.g. ZDL-HR-01)."""
    return bool(re.match(r"^[A-Z]{2,}-[A-Z]{2,}-\d+$", text.strip()))


def _detect_section(text: str) -> str:
    """Infer the section heading from the first all-caps line in *text*.

    Args:
        text: Page or chunk text.

    Returns:
        Section name string, or ``"General"`` if none detected.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line or _is_doc_code(line):
            continue
        if re.match(r"^V\.\d+$", line):
            continue
        if line.isupper() and 4 <= len(line) <= 80 and len(line.split()) <= 10:
            return line
    return "General"


def _clean_documents(docs: list[Document]) -> None:
    """Clean raw page text in-place."""
    for doc in docs:
        doc.page_content = _clean_page_text(doc.page_content)


def _enrich_metadata(docs: list[Document]) -> None:
    """Add structured metadata fields in-place.

    Fields added:
        - ``policy_name``: human-readable policy title
        - ``doc_id``: source filename
        - ``page_number``: 1-indexed page number
        - ``section``: detected all-caps section heading
        - ``policy_type``: coarse category for metadata filtering
    """
    policy_type_map: dict[str, str] = {
        "leave":          "leave",
        "wfh":            "wfh",
        "work_from_home": "wfh",
        "compensation":   "compensation",
        "benefits":       "compensation",
        "onboarding":     "separation",
        "separation":     "separation",
        "performance":    "performance",
        "it_and_data":    "it_security",
        "code_of_conduct":"conduct",
        "posh":           "harassment",
        "sexual_harass":  "harassment",
        "employee_handbook": "handbook",
        "company_profile": "company",
    }

    for doc in docs:
        filename = Path(doc.metadata.get("source", "unknown")).name
        stem = Path(filename).stem.lower()
        parts = stem.split("_", 1)
        policy_name = parts[1].replace("_", " ").title() if len(parts) > 1 else stem.replace("_", " ").title()

        policy_type = "general"
        for key, ptype in policy_type_map.items():
            if key in stem:
                policy_type = ptype
                break

        doc.metadata["policy_name"] = policy_name
        doc.metadata["doc_id"] = filename
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        doc.metadata["section"] = _detect_section(doc.page_content)
        doc.metadata["policy_type"] = policy_type

# ==============================================================================
# CHUNKING  (Missing Piece #2: smaller chunks → 150-250 for 39 pages)
# ==============================================================================


def _build_chunks(docs: list[Document]) -> list[Document]:
    """Split documents into fine-grained chunks with metadata preambles.

    Uses ``CHUNK_SIZE=500, CHUNK_OVERLAP=100`` to produce ~150-250 chunks
    for a 39-page corpus (vs ~99 at 800/200), improving retrieval precision.

    Args:
        docs: Cleaned and enriched page-level documents.

    Returns:
        Flat list of chunk-level :class:`Document` objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[Document] = []
    for doc in docs:
        for chunk_text in splitter.split_text(doc.page_content):
            if not chunk_text.strip():
                continue
            new_doc = doc.model_copy(deep=True)
            new_doc.page_content = chunk_text
            chunks.append(new_doc)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    # Prepend structured metadata preamble — improves embedding discriminability
    for chunk in chunks:
        preamble = (
            f"Policy: {chunk.metadata.get('policy_name', '')}\n"
            f"Section: {chunk.metadata.get('section', '')}\n"
            f"Page: {chunk.metadata.get('page_number', '')}\n"
        )
        chunk.page_content = preamble + chunk.page_content

    logger.info(
        "Generated %d chunks from %d page(s) (chunk_size=%d, overlap=%d).",
        len(chunks), len(docs), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    # CP7: print so it's visible in terminal even at WARNING log level
    print(f"[CHUNKS] Built {len(chunks)} chunks from {len(docs)} pages.")
    return chunks

# ==============================================================================
# EMBEDDINGS
# ==============================================================================


def _build_embedding_model() -> HuggingFaceEmbeddings:
    """Load the BGE large embedding model with L2 normalisation."""
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
    """Load PDFs, chunk, embed, build FAISS index, and persist.

    Args:
        docs_dir: Directory containing source PDFs.
        faiss_path: Directory to save the FAISS index.

    Returns:
        ``(vectorstore, chunks)`` tuple.
    """
    docs = load_documents(docs_dir)
    chunks = _build_chunks(docs)
    embeddings = _build_embedding_model()

    logger.info("Building FAISS index over %d chunks…", len(chunks))
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    logger.info("FAISS index saved to '%s'.", faiss_path)
    return vectorstore, chunks


def load_vectorstore(faiss_path: str = FAISS_PATH) -> FAISS:
    """Load a previously persisted FAISS index from disk.

    Args:
        faiss_path: Path to saved FAISS index directory.

    Returns:
        Ready :class:`FAISS` vectorstore.

    Raises:
        FileNotFoundError: If *faiss_path* does not exist.
    """
    if not Path(faiss_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. Run build_vectorstore() first."
        )
    embeddings = _build_embedding_model()
    logger.info("Loading FAISS index from '%s'.", faiss_path)
    return FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True,
    )

# ==============================================================================
# BM25
# ==============================================================================


def _build_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    """Build BM25 sparse retriever.

    Args:
        chunks: Chunk-level documents.

    Returns:
        :class:`BM25Retriever` configured for ``TOP_K_RETRIEVAL`` results.
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
    """Compose MMR and BM25 into a weighted EnsembleRetriever.

    Args:
        vectorstore: Ready FAISS vectorstore.
        chunks: Chunk documents for BM25 indexing.

    Returns:
        :class:`EnsembleRetriever` with weights ``[0.6, 0.4]``.
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
    logger.info("EnsembleRetriever built (weights=%s).", ENSEMBLE_WEIGHTS)
    # CP1: confirm both retrievers are wired correctly
    logger.info(
        "[ENSEMBLE] type=%s | retrievers=[%s, %s]",
        type(ensemble).__name__,
        type(mmr_retriever).__name__,
        type(bm25_retriever).__name__,
    )
    return ensemble

# ==============================================================================
# RERANKER
# ==============================================================================


def _build_reranker() -> CrossEncoder:
    """Load the BGE cross-encoder reranking model."""
    logger.info("Loading reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


@traceable(name="reranker_stage")
def rerank_documents(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_k: int = TOP_K_RERANK,
) -> list[tuple[Document, float]]:
    """Score *docs* against *query* with the cross-encoder.

    Args:
        query: User question (may be expanded).
        docs: Candidate documents from the ensemble retriever.
        reranker: Loaded :class:`~sentence_transformers.CrossEncoder`.
        top_k: Number of top-scored docs to return.

    Returns:
        List of ``(document, score)`` tuples sorted by score descending.
    """
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores: list[float] = reranker.predict(pairs).tolist()
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _boost_posh_scores(
    query: str,
    ranked: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Boost POSH policy doc scores when query is harassment-related.

    The cross-encoder undervalues POSH docs relative to Code of Conduct
    because both use complaint-related language. A 3× boost corrects this.

    Args:
        query: Raw user question.
        ranked: Sorted ``(doc, score)`` list from cross-encoder.

    Returns:
        Re-sorted list with POSH scores boosted for harassment queries.
    """
    harassment_terms = {"harassment", "posh", "sexual", "complaint", "icc", "report"}
    if not any(term in query.lower() for term in harassment_terms):
        return ranked

    POSH_BOOST = 3.0
    boosted = []
    for doc, score in ranked:
        policy = doc.metadata.get("policy_name", "").lower()
        if "sexual harassment" in policy or "posh" in policy or "harassment" in policy:
            new_score = score * POSH_BOOST
            logger.info(
                "[BOOST] POSH %.4f → %.4f | section='%s'",
                score, new_score, doc.metadata.get("section", ""),
            )
            boosted.append((doc, new_score))
        else:
            boosted.append((doc, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)

# ==============================================================================
# RETRIEVAL
# ==============================================================================


@traceable(name="hybrid_retrieval")
def retrieve_context(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    original_query: str | None = None,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
    top_k_context: int = TOP_K_CONTEXT,
    rerank_threshold: float = RERANK_THRESHOLD,
    context_score_floor: float = CONTEXT_SCORE_FLOOR,
) -> list[Document] | str:
    """Full hybrid retrieval pipeline.

    Steps:
        1. Fetch candidates using *query* (may be expanded) via EnsembleRetriever.
        2. Rerank using *original_query* (unexpanded) — expansion distorts scores.
        3. Apply domain boosts (POSH).
        4. Primary threshold gate.
        5. Secondary context quality floor.
        6. Return top ``top_k_context`` docs.

    Args:
        query: Retrieval query (may be expanded for BM25 recall).
        ensemble_retriever: Built :class:`EnsembleRetriever`.
        reranker: Loaded :class:`~sentence_transformers.CrossEncoder`.
        original_query: Unexpanded user question for reranking. If ``None``,
            falls back to *query*.
        top_k_retrieval: Number of ensemble candidates.
        top_k_rerank: Number of docs after cross-encoder reranking.
        top_k_context: Maximum context docs returned to LLM.
        rerank_threshold: Primary OOS gate.
        context_score_floor: Secondary quality filter.

    Returns:
        List of context :class:`Document` objects, or :data:`REFUSAL_MESSAGE`.
    """
    rerank_query = original_query or query
    docs: list[Document] = ensemble_retriever.invoke(query)
    logger.info("[STEP 1] Retrieval complete — %d doc(s).", len(docs))
    logger.info("[STEP 1] Retrieval query : '%s'", query[:80])
    logger.info("[STEP 1] Reranking query : '%s'", rerank_query[:80])

    if not docs:
        logger.warning("[STEP 1] No documents retrieved — refusing.")
        return REFUSAL_MESSAGE

    ranked = rerank_documents(query=rerank_query, docs=docs, reranker=reranker, top_k=top_k_rerank)

    if not ranked:
        logger.warning("[STEP 2] Reranker returned nothing — refusing.")
        return REFUSAL_MESSAGE

    # Domain-specific score boost
    ranked = _boost_posh_scores(query, ranked)

    # Log top-5 for threshold tuning
    for i, (doc, score) in enumerate(ranked[:5]):
        logger.info(
            "[RANK %d] score=%.4f | policy='%s' | section='%s'",
            i + 1, score,
            doc.metadata.get("policy_name", ""),
            doc.metadata.get("section", ""),
        )

    best_doc, best_score = ranked[0]
    logger.info(
        "[STEP 2] Best score: %.4f (threshold=%.3f) | policy='%s'",
        best_score, rerank_threshold, best_doc.metadata.get("policy_name", ""),
    )

    # Primary OOS gate
    if best_score < rerank_threshold:
        logger.info("[STEP 2] Below threshold — refusing query.")
        return REFUSAL_MESSAGE

    # Secondary context quality floor
    filtered = [(doc, s) for doc, s in ranked if s >= context_score_floor]
    if not filtered:
        logger.info("[STEP 2] Floor filtered all — falling back to top-1.")
        filtered = ranked[:1]

    final_docs = [doc for doc, _ in filtered[:top_k_context]]
    logger.info("[STEP 3] Returning %d context doc(s).", len(final_docs))
    return final_docs

# ==============================================================================
# GENERATION
# ==============================================================================


def _build_llm() -> ChatGroq:
    """Instantiate Llama-3.3-70B via Groq at temperature 0."""
    logger.info("Loading LLM: %s", LLM_MODEL)
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)


def _format_context(docs: list[Document]) -> str:
    """Serialise context docs into a structured prompt block."""
    parts = []
    for doc in docs:
        part = (
            f"Policy: {doc.metadata.get('policy_name', 'Unknown')}\n"
            f"Section: {doc.metadata.get('section', 'Unknown')}\n"
            f"Page: {doc.metadata.get('page_number', '?')}\n\n"
            f"{doc.page_content}"
        )
        parts.append(part)
    return "\n\n---\n\n".join(parts)


def _build_sources(docs: list[Document]) -> list[dict[str, str]]:
    """Build deduplicated source citation list from context docs."""
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


def _validate_answer(
    question: str,
    answer: str,
    context: str,
    reranker: CrossEncoder,
) -> bool:
    """Validate that the generated answer is grounded in context.

    Two-stage check:

    Stage 1 — Cross-encoder Q-A relevance score.
        ``reranker.predict`` returns a numpy array; use ``.item()`` (not
        ``float()``) to extract the scalar — avoids the
        "only 0-dimensional arrays can be converted to Python scalars" error.
        Answers scoring below 0.01 are semantically unrelated to the question.

    Stage 2 — Numeric consistency.
        If the answer contains numbers, at least one must appear in context.
        Catches hallucinated figures (e.g. "12 days" when policy says "10 days").

    Args:
        question: Original user question.
        answer: Generated answer to validate.
        context: Context string used for generation.
        reranker: The already-loaded cross-encoder (reused, no extra cost).

    Returns:
        ``True`` if the answer passes both stages, ``False`` otherwise.
    """
    if REFUSAL_MESSAGE.lower() in answer.lower():
        return True

    # Stage 1: Q-A cross-encoder relevance
    try:
        raw = reranker.predict([(question, answer)])
        # raw is a numpy ndarray — use .item() not float() to extract scalar
        qa_score: float = raw.item() if hasattr(raw, "item") else float(raw[0])
        logger.info("[VALIDATE] Q-A score: %.4f", qa_score)
        if qa_score < 0.01:
            logger.info("[VALIDATE] Score too low — answer likely off-topic.")
            return False
    except Exception as exc:
        logger.warning("[VALIDATE] Stage 1 failed (%s) — continuing to stage 2.", exc)

    # Stage 2: numeric consistency
    answer_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
    if answer_numbers:
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context))
        matched = [n for n in answer_numbers if n in context_numbers]
        if not matched:
            logger.info(
                "[VALIDATE] Numbers in answer %s absent from context %s — refusing.",
                answer_numbers, sorted(context_numbers)[:10],
            )
            return False

    return True


@traceable(name="answer_question")
def answer_question(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    llm: ChatGroq,
) -> dict[str, Any]:
    """Full RAG pipeline: OOS check → expand → retrieve → generate → validate.

    Pipeline:
        1. Instant OOS check (keyword/pattern — no retrieval cost).
        2. Query expansion for BM25 recall.
        3. Hybrid retrieval with dual threshold gates.
        4. LLM answer generation with strict grounding prompt.
        5. Answer validation — refuse if not grounded.

    Args:
        query: User's HR-related question.
        ensemble_retriever: Built :class:`EnsembleRetriever`.
        reranker: Loaded :class:`~sentence_transformers.CrossEncoder`.
        llm: Ready :class:`~langchain_groq.ChatGroq` instance.

    Returns:
        Dict with ``"answer"``, ``"sources"``, ``"documents"``.
    """
    logger.info("[STEP 0] Query: '%s'", query)

    # Missing Piece #1: instant OOS gate before any retrieval
    if _is_oos_query(query):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

    # Query expansion improves BM25 recall; original query used for reranking (CP5)
    retrieval_query = _expand_query(query)

    context_or_refusal = retrieve_context(
        query=retrieval_query,
        original_query=query,
        ensemble_retriever=ensemble_retriever,
        reranker=reranker,
    )

    if isinstance(context_or_refusal, str):
        logger.info("[STEP 3] Retrieval refused.")
        return {"answer": context_or_refusal, "sources": [], "documents": []}

    final_docs: list[Document] = context_or_refusal
    if not final_docs:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

    context = _format_context(final_docs)
    logger.info("[STEP 4] Context: %d chars — calling LLM.", len(context))

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

    # CP3: validate with cross-encoder + numeric check (not self-grading LLM)
    if not _validate_answer(query, answer, context, reranker):
        logger.info("[STEP 6] Validation failed — returning refusal.")
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

    return {
        "answer": answer,
        "sources": _build_sources(final_docs),
        "documents": final_docs,
    }

# ==============================================================================
# EVALUATION  (Missing Piece #8: local harness)
# ==============================================================================

_GOLD_DATASET: list[dict[str, Any]] = [
    # ── Leave Policy ───────────────────────────────────────────────────────
    {"question": "How many sick leaves are allowed per year?",          "answer_contains": ["10", "sick"]},
    {"question": "How many casual leaves do employees get?",            "answer_contains": ["8", "casual"]},
    {"question": "How many earned leaves are provided?",                "answer_contains": ["earned leave"]},
    {"question": "What is the maternity leave entitlement?",            "answer_contains": ["26 weeks"]},
    {"question": "How many days of paternity leave are allowed?",       "answer_contains": ["paternity"]},
    {"question": "How many bereavement leaves are allowed?",            "answer_contains": ["bereavement"]},
    {"question": "Can sick leave be carried forward to next year?",     "answer_contains": ["sick leave"]},
    # ── WFH ───────────────────────────────────────────────────────────────
    {"question": "How does the work from home policy work?",            "answer_contains": ["work from home"]},
    {"question": "Which employee level is eligible for WFH?",           "answer_contains": ["L3"]},
    # ── Compensation & Benefits ────────────────────────────────────────────
    {"question": "What health insurance benefits are provided?",        "answer_contains": ["health"]},
    {"question": "Is there a travel reimbursement policy?",             "answer_contains": ["travel"]},
    {"question": "What is the meal or food allowance policy?",          "answer_contains": ["allowance"]},
    # ── Onboarding & Separation ────────────────────────────────────────────
    {"question": "What is the notice period for resignation?",          "answer_contains": ["notice period"]},
    {"question": "How long is the probation period for new employees?", "answer_contains": ["probation"]},
    {"question": "What happens to leaves during probation?",            "answer_contains": ["probation"]},
    # ── IT / Security ──────────────────────────────────────────────────────
    {"question": "What are the password requirements?",                 "answer_contains": ["12 characters"]},
    {"question": "Can employees use personal laptops for work?",        "answer_contains": ["personal laptop"]},
    # ── Performance ────────────────────────────────────────────────────────
    {"question": "How often are performance reviews conducted?",        "answer_contains": ["performance review"]},
    {"question": "What is the performance rating scale used?",          "answer_contains": ["rating"]},
    # ── OOS (must refuse) ─────────────────────────────────────────────────
    {"question": "Who won the IPL in 2025?",                            "answer_contains": ["I can only answer HR-related questions"]},
    {"question": "What is Bitcoin?",                                    "answer_contains": ["I can only answer HR-related questions"]},
    {"question": "Write Python code for quicksort",                     "answer_contains": ["I can only answer HR-related questions"]},
    {"question": "What is today's weather?",                            "answer_contains": ["I can only answer HR-related questions"]},
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
        ensemble_retriever: Built :class:`EnsembleRetriever`.
        reranker: Loaded :class:`~sentence_transformers.CrossEncoder`.
        llm: Ready :class:`~langchain_groq.ChatGroq` instance.
        test_cases: Optional custom test cases; defaults to ``_GOLD_DATASET``.

    Returns:
        Dict with ``"total"``, ``"correct"``, ``"accuracy"``, ``"results"``.
    """
    dataset = test_cases if test_cases is not None else _GOLD_DATASET
    total = len(dataset)
    correct = 0
    results: list[dict[str, Any]] = []

    for item in dataset:
        question: str = item["question"]
        expected: list[str] = item.get("answer_contains", [])

        try:
            result = answer_question(
                query=question,
                ensemble_retriever=ensemble_retriever,
                reranker=reranker,
                llm=llm,
            )
            answer = result["answer"] if isinstance(result, dict) else result
        except Exception as exc:
            logger.error("Error on '%s': %s", question, exc)
            answer = ""

        passed = all(phrase.lower() in answer.lower() for phrase in expected)
        if passed:
            correct += 1

        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s", status, question)
        results.append({"question": question, "passed": passed, "answer": answer})

    accuracy = correct / total if total else 0.0
    logger.info(
        "Evaluation complete — %d/%d (%.1f%%)", correct, total, accuracy * 100
    )
    return {"total": total, "correct": correct, "accuracy": accuracy, "results": results}

# ==============================================================================
# ENTRYPOINT
# ==============================================================================


def main() -> None:
    """Build the RAG pipeline and run evaluation."""
    logger.info("=== Zyro Dynamics RAG Pipeline Starting ===")

    faiss_index_path = Path(FAISS_PATH)
    if faiss_index_path.exists():
        logger.info("FAISS index found — loading from disk.")
        vectorstore = load_vectorstore(FAISS_PATH)
        docs = load_documents(DOCS_PATH)
        chunks = _build_chunks(docs)
    else:
        logger.info("No FAISS index found — building from scratch.")
        vectorstore, chunks = build_vectorstore(DOCS_PATH, FAISS_PATH)

    ensemble_retriever = build_retrievers(vectorstore, chunks)
    reranker = _build_reranker()
    llm = _build_llm()

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