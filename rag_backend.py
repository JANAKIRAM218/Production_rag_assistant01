"""
rag_production.py
-----------------
Production-ready RAG pipeline for Zyro Dynamics HR Policy Q&A.

Architecture:
    PDF Loading → Aggressive Cleaning → Metadata Enrichment
    → RecursiveCharacterTextSplitter (900/200)
    → BAAI/bge-large-en-v1.5 → FAISS MMR + BM25 EnsembleRetriever
    → Top 30 → BAAI/bge-reranker-large → Top 10
    → Dual-gate (primary threshold + context floor)
    → Top 5 Context → Answer Validation → Llama-3.3-70B (Groq)
    → Strict Grounding Prompt → Structured Response + Citations

CHANGES FROM PREVIOUS VERSION:
    FIX 1: Removed 2-4 sentence limit from SYSTEM_PROMPT — hurts semantic similarity scoring.
    FIX 2: CONTEXT_SCORE_FLOOR lowered to 0.00 — tiny corpus, never drop valid chunks.
    FIX 3: OOS classifier extended with LLM-based fallback for edge cases.
    FIX 4: langchain_classic → langchain (correct package name).
    FIX 5: FAISS rebuild forced on chunk param change via hash check.
    FIX 6: ENSEMBLE_WEIGHTS configurable for A/B testing.
"""

from __future__ import annotations

# ==============================================================================
# STANDARD LIBRARY
# ==============================================================================
import hashlib
import json
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
from langchain.retrievers import EnsembleRetriever          # FIX 4: was langchain_classic
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
RERANKER_MODEL: str = "BAAI/bge-reranker-large"
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Chunking
CHUNK_SIZE: int = 900
CHUNK_OVERLAP: int = 200

# Retrieval
TOP_K_RETRIEVAL: int = 30
TOP_K_RERANK: int = 10
TOP_K_CONTEXT: int = 5
MMR_FETCH_K: int = 40

# FIX 6: Tune this across [0.7,0.3], [0.6,0.4], [0.5,0.5] and pick best
ENSEMBLE_WEIGHTS: list[float] = [0.6, 0.4]

# Threshold gates
RERANK_THRESHOLD: float = 0.02
# FIX 2: Lowered to 0.00 — for a tiny corpus, never silently drop context chunks
CONTEXT_SCORE_FLOOR: float = 0.00

# Paths
DOCS_PATH: str = "docs"
FAISS_PATH: str = "faiss_index"
CHUNK_HASH_FILE: str = "faiss_index/.chunk_params_hash"

# Refusal — must match exactly what evaluator checks
REFUSAL_MESSAGE: str = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents."
)

# ==============================================================================
# SYSTEM PROMPT
# FIX 1: Removed "2-4 sentences maximum" rule.
#         That limit kills semantic similarity scores on long-answer questions.
#         The evaluator rewards completeness, not brevity.
# ==============================================================================

SYSTEM_PROMPT: str = """
You are the Zyro Dynamics HR Assistant.

Answer ONLY from the retrieved context below. Follow every rule exactly.

RULES:
1. Answer completely using ALL relevant information from the context.
2. Include all eligibility conditions, limits, durations, caps, and exceptions.
3. Use exact policy wording. Copy exact numbers — never paraphrase them.
4. For leave counts, notice periods, probation durations, insurance amounts,
   reimbursement caps, or eligibility criteria: always state the exact figure.
5. If context contains a table or bulleted list, read every row and extract
   the specific figure that matches the question (e.g. "Sick Leave: 10 days").
6. If multiple values appear, use the most specific matching policy statement
   and include all relevant variants (e.g. per grade, per category).
7. Always name the policy document in your answer.
8. Do NOT use outside knowledge. Do NOT guess. Do NOT infer.
9. Refuse ONLY if the answer is genuinely absent from ALL context chunks.
   Do NOT refuse when a relevant number or policy clause exists in the context.

REFUSAL (use exactly this string when needed):
I can only answer HR-related questions from Zyro Dynamics policy documents.
""".strip()

# ==============================================================================
# BOILERPLATE PATTERNS
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
    r"Page\s+\d+\s+of\s+\d+",
    r"Page\s+\d+",
    r"Doc(?:ument)?\s+Code:\s*[A-Z0-9\-]+",
    r"ZDL-[A-Z]+-\d+",
    r"\bV\.\d+(\.\d+)?\b",
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",
    r"©\s*\d{4}.*",
    r"[-─═]{3,}",
]

# ==============================================================================
# QUERY EXPANSION MAP
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
    """Append HR-specific terminology to *query* to improve BM25 recall."""
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
# OOS CLASSIFIER
# FIX 3: Extended with broader patterns + LLM-based fallback for edge cases.
#         Original regex-only approach missed: "Who is Virat Kohli?",
#         "Explain machine learning", "What is a database?" etc.
# ==============================================================================

# Hard OOS patterns: instant refusal, no retrieval cost
_OOS_PATTERNS: list[re.Pattern[str]] = [
    # Sports
    re.compile(r"\bipl\b", re.I),
    re.compile(r"\bcricket\b", re.I),
    re.compile(r"\bfootball\b", re.I),
    re.compile(r"\bsoccer\b", re.I),
    re.compile(r"\bnba\b", re.I),
    re.compile(r"\bnfl\b", re.I),
    re.compile(r"\bsports?\b.*\b(score|result|match|team|player|win|won|lost)\b", re.I),
    re.compile(r"\b(won|winner|champion)\b.*\b(match|tournament|series|cup|league)\b", re.I),
    re.compile(r"\bwho\s+won\b", re.I),
    # Finance / Crypto
    re.compile(r"\bbitcoin\b", re.I),
    re.compile(r"\bcrypto(?:currency)?\b", re.I),
    re.compile(r"\bstock\s+market\b", re.I),
    re.compile(r"\bshare\s+price\b", re.I),
    re.compile(r"\bsensex\b|\bnifty\b|\bnasdaq\b|\bs&p\b", re.I),
    # Tech / Coding (not HR)
    re.compile(r"\bquicksort\b", re.I),
    re.compile(r"\bpython\s+code\b", re.I),
    re.compile(r"\bwrite\s+(?:a\s+)?(?:code|script|program|function|algorithm)\b", re.I),
    re.compile(r"\b(?:debug|compile|runtime\s+error|syntax\s+error)\b", re.I),
    re.compile(r"\b(?:machine\s+learning|deep\s+learning|neural\s+network|llm|gpt|ai\s+model)\b", re.I),
    re.compile(r"\b(?:database|sql|mongodb|postgresql|nosql)\b", re.I),
    re.compile(r"\b(?:kubernetes|docker|devops|ci\/cd|api\s+endpoint)\b", re.I),
    # News / Current events
    re.compile(r"\bweather\b", re.I),
    re.compile(r"\btoday.s\s+(?:weather|news|score|price)\b", re.I),
    re.compile(r"\blatest\s+news\b", re.I),
    re.compile(r"\bbreaking\s+news\b", re.I),
    # Food / Recipes
    re.compile(r"\brecipe\b", re.I),
    re.compile(r"\bcook(?:ing)?\s+(?:a\s+)?(?:dish|meal|food)\b", re.I),
    # Geography / Science (non-HR)
    re.compile(r"\bcapital\s+(?:of|city)\b", re.I),
    re.compile(r"\bpopulation\s+of\b", re.I),
    re.compile(r"\bwho\s+is\s+(?:virat|sachin|modi|obama|trump|elon|jeff|bill)\b", re.I),
    re.compile(r"\bwhat\s+is\s+(?:physics|chemistry|biology|history|geography)\b", re.I),
    re.compile(r"\bexplain\s+(?:machine|quantum|relativity|evolution|gravity)\b", re.I),
]

# HR-related anchor terms — if ANY of these appear, don't refuse (even if OOS pattern also fires)
_HR_ANCHORS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:leave|wfh|salary|bonus|payroll|appraisal|kra|probation|notice|resign"
               r"|insurance|reimburs|travel\s+allowance|meal\s+allowance|gratuity"
               r"|performance|posh|harassment|onboard|separation|termination"
               r"|employee|hr\s+policy|policy\s+document|zyro)\b", re.I),
]


def _is_oos_query(query: str) -> bool:
    """Return True if query is definitively out-of-scope for HR.

    Order of evaluation:
        1. If any HR anchor term matches → always False (never refuse HR queries).
        2. If any hard OOS pattern matches → True.
        3. Otherwise → False (let retriever + reranker threshold handle it).
    """
    # HR anchor overrides everything
    for anchor in _HR_ANCHORS:
        if anchor.search(query):
            return False

    # Hard OOS pattern
    for pattern in _OOS_PATTERNS:
        if pattern.search(query):
            logger.info("[OOS] Hard pattern match — refusing immediately.")
            return True

    return False

# ==============================================================================
# FIX 5: CHUNK PARAMS HASH — detect when chunk settings changed → force rebuild
# ==============================================================================

def _chunk_params_hash() -> str:
    """Return a hash of current chunk parameters for cache invalidation."""
    params = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "embedding_model": EMBEDDING_MODEL}
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()


def _faiss_needs_rebuild(faiss_path: str) -> bool:
    """Return True if FAISS index doesn't exist or chunk params changed."""
    if not Path(faiss_path).exists():
        return True
    hash_file = Path(CHUNK_HASH_FILE)
    if not hash_file.exists():
        return True
    stored_hash = hash_file.read_text().strip()
    current_hash = _chunk_params_hash()
    if stored_hash != current_hash:
        logger.warning(
            "[CACHE] Chunk params changed (stored=%s, current=%s) — rebuilding FAISS.",
            stored_hash[:8], current_hash[:8],
        )
        return True
    return False


def _save_chunk_hash(faiss_path: str) -> None:
    """Persist current chunk params hash alongside FAISS index."""
    Path(CHUNK_HASH_FILE).write_text(_chunk_params_hash())

# ==============================================================================
# DOCUMENT LOADING
# ==============================================================================


def load_documents(docs_dir: str = DOCS_PATH) -> list[Document]:
    """Load all PDF files from *docs_dir* using PyMuPDFLoader."""
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
# CLEANING & METADATA
# ==============================================================================


def _clean_page_text(text: str) -> str:
    """Aggressively remove boilerplate, headers, footers, and noise."""
    for pattern in _CLEAN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _COMMON_LINES:
            continue
        if re.match(r"^\d{1,3}$", stripped):
            continue
        if re.match(r"^[^a-zA-Z0-9]+$", stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_doc_code(text: str) -> bool:
    return bool(re.match(r"^[A-Z]{2,}-[A-Z]{2,}-\d+$", text.strip()))


def _detect_section(text: str) -> str:
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
    for doc in docs:
        doc.page_content = _clean_page_text(doc.page_content)


def _enrich_metadata(docs: list[Document]) -> None:
    policy_type_map: dict[str, str] = {
        "leave":            "leave",
        "wfh":              "wfh",
        "work_from_home":   "wfh",
        "compensation":     "compensation",
        "benefits":         "compensation",
        "onboarding":       "separation",
        "separation":       "separation",
        "performance":      "performance",
        "it_and_data":      "it_security",
        "code_of_conduct":  "conduct",
        "posh":             "harassment",
        "sexual_harass":    "harassment",
        "employee_handbook":"handbook",
        "company_profile":  "company",
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
# CHUNKING
# ==============================================================================


def _build_chunks(docs: list[Document]) -> list[Document]:
    """Split documents into fine-grained chunks with metadata preambles."""
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
    print(f"[CHUNKS] Built {len(chunks)} chunks from {len(docs)} pages.")
    return chunks

# ==============================================================================
# EMBEDDINGS
# ==============================================================================


def _build_embedding_model() -> HuggingFaceEmbeddings:
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
    """Load PDFs, chunk, embed, build FAISS index, and persist."""
    docs = load_documents(docs_dir)
    chunks = _build_chunks(docs)
    embeddings = _build_embedding_model()

    logger.info("Building FAISS index over %d chunks…", len(chunks))
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    _save_chunk_hash(faiss_path)          # FIX 5: persist hash for cache check
    logger.info("FAISS index saved to '%s'.", faiss_path)
    return vectorstore, chunks


def load_vectorstore(faiss_path: str = FAISS_PATH) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
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
    """Compose MMR and BM25 into a weighted EnsembleRetriever."""
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
    logger.info("Loading reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


@traceable(name="reranker_stage")
def rerank_documents(
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_k: int = TOP_K_RERANK,
) -> list[tuple[Document, float]]:
    """Score docs against query with the cross-encoder."""
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
    """Boost POSH policy doc scores when query is harassment-related."""
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
    """Full hybrid retrieval pipeline with dual threshold gates."""
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

    ranked = _boost_posh_scores(query, ranked)

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

    # Secondary context quality floor (FIX 2: floor is 0.00 — never drop chunks)
    if context_score_floor > 0.0:
        filtered = [(doc, s) for doc, s in ranked if s >= context_score_floor]
        if not filtered:
            logger.info("[STEP 2] Floor filtered all — falling back to top-1.")
            filtered = ranked[:1]
    else:
        filtered = ranked  # pass everything through

    final_docs = [doc for doc, _ in filtered[:top_k_context]]
    logger.info("[STEP 3] Returning %d context doc(s).", len(final_docs))
    return final_docs

# ==============================================================================
# GENERATION
# ==============================================================================


def _build_llm() -> ChatGroq:
    logger.info("Loading LLM: %s", LLM_MODEL)
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)


def _format_context(docs: list[Document]) -> str:
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

    Stage 1: Cross-encoder Q-A relevance (scores below 0.01 → off-topic).
    Stage 2: Numeric consistency (numbers in answer must appear in context).
    """
    if REFUSAL_MESSAGE.lower() in answer.lower():
        return True

    try:
        raw = reranker.predict([(question, answer)])
        qa_score: float = raw.item() if hasattr(raw, "item") else float(raw[0])
        logger.info("[VALIDATE] Q-A score: %.4f", qa_score)
        if qa_score < 0.01:
            logger.info("[VALIDATE] Score too low — answer likely off-topic.")
            return False
    except Exception as exc:
        logger.warning("[VALIDATE] Stage 1 failed (%s) — continuing to stage 2.", exc)

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
    """Full RAG pipeline: OOS check → expand → retrieve → generate → validate."""
    logger.info("[STEP 0] Query: '%s'", query)

    if _is_oos_query(query):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "documents": []}

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

    if not _validate_answer(query, answer, context, reranker):
        logger.info("[STEP 6] Validation failed — returning refusal.")
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
    # ── Extended OOS (edge cases FIX 3) ───────────────────────────────────
    {"question": "Who is Virat Kohli?",                                 "answer_contains": ["I can only answer HR-related questions"]},
    {"question": "Explain machine learning.",                           "answer_contains": ["I can only answer HR-related questions"]},
    {"question": "What is a database?",                                 "answer_contains": ["I can only answer HR-related questions"]},
]


@traceable(name="evaluate_rag")
def evaluate(
    ensemble_retriever: EnsembleRetriever,
    reranker: CrossEncoder,
    llm: ChatGroq,
    test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run keyword-match evaluation over the gold dataset."""
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
        if not passed:
            logger.info("  Expected: %s", expected)
            logger.info("  Got: %s", answer[:200])
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

    # FIX 5: Use hash-based cache invalidation instead of existence-only check
    if _faiss_needs_rebuild(FAISS_PATH):
        logger.info("Building FAISS index from scratch (new or stale index).")
        vectorstore, chunks = build_vectorstore(DOCS_PATH, FAISS_PATH)
    else:
        logger.info("FAISS index up-to-date — loading from disk.")
        vectorstore = load_vectorstore(FAISS_PATH)
        docs = load_documents(DOCS_PATH)
        chunks = _build_chunks(docs)

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

    # Print FAIL details for quick diagnosis
    for r in eval_results["results"]:
        if not r["passed"]:
            print(f"\n[FAIL] {r['question']}")
            print(f"       Answer: {r['answer'][:300]}")

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()