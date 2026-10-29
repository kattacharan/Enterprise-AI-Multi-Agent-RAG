import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep, ChunkVerification


# ─────────────────────────────────────────────────────────────────────────────
# Context-collision detection tables
# ─────────────────────────────────────────────────────────────────────────────

# Terms that signal SHORT-TERM incentive context
STI_TERMS = {
    "short-term", "short term", "sti", "annual incentive",
    "annual bonus", "annual performance", "annual award"
}

# Terms that signal LONG-TERM incentive context
LTI_TERMS = {
    "long-term", "long term", "lti", "performance share",
    "restricted share", "share award", "vesting", "vest"
}

# Remuneration component labels (used for cross-component collision detection)
COMPONENT_PAIRS: List[Tuple[set, set]] = [
    (STI_TERMS, LTI_TERMS),
]

# Period markers  2025-2029, 2025 to 2029, etc.
PERIOD_RE = re.compile(r'\b(20\d{2})\s*(?:to|–|-|—)\s*(20\d{2})\b', re.IGNORECASE)

# Role markers
PRESIDENT_CEO_TERMS = {"president and ceo", "president & ceo", "ceo", "chief executive"}
BOARD_TERMS = {"board", "board member", "non-executive", "non executive", "ned"}


# ─────────────────────────────────────────────────────────────────────────────
# Key-concept extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_query_concepts(query: str) -> Dict[str, Any]:
    """
    Extracts entities and attributes from the user question to guide
    contextual verification.
    Returns:
        requires_sti: question is about short-term incentive
        requires_lti: question is about long-term incentive
        requires_ceo: question is about President and CEO
        requires_board: question is about Board / non-execs
        period_years: set of year strings mentioned in the question
        key_numbers: set of percentage/numeric strings in the question
        key_terms: important domain words extracted
    """
    q = query.lower()

    requires_sti = any(t in q for t in STI_TERMS)
    requires_lti = any(t in q for t in LTI_TERMS)
    requires_ceo = any(t in q for t in PRESIDENT_CEO_TERMS)
    requires_board = any(t in q for t in BOARD_TERMS)

    # Extract year ranges from query
    period_years: set = set()
    for m in PERIOD_RE.finditer(q):
        period_years.add(m.group(1))
        period_years.add(m.group(2))

    # Extract percentage numbers from query
    key_numbers: set = set()
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', q):
        key_numbers.add(m.group(1))

    # Important domain words (filter stop-words)
    stop = {"what", "is", "the", "for", "and", "of", "in", "a", "an",
            "are", "how", "does", "do", "its", "from", "to", "at", "by",
            "be", "with", "their", "this", "that", "which"}
    key_terms = {w for w in re.findall(r'\b[a-z]{3,}\b', q) if w not in stop}

    return {
        "requires_sti": requires_sti,
        "requires_lti": requires_lti,
        "requires_ceo": requires_ceo,
        "requires_board": requires_board,
        "period_years": period_years,
        "key_numbers": key_numbers,
        "key_terms": key_terms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic chunk evaluator (no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_verify_chunk(
    chunk_text: str,
    query: str,
    concepts: Dict[str, Any],
    faiss_score: float
) -> Dict[str, Any]:
    """
    Evaluates a single chunk against the query using deterministic rules.
    Returns a verification dict compatible with the LLM response format.
    """
    c = chunk_text.lower()
    q_concepts = concepts

    reasons: List[str] = []
    reject_reasons: List[str] = []

    # ── 1. STI vs LTI context collision detection ──────────────────────────
    chunk_has_sti = any(t in c for t in STI_TERMS)
    chunk_has_lti = any(t in c for t in LTI_TERMS)

    context_ok = True

    if q_concepts["requires_sti"] and chunk_has_lti and not chunk_has_sti:
        # Question is about STI but chunk is about LTI
        context_ok = False
        reject_reasons.append("Chunk refers to LTI (long-term incentive) but question asks about STI.")
    elif q_concepts["requires_lti"] and chunk_has_sti and not chunk_has_lti:
        context_ok = False
        reject_reasons.append("Chunk refers to STI (short-term incentive) but question asks about LTI.")

    # ── 2. Role collision detection ─────────────────────────────────────────
    chunk_has_ceo = any(t in c for t in PRESIDENT_CEO_TERMS)
    chunk_has_board = any(t in c for t in BOARD_TERMS)

    if q_concepts["requires_ceo"] and chunk_has_board and not chunk_has_ceo:
        context_ok = False
        reject_reasons.append("Chunk is about Board/NEDs but question asks about President and CEO.")
    elif q_concepts["requires_board"] and chunk_has_ceo and not chunk_has_board:
        context_ok = False
        reject_reasons.append("Chunk is about President and CEO but question asks about Board.")

    # ── 3. Period mismatch detection ────────────────────────────────────────
    chunk_years: set = set()
    for m in PERIOD_RE.finditer(c):
        chunk_years.add(m.group(1))
        chunk_years.add(m.group(2))

    period_match = True
    if q_concepts["period_years"] and chunk_years:
        if not q_concepts["period_years"].intersection(chunk_years):
            period_match = False
            reject_reasons.append(
                f"Period mismatch: question about {q_concepts['period_years']} "
                f"but chunk mentions {chunk_years}."
            )

    # ── 4. Key-term overlap (how many query concepts appear in the chunk) ────
    key_terms = q_concepts["key_terms"]
    if key_terms:
        matched_terms = {t for t in key_terms if t in c}
        term_overlap_ratio = len(matched_terms) / len(key_terms)
    else:
        term_overlap_ratio = 0.5

    if term_overlap_ratio < 0.15:
        reasons.append(f"Low term overlap ({term_overlap_ratio:.0%}).")

    # ── 5. Direct-answer signal: chunk contains key percentages or facts ─────
    direct_answer = False
    if q_concepts["key_numbers"]:
        chunk_numbers = {m.group(1) for m in re.finditer(r'(\d+(?:\.\d+)?)\s*%', c)}
        if q_concepts["key_numbers"].intersection(chunk_numbers):
            direct_answer = True
            reasons.append("Chunk contains specific percentage matching the question.")
    else:
        # No specific number in question; if term overlap is high enough, consider direct
        if term_overlap_ratio >= 0.4:
            direct_answer = True

    # ── 6. Compute verifier_score ────────────────────────────────────────────
    # Base score from FAISS similarity (scaled)
    base = faiss_score * 0.4

    # Term overlap contribution (up to 0.3)
    overlap_contrib = term_overlap_ratio * 0.3

    # Direct answer bonus
    direct_bonus = 0.2 if direct_answer else 0.0

    # Context penalty
    context_penalty = 0.35 if not context_ok else 0.0

    # Period mismatch penalty
    period_penalty = 0.15 if not period_match else 0.0

    verifier_score = max(0.0, min(1.0, base + overlap_contrib + direct_bonus - context_penalty - period_penalty))

    # ── 7. is_relevant decision ──────────────────────────────────────────────
    is_relevant = context_ok and period_match and (verifier_score >= settings.VERIFIER_ACCEPT_THRESHOLD)

    if not context_ok:
        reason_text = " | ".join(reject_reasons)
    elif not period_match:
        reason_text = " | ".join(reject_reasons)
    elif is_relevant:
        reason_text = f"Relevant — term overlap {term_overlap_ratio:.0%}, direct={direct_answer}."
    else:
        reason_text = f"Low verifier score ({verifier_score:.2f}); insufficient evidence."

    return {
        "is_relevant": is_relevant,
        "relevance_score": round(verifier_score, 3),
        "direct_answer": direct_answer,
        "context_match": context_ok and period_match,
        "reason": reason_text,
        "evidence": chunk_text[:200],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM-powered chunk evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _llm_verify_chunk(chunk_text: str, query: str, llm) -> Optional[Dict[str, Any]]:
    """Uses OpenAI to evaluate a chunk with structured output."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        import json

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert Evidence Relevance Grader for an enterprise RAG system.\n"
                "Your task: evaluate whether a retrieved document chunk directly supports "
                "answering the user's EXACT question.\n\n"
                "Evaluate ALL of the following:\n"
                "- Does the chunk directly answer the question?\n"
                "- Does it refer to the SAME remuneration component? "
                "  (e.g. STI vs LTI, base salary vs bonus)\n"
                "- Does it refer to the SAME person/role? "
                "  (e.g. President and CEO vs Board member)\n"
                "- Does it refer to the SAME time period if specified?\n"
                "- Does it contain the specific fact/number asked about?\n"
                "- Or is it merely topically related but about a DIFFERENT context?\n\n"
                "IMPORTANT: A chunk is NOT relevant if it contains a similar number "
                "but in a DIFFERENT context (e.g. chunk about LTI contains 60% but "
                "question asks about STI's 60% requirement — these are DIFFERENT).\n\n"
                "Respond ONLY with JSON:\n"
                '{"is_relevant": true/false, "relevance_score": 0.0-1.0, '
                '"direct_answer": true/false, "context_match": true/false, '
                '"reason": "...", "evidence": "..."}'
            )),
            ("user", f"Question: {query}\n\nChunk:\n{chunk_text}")
        ])
        chain = prompt | llm
        res = chain.invoke({})
        content = res.content.strip()
        if "{" in content and "}" in content:
            return json.loads(content[content.find("{"):content.rfind("}") + 1])
    except Exception as e:
        print(f"[Verifier] LLM grader error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Composite evidence score
# ─────────────────────────────────────────────────────────────────────────────

def _composite_score(
    faiss_sim: float,
    verifier_score: float,
    direct_answer: bool,
    context_match: bool,
) -> float:
    """
    Combines multiple signals into a single ranking score.
    Weights are configurable via config.py.
    """
    context_bonus = verifier_score * settings.RANK_WEIGHT_CONTEXT if context_match else 0.0
    direct_bonus = settings.RANK_WEIGHT_DIRECT if direct_answer else 0.0
    semantic = faiss_sim * settings.RANK_WEIGHT_SEMANTIC
    verifier_contrib = verifier_score * settings.RANK_WEIGHT_VERIFIER
    return round(min(1.0, semantic + context_bonus + direct_bonus + verifier_contrib), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Verifier agent node
# ─────────────────────────────────────────────────────────────────────────────

def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Verifier / Hallucination Grader Agent (improved):

    1. Extracts key concepts (STI/LTI, role, period, numbers) from the question.
    2. Evaluates EVERY candidate chunk for:
       - Context collision (STI vs LTI, CEO vs Board, period mismatch)
       - Term overlap with the question
       - Direct answer signal (specific fact/number)
    3. Produces a per-chunk ChunkVerification audit record.
    4. Ranks accepted chunks by composite score.
    5. Passes only TOP_K_RETRIEVAL best evidence chunks to the Synthesizer.
    """
    query = state.get("query", "")
    # Use the full candidate pool from Retriever
    candidates = state.get("retrieved_candidates") or state.get("documents", [])
    steps = list(state.get("steps_executed", []))

    if not candidates:
        chunk_verifications = []
        verification_status = "NO_RELEVANT_DOCS"
        verification_score = 0.0
        verification_reason = "No candidate documents matched the retrieval threshold."
        graded_docs = []
    else:
        # Extract concepts from the question for deterministic verification
        concepts = _extract_query_concepts(query)

        # Try to get OpenAI LLM for enhanced grading
        llm = None
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    temperature=0.0,
                    api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                print(f"[Verifier] LLM init failed: {e}. Using deterministic verifier.")

        chunk_verifications: List[ChunkVerification] = []
        graded_docs: List[Dict[str, Any]] = []

        for rank_idx, doc in enumerate(candidates, start=1):
            chunk_text = doc.get("content", "")
            metadata = doc.get("metadata", {})
            faiss_sim = doc.get("score", 0.0)
            retrieval_rank = doc.get("retrieval_rank", rank_idx)

            # Run LLM grader if available, else deterministic
            if llm:
                grade = _llm_verify_chunk(chunk_text, query, llm)
                if grade is None:
                    grade = _deterministic_verify_chunk(chunk_text, query, concepts, faiss_sim)
            else:
                grade = _deterministic_verify_chunk(chunk_text, query, concepts, faiss_sim)

            is_relevant = grade.get("is_relevant", False)
            verifier_score = float(grade.get("relevance_score", 0.0))
            direct_answer = grade.get("direct_answer", False)
            context_match = grade.get("context_match", True)
            reason = grade.get("reason", "")

            comp_score = _composite_score(faiss_sim, verifier_score, direct_answer, context_match)

            # Record per-chunk audit trail
            cv: ChunkVerification = {
                "chunk_index": metadata.get("chunk_index", rank_idx),
                "filename": metadata.get("filename", "unknown"),
                "page_number": metadata.get("page_number"),
                "retrieval_rank": retrieval_rank,
                "faiss_similarity": round(faiss_sim, 4),
                "is_relevant": is_relevant,
                "direct_answer": direct_answer,
                "context_match": context_match,
                "verifier_score": round(verifier_score, 4),
                "composite_score": comp_score,
                "accepted": is_relevant,
                "rejection_reason": "" if is_relevant else reason,
                "snippet": chunk_text[:120],
            }
            chunk_verifications.append(cv)

            if is_relevant:
                enriched = dict(doc)
                enriched["verifier_score"] = verifier_score
                enriched["composite_score"] = comp_score
                enriched["direct_answer"] = direct_answer
                enriched["context_match"] = context_match
                enriched["grade_reason"] = reason
                graded_docs.append(enriched)

        # Rank accepted chunks by composite score (highest first)
        graded_docs.sort(key=lambda d: d.get("composite_score", 0.0), reverse=True)

        # Limit to TOP_K_RETRIEVAL
        graded_docs = graded_docs[: settings.TOP_K_RETRIEVAL]

        # Compute overall verification score
        accepted_count = len(graded_docs)
        rejected_count = len(candidates) - accepted_count

        if not graded_docs:
            verification_status = "NO_RELEVANT_DOCS"
            verification_score = 0.0
            verification_reason = (
                f"All {len(candidates)} retrieved chunks failed contextual verification. "
                f"Common causes: STI/LTI context collision, role mismatch, or insufficient "
                f"term overlap. Try rephrasing your query."
            )
        else:
            scores = [d.get("composite_score", 0.0) for d in graded_docs]
            verification_score = round(sum(scores) / len(scores), 3)
            if verification_score >= settings.VERIFIER_DIRECT_THRESHOLD:
                verification_status = "PASSED"
                verification_reason = (
                    f"High-confidence evidence: {accepted_count} chunk(s) passed "
                    f"contextual verification (rejected {rejected_count})."
                )
            else:
                verification_status = "PARTIAL"
                verification_reason = (
                    f"Moderate evidence: {accepted_count} chunk(s) accepted "
                    f"(score={verification_score}, rejected {rejected_count})."
                )

    # Build readable debug summary for the UI execution trace
    accepted_chunks = [cv for cv in chunk_verifications if cv["accepted"]]
    rejected_chunks = [cv for cv in chunk_verifications if not cv["accepted"]]

    debug_lines = []
    for cv in chunk_verifications:
        status_icon = "✓ ACCEPTED" if cv["accepted"] else "✗ REJECTED"
        page_str = f" p.{cv['page_number']}" if cv.get("page_number") else ""
        debug_lines.append(
            f"{status_icon} | Rank {cv['retrieval_rank']} | "
            f"{cv['filename']}{page_str} | "
            f"sim={cv['faiss_similarity']:.3f} verifier={cv['verifier_score']:.3f} "
            f"composite={cv['composite_score']:.3f}"
            + (f" | Reason: {cv['rejection_reason']}" if not cv["accepted"] else "")
        )

    thought = (
        f"Evaluated {len(chunk_verifications)} candidates → "
        f"{len(accepted_chunks)} accepted, {len(rejected_chunks)} rejected. "
        f"Status: {verification_status} (score={verification_score})."
    )

    step_log: AgentStep = {
        "agent": "Verifier Agent",
        "action": "Contextual Relevance & Anti-Hallucination Verification",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "status": verification_status,
            "verification_score": verification_score,
            "candidates_evaluated": len(chunk_verifications),
            "accepted_chunks": len(accepted_chunks),
            "rejected_chunks": len(rejected_chunks),
            "reason": verification_reason,
            "chunk_audit": debug_lines,
        }
    }
    steps.append(step_log)

    return {
        "chunk_verifications": chunk_verifications,
        "graded_documents": graded_docs,
        "verification_score": verification_score,
        "verification_status": verification_status,
        "verification_reason": verification_reason,
        "steps_executed": steps
    }


