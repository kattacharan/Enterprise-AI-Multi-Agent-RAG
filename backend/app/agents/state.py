from typing import TypedDict, List, Dict, Any, Optional


class AgentStep(TypedDict):
    agent: str
    action: str
    thought: str
    timestamp: str
    details: Optional[Dict[str, Any]]


class ChunkVerification(TypedDict):
    """Per-chunk verifier decision for debugging and UI display."""
    chunk_index: int
    filename: str
    page_number: Optional[int]
    retrieval_rank: int
    faiss_similarity: float
    is_relevant: bool
    direct_answer: bool
    context_match: bool
    verifier_score: float
    composite_score: float
    accepted: bool
    rejection_reason: str
    snippet: str


class Citation(TypedDict):
    index: int
    document_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int]
    section: Optional[str]
    snippet: str
    score: float              # FAISS similarity score
    verification_score: float # Verifier confidence score
    direct_answer: bool


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph agent nodes.
    """
    # User Input & Context
    query: str
    chat_history: List[Dict[str, str]]

    # Planner Agent Output
    requires_retrieval: bool
    rewritten_query: str

    # Retriever Agent Output — full CANDIDATE_K pool before verification
    retrieved_candidates: List[Dict[str, Any]]
    # Legacy alias kept for graph.py sequential fallback compat
    documents: List[Dict[str, Any]]

    # Verifier Agent Output
    chunk_verifications: List[ChunkVerification]  # per-chunk audit trail
    graded_documents: List[Dict[str, Any]]
    verification_score: float
    verification_status: str  # "PASSED", "PARTIAL", "NO_RELEVANT_DOCS"
    verification_reason: str

    # Synthesizer Agent Output
    response: str
    citations: List[Citation]

    # Execution Timeline for UI Visualizer
    steps_executed: List[AgentStep]
