from typing import TypedDict, List, Dict, Any, Optional


class AgentStep(TypedDict):
    agent: str
    action: str
    thought: str
    timestamp: str
    details: Optional[Dict[str, Any]]


class Citation(TypedDict):
    index: int
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float


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
    
    # Retriever Agent Output
    documents: List[Dict[str, Any]]
    
    # Verifier Agent Output
    graded_documents: List[Dict[str, Any]]
    verification_score: float
    verification_status: str  # "PASSED", "PARTIAL", "NO_RELEVANT_DOCS"
    verification_reason: str
    
    # Synthesizer Agent Output
    response: str
    citations: List[Citation]
    
    # Execution Timeline for UI Visualizer
    steps_executed: List[AgentStep]
