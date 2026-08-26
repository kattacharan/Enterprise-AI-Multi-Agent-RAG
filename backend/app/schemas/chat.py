from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="User's query or question")
    conversation_id: Optional[str] = Field(None, description="Existing conversation UUID")


class CitationModel(BaseModel):
    index: int
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float


class AgentStepModel(BaseModel):
    agent: str
    action: str
    thought: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    role: str = "assistant"
    response: str
    citations: List[CitationModel] = []
    agent_steps: List[AgentStepModel] = []
    verification_score: Optional[float] = None
    verification_status: Optional[str] = None
    created_at: str
