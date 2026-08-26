import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database.session import Base


def utc_now():
    return datetime.now(timezone.utc)


class Conversation(Base):
    """Stores conversation sessions."""
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    """Stores user queries, AI responses, citations, and agent execution steps."""
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    
    # RAG metadata and multi-agent execution tracing
    citations = Column(JSON, nullable=True, default=list)  # List of citation objects
    agent_steps = Column(JSON, nullable=True, default=list)  # Step-by-step agent execution trace
    verification_score = Column(Float, nullable=True)  # Hallucination / relevance score
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class DocumentRecord(Base):
    """Tracks ingested enterprise documents."""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(50), default="indexed")  # 'indexed', 'processing', 'failed'
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    chunks = relationship("DocumentChunkRecord", back_populates="document", cascade="all, delete-orphan")


class DocumentChunkRecord(Base):
    """Tracks individual chunks extracted and indexed in FAISS."""
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True, default=dict)
    faiss_doc_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    document = relationship("DocumentRecord", back_populates="chunks")
