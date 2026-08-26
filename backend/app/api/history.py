import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import Conversation, Message

router = APIRouter(prefix="/api/conversations", tags=["Conversation History & Memory"])


@router.get("")
def list_conversations(db: Session = Depends(get_db)):
    """Returns all conversations ordered by recent activity."""
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "message_count": len(c.messages)
        }
        for c in conversations
    ]


@router.post("")
def create_conversation(title: str = "New Conversation", db: Session = Depends(get_db)):
    """Creates a new conversation session."""
    conv = Conversation(
        id=str(uuid.uuid4()),
        title=title
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat()
    }


@router.get("/{conversation_id}")
def get_conversation_history(conversation_id: str, db: Session = Depends(get_db)):
    """Fetches all messages, citations, and agent execution steps for a specific conversation."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found."
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "citations": m.citations or [],
                "agent_steps": m.agent_steps or [],
                "verification_score": m.verification_score,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in messages
        ]
    }


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Deletes a conversation and its messages."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found."
        )

    db.delete(conv)
    db.commit()
    return {"success": True, "message": "Conversation deleted."}
