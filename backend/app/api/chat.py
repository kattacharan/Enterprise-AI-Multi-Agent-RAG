import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import Conversation, Message
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.agents.graph import execute_multi_agent_flow

router = APIRouter(prefix="/api", tags=["Chat & Multi-Agent"])


@router.post("/chat", response_model=ChatResponse)
def handle_chat_query(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Executes the LangGraph Multi-Agent Workflow:
    Planner -> Retriever -> Verifier -> Synthesizer
    Persists query and grounded response with citations in SQLite.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )

    # 1. Get or create conversation session
    conversation_id = request.conversation_id
    if not conversation_id:
        # Create new conversation
        conv_title = query[:40] + ("..." if len(query) > 40 else "")
        conv = Conversation(
            id=str(uuid.uuid4()),
            title=conv_title
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conversation_id = conv.id
    else:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            conv = Conversation(
                id=conversation_id,
                title=query[:40] + ("..." if len(query) > 40 else "")
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)

    # 2. Fetch recent conversation history for memory context
    recent_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(6)
        .all()
    )
    # Reverse to chronological order
    recent_messages.reverse()
    chat_history = [{"role": m.role, "content": m.content} for m in recent_messages]

    # 3. Save User Message
    user_msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=query
    )
    db.add(user_msg)
    db.commit()

    # 4. Run LangGraph Multi-Agent Workflow
    try:
        agent_result = execute_multi_agent_flow(query=query, chat_history=chat_history)
    except Exception as e:
        print(f"[API /chat] Execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow execution error: {str(e)}"
        )

    response_text = agent_result.get("response", "")
    citations = agent_result.get("citations", [])
    agent_steps = agent_result.get("steps_executed", [])
    verification_score = agent_result.get("verification_score")
    verification_status = agent_result.get("verification_status")

    # 5. Save Assistant Message with full RAG metadata & execution tracing
    asst_msg_id = str(uuid.uuid4())
    asst_msg = Message(
        id=asst_msg_id,
        conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        citations=citations,
        agent_steps=agent_steps,
        verification_score=verification_score
    )
    db.add(asst_msg)
    conv.updated_at = datetime.now(timezone.utc)
    db.commit()

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=asst_msg_id,
        role="assistant",
        response=response_text,
        citations=citations,
        agent_steps=agent_steps,
        verification_score=verification_score,
        verification_status=verification_status,
        created_at=datetime.now(timezone.utc).isoformat()
    )
