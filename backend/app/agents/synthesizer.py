from datetime import datetime
from typing import Dict, Any, List
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep, Citation


def _format_context_with_citations(graded_docs: List[Dict[str, Any]]) -> (str, List[Citation]):
    """Formats verified chunks with bracketed citation markers."""
    context_str = ""
    citations: List[Citation] = []

    for i, doc in enumerate(graded_docs, start=1):
        content = doc.get("content", "").strip()
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "Enterprise Document")
        chunk_idx = metadata.get("chunk_index", 0)
        doc_id = metadata.get("document_id", "")
        score = doc.get("score", 0.0)

        context_str += f"\n--- Source [{i}]: {filename} (Chunk {chunk_idx}) ---\n{content}\n"
        
        # Snippet preview (first 180 chars)
        snippet = content[:180] + "..." if len(content) > 180 else content
        citations.append({
            "index": i,
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": chunk_idx,
            "snippet": snippet,
            "score": round(score, 3)
        })

    return context_str, citations


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizer Agent:
    1. Synthesizes a factual, grounded response using verified document context.
    2. Embeds precise citation markers [1], [2] referencing source documents.
    3. Handles conversational intents and missing knowledge gracefully.
    """
    query = state.get("query", "")
    history = state.get("chat_history", [])
    requires_retrieval = state.get("requires_retrieval", False)
    verification_status = state.get("verification_status", "PASSED")
    graded_docs = state.get("graded_documents", [])
    steps = list(state.get("steps_executed", []))

    citations: List[Citation] = []
    response = ""

    # Case 1: Pure Conversational / Greeting / Capabilities
    if not requires_retrieval:
        q_lower = query.lower()
        if any(p in q_lower for p in ["what can you do", "what can ypu do", "how can you help", "capabilities", "what do you do", "about yourself"]):
            response = (
                "### Enterprise Knowledge Assistant Capabilities\n\n"
                "I am an autonomous **Multi-Agent RAG Assistant** powered by LangGraph, FAISS vector search, and FastAPI. "
                "Here is what I can help you with:\n\n"
                "- **Company Policies & HR**: Remote work guidelines, WFH equipment stipend ($1,500), PTO leave policy (22 days), and professional learning allowance ($2,000).\n"
                "- **Engineering Standards**: Code review requirements (>=2 approvals), unit test coverage standards (>=80%), REST API design, and microservices resilience.\n"
                "- **Cloud Security & IAM**: Least privilege (PoLP), AWS KMS AES-256 encryption, TLS 1.3 standards, and security vulnerability remediation SLAs.\n"
                "- **Custom Knowledge Base**: Ingest and query your own `.pdf`, `.md`, or `.txt` enterprise documents via the upload manager.\n\n"
                "**Try asking a specific question** or click any sample prompt to get started!"
            )
        else:
            response = (
                "Hello! I am your **Enterprise Knowledge Assistant** powered by LangGraph multi-agent RAG. "
                "I can help you search and verify information from internal company policies, engineering standards, "
                "and cloud security guidelines.\n\n"
                "How can I assist you today?"
            )
        thought = "Synthesized conversational greeting / capabilities overview."

    # Case 2: No relevant documents found after retrieval/verification
    elif verification_status == "NO_RELEVANT_DOCS" or not graded_docs:
        response = (
            f"I searched the internal enterprise knowledge base for **\"{query}\"**, but could not find any relevant "
            "approved documentation matching your inquiry.\n\n"
            "Please consider verifying your keywords or uploading additional reference documentation to the knowledge base."
        )
        thought = "Synthesized safe no-hallucination notice because no verified documents were found."

    # Case 3: Verified RAG Synthesis
    else:
        context_str, citations = _format_context_with_citations(graded_docs)

        # If OpenAI API is available, use LLM for synthesis
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                
                llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    temperature=settings.OPENAI_TEMPERATURE,
                    api_key=settings.OPENAI_API_KEY
                )

                system_prompt = (
                    "You are a professional Enterprise Knowledge Assistant. Answer the user's question accurately, "
                    "strictly relying on the provided verified context passages below.\n"
                    "Rules:\n"
                    "1. When stating facts, ALWAYS cite the source using numbered brackets like [1] or [2] corresponding to the source number.\n"
                    "2. Structure your answer clearly with bullet points and bold headings where appropriate.\n"
                    "3. Do NOT make up information not present in the verified context.\n\n"
                    f"VERIFIED ENTERPRISE CONTEXT:\n{context_str}"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", f"User Question: {query}")
                ])

                chain = prompt | llm
                res = chain.invoke({})
                response = res.content.strip()
                thought = f"Synthesized grounded response with {len(citations)} citations via OpenAI LLM."
            except Exception as e:
                print(f"[Synthesizer Agent] LLM generation error: {e}. Falling back to deterministic synthesis.")
                response = _deterministic_synthesis(query, graded_docs, citations)
                thought = f"Synthesized response using deterministic RAG formatter with {len(citations)} citations."
        else:
            # Deterministic grounded synthesis fallback
            response = _deterministic_synthesis(query, graded_docs, citations)
            thought = f"Synthesized response using deterministic grounded synthesizer with {len(citations)} citations."

    step_log: AgentStep = {
        "agent": "Synthesizer Agent",
        "action": "Response Generation & Citation Formatting",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "citations_generated": len(citations),
            "response_length_chars": len(response)
        }
    }
    steps.append(step_log)

    return {
        "response": response,
        "citations": citations,
        "steps_executed": steps
    }


def _deterministic_synthesis(query: str, graded_docs: List[Dict[str, Any]], citations: List[Citation]) -> str:
    """Generates structured response with citation references without external API dependencies."""
    output_parts = [
        f"Based on the verified enterprise documentation for **\"{query}\"**:\n"
    ]

    for i, doc in enumerate(graded_docs, start=1):
        meta = doc.get("metadata", {})
        filename = meta.get("filename", "Internal Document")
        content = doc.get("content", "").strip()

        output_parts.append(f"#### From `{filename}` [{i}]:")
        output_parts.append(f"{content}\n")

    return "\n".join(output_parts)
