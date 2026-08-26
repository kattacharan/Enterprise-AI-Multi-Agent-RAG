from datetime import datetime
from typing import Dict, Any, List
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Verifier / Hallucination Grader Agent:
    1. Evaluates retrieved document chunks against the user query for relevance and grounding.
    2. Filters out low-relevance or spurious chunks.
    3. Calculates an overall verification score to guard against hallucinations.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    steps = list(state.get("steps_executed", []))

    if not documents:
        verification_status = "NO_RELEVANT_DOCS"
        verification_score = 0.0
        verification_reason = "No candidate documents matched the retrieval threshold."
        graded_docs = []
    else:
        graded_docs: List[Dict[str, Any]] = []
        scores = []

        # If OpenAI API is available, perform LLM grading on retrieved chunks
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                
                llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL_NAME,
                    temperature=0.0,
                    api_key=settings.OPENAI_API_KEY
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are an expert Document Relevance and Grounding Grader.\n"
                        "Given a user query and a retrieved document chunk, determine if the chunk contains information relevant to answering the query.\n"
                        "Respond with a JSON object: {\"is_relevant\": true/false, \"relevance_score\": 0.0-1.0, \"reason\": \"...\"}"
                    )),
                    ("user", "Query: {query}\n\nChunk: {chunk}")
                ])

                chain = prompt | llm
                
                for doc in documents:
                    res = chain.invoke({"query": query, "chunk": doc["content"]})
                    content = res.content.strip()
                    import json
                    if "{" in content and "}" in content:
                        data = json.loads(content[content.find("{"):content.rfind("}") + 1])
                        if data.get("is_relevant", True):
                            doc["grade_reason"] = data.get("reason", "Relevant")
                            graded_docs.append(doc)
                            scores.append(float(data.get("relevance_score", doc.get("score", 0.8))))
                    else:
                        graded_docs.append(doc)
                        scores.append(doc.get("score", 0.5))

            except Exception as e:
                print(f"[Verifier Agent] LLM Grader error: {e}. Using deterministic score verification.")
                # Fallback to vector similarity score grading
                for doc in documents:
                    if doc.get("score", 0.0) >= settings.SIMILARITY_THRESHOLD:
                        graded_docs.append(doc)
                        scores.append(doc.get("score", 0.0))
        else:
            # Deterministic score verification
            for doc in documents:
                if doc.get("score", 0.0) >= settings.SIMILARITY_THRESHOLD:
                    graded_docs.append(doc)
                    scores.append(doc.get("score", 0.0))

        if not graded_docs:
            verification_status = "NO_RELEVANT_DOCS"
            verification_score = 0.0
            verification_reason = "Retrieved chunks failed relevance verification criteria."
        else:
            verification_score = round(sum(scores) / len(scores), 3) if scores else 0.5
            if verification_score >= 0.65:
                verification_status = "PASSED"
                verification_reason = f"High confidence grounding verified across {len(graded_docs)} passages."
            else:
                verification_status = "PARTIAL"
                verification_reason = f"Moderate relevance score ({verification_score}). Output verified with caveat."

    thought = f"Verifier evaluated {len(documents)} chunks -> {len(graded_docs)} passed verification. Status: {verification_status} (Score: {verification_score})."

    step_log: AgentStep = {
        "agent": "Verifier Agent",
        "action": "Hallucination & Relevance Verification",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "status": verification_status,
            "verification_score": verification_score,
            "passed_chunks": len(graded_docs),
            "reason": verification_reason
        }
    }
    steps.append(step_log)

    return {
        "graded_documents": graded_docs,
        "verification_score": verification_score,
        "verification_status": verification_status,
        "verification_reason": verification_reason,
        "steps_executed": steps
    }
