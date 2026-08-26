import os
from datetime import datetime
from typing import Dict, Any
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner / Query Router Agent:
    1. Determines whether the user query requires internal enterprise document retrieval.
    2. Optimizes and rewrites the query for semantic vector search against FAISS.
    """
    query = state.get("query", "").strip()
    history = state.get("chat_history", [])
    steps = list(state.get("steps_executed", []))

    # Conversational intent detection (greetings, identity, capabilities, general FAQs)
    conversational_patterns = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon", "greetings",
        "how are you", "who are you", "who r u", "what can you do", "what can ypu do",
        "what do you do", "how can you help", "tell me about yourself", "what are you",
        "help", "thanks", "thank you", "what are your capabilities", "how does this work"
    ]
    cleaned_query = "".join(c for c in query.lower() if c.isalnum() or c.isspace()).strip()

    requires_retrieval = True
    rewritten_query = query
    thought = ""

    is_conversational = (
        cleaned_query in conversational_patterns or
        (len(query.split()) <= 2 and any(cleaned_query.startswith(w) for w in ["hi", "hello", "hey", "thanks", "help"])) or
        any(cleaned_query.startswith(g) for g in ["hi ", "hello ", "hey ", "good morning", "good evening", "good afternoon", "thanks", "thank you"]) or
        any(phrase in cleaned_query for phrase in [
            "how are you", "who are you", "what can you do", "what can ypu do",
            "how can you help", "what do you do", "who made you", "about yourself"
        ])
    )

    if is_conversational:
        requires_retrieval = False
        thought = f"Query '{query}' classified as conversational / assistant capability question. Direct synthesis without retrieval."
    else:
        # Check if OpenAI LLM is available for advanced query planning & query expansion
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
                        "You are an expert Query Planning Agent for an enterprise knowledge retrieval system.\n"
                        "Given the user's latest query and conversation context:\n"
                        "1. Decide if enterprise documentation retrieval is needed (yes/no).\n"
                        "2. If yes, generate an optimized, dense semantic search query that captures key enterprise terms, policies, or technical concepts.\n"
                        "Respond in JSON format: {\"requires_retrieval\": true/false, \"optimized_query\": \"...\", \"rationale\": \"...\"}"
                    )),
                    ("user", f"Context: {history[-3:] if history else 'None'}\nQuery: {query}")
                ])
                
                chain = prompt | llm
                res = chain.invoke({})
                content = res.content.strip()
                
                # Parse JSON or fallback
                import json
                if "{" in content and "}" in content:
                    json_str = content[content.find("{"):content.rfind("}") + 1]
                    data = json.loads(json_str)
                    requires_retrieval = data.get("requires_retrieval", True)
                    rewritten_query = data.get("optimized_query", query)
                    thought = f"LLM Planner: {data.get('rationale', 'Optimized query for semantic vector search.')}"
            except Exception as e:
                print(f"[Planner Agent] LLM invocation error: {e}. Using deterministic planner.")
                requires_retrieval = True
                rewritten_query = query
                thought = f"Plan: Searching enterprise knowledge base for '{query}'"
        else:
            # Deterministic query optimization fallback
            requires_retrieval = True
            rewritten_query = query
            thought = f"Plan: Directing query '{query}' to FAISS semantic retriever."

    # Record step log for UI visualizer
    step_log: AgentStep = {
        "agent": "Planner Agent",
        "action": "Query Routing & Planning",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "requires_retrieval": requires_retrieval,
            "original_query": query,
            "rewritten_query": rewritten_query
        }
    }
    steps.append(step_log)

    return {
        "requires_retrieval": requires_retrieval,
        "rewritten_query": rewritten_query,
        "steps_executed": steps
    }
