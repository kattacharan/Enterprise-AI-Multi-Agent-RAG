from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from backend.app.agents.state import AgentState
from backend.app.agents.planner import planner_node
from backend.app.agents.retriever import retriever_node
from backend.app.agents.verifier import verifier_node
from backend.app.agents.synthesizer import synthesizer_node


def route_after_planner(state: AgentState) -> str:
    """Conditional router: Determines if retrieval is required."""
    if state.get("requires_retrieval", True):
        return "retriever"
    return "synthesizer"


def build_agent_graph():
    """
    Constructs and compiles the LangGraph Multi-Agent Workflow:
    Planner -> (Retriever -> Verifier -> Synthesizer) | Synthesizer -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add agent nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Set entry point
    workflow.set_entry_point("planner")

    # 3. Add conditional edge from planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "synthesizer": "synthesizer"
        }
    )

    # 4. Connect RAG sequence: retriever -> verifier -> synthesizer
    workflow.add_edge("retriever", "verifier")
    workflow.add_edge("verifier", "synthesizer")

    # 5. Connect terminal edge
    workflow.add_edge("synthesizer", END)

    # Compile the graph
    app = workflow.compile()
    return app


_compiled_graph = None


def get_compiled_graph():
    """Lazily compiles and returns the LangGraph workflow singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        try:
            _compiled_graph = build_agent_graph()
        except Exception as e:
            print(f"[LangGraph] Warning: Graph compilation failed ({e}). Falling back to sequential execution.")
            _compiled_graph = False
    return _compiled_graph


# Pre-compile singleton for direct access if supported
try:
    compiled_graph = build_agent_graph()
except Exception:
    compiled_graph = None


def execute_multi_agent_flow(
    query: str,
    chat_history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Executes the Multi-Agent workflow via compiled LangGraph StateGraph,
    with automatic graceful sequential fallback.
    """
    initial_state: AgentState = {
        "query": query,
        "chat_history": chat_history or [],
        "requires_retrieval": True,
        "rewritten_query": query,
        "documents": [],
        "graded_documents": [],
        "verification_score": 0.0,
        "verification_status": "PENDING",
        "verification_reason": "",
        "response": "",
        "citations": [],
        "steps_executed": []
    }

    graph = get_compiled_graph()
    if graph:
        try:
            result = graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"[LangGraph] Graph invocation error: {e}. Executing sequential pipeline fallback.")

    # Resilient sequential execution fallback
    state = dict(initial_state)
    planner_res = planner_node(state)
    state.update(planner_res)

    if state.get("requires_retrieval", True):
        retriever_res = retriever_node(state)
        state.update(retriever_res)

        verifier_res = verifier_node(state)
        state.update(verifier_res)

    synthesizer_res = synthesizer_node(state)
    state.update(synthesizer_res)

    return state
