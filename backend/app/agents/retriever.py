from datetime import datetime
from typing import Dict, Any, List
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep
from backend.app.rag.vectorstore import vector_store_manager


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retriever Agent:
    Executes semantic similarity search against the FAISS vector database using
    CANDIDATE_K (larger pool), then returns all candidates for the Verifier to
    perform contextual filtering. Only TOP_K_RETRIEVAL evidence chunks reach
    the Synthesizer.
    """
    search_query = state.get("rewritten_query") or state.get("query", "")
    steps = list(state.get("steps_executed", []))

    # Fetch larger candidate pool (CANDIDATE_K) from FAISS
    scored_docs = vector_store_manager.similarity_search(
        query=search_query,
        k=settings.CANDIDATE_K,
        threshold=settings.SIMILARITY_THRESHOLD
    )

    candidates: List[Dict[str, Any]] = []
    filenames_found = set()
    candidate_debug = []

    for rank, (doc, score) in enumerate(scored_docs, start=1):
        metadata = doc.metadata
        doc_dict = {
            "content": doc.page_content,
            "metadata": metadata,
            "score": score,
            "retrieval_rank": rank,
        }
        candidates.append(doc_dict)
        if "filename" in metadata:
            filenames_found.add(metadata["filename"])
        # Debug line per chunk
        page_str = f" p.{metadata.get('page_number')}" if metadata.get("page_number") else ""
        candidate_debug.append(
            f"  Rank {rank}: {metadata.get('filename','?')}{page_str} | sim={score:.3f}"
        )

    thought = (
        f"Fetched {len(candidates)} candidate chunks (pool={settings.CANDIDATE_K}) from FAISS "
        f"across sources: {list(filenames_found) if filenames_found else 'None'}. "
        f"Top similarity: {candidates[0]['score']:.3f}." if candidates else
        f"No chunks found above threshold={settings.SIMILARITY_THRESHOLD}."
    )

    step_log: AgentStep = {
        "agent": "Retriever Agent",
        "action": "Semantic Vector Search (FAISS)",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "search_query": search_query,
            "candidate_pool_size": len(candidates),
            "top_k_final": settings.TOP_K_RETRIEVAL,
            "sources": list(filenames_found),
            "top_score": candidates[0]["score"] if candidates else 0.0,
            "candidate_breakdown": candidate_debug,
        }
    }
    steps.append(step_log)

    return {
        "retrieved_candidates": candidates,
        "documents": candidates,   # backward compat for sequential fallback
        "steps_executed": steps
    }

