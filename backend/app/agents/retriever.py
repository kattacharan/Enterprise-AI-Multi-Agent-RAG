from datetime import datetime
from typing import Dict, Any, List
from backend.app.config import settings
from backend.app.agents.state import AgentState, AgentStep
from backend.app.rag.vectorstore import vector_store_manager


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Retriever Agent:
    Executes semantic similarity search against the FAISS vector database
    using the optimized query and Hugging Face embeddings.
    """
    search_query = state.get("rewritten_query") or state.get("query", "")
    steps = list(state.get("steps_executed", []))

    # Perform similarity search with FAISS
    scored_docs = vector_store_manager.similarity_search(
        query=search_query,
        k=settings.TOP_K_RETRIEVAL,
        threshold=settings.SIMILARITY_THRESHOLD
    )

    documents: List[Dict[str, Any]] = []
    filenames_found = set()

    for doc, score in scored_docs:
        doc_dict = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        documents.append(doc_dict)
        if "filename" in doc.metadata:
            filenames_found.add(doc.metadata["filename"])

    thought = (
        f"Retrieved {len(documents)} candidate chunks from FAISS "
        f"across sources: {list(filenames_found) if filenames_found else 'None'}. "
        f"Top similarity score: {documents[0]['score'] if documents else 0.0}."
    )

    step_log: AgentStep = {
        "agent": "Retriever Agent",
        "action": "Semantic Vector Search (FAISS)",
        "thought": thought,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "details": {
            "search_query": search_query,
            "chunks_retrieved": len(documents),
            "sources": list(filenames_found),
            "top_score": documents[0]["score"] if documents else 0.0
        }
    }
    steps.append(step_log)

    return {
        "documents": documents,
        "steps_executed": steps
    }
