import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from backend.app.config import settings
from backend.app.rag.embeddings import get_embedding_model


class VectorStoreManager:
    """
    Manages the FAISS Vector Database for enterprise document retrieval.
    Handles indexing, persistence, and similarity search with score thresholding.
    """
    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or settings.FAISS_INDEX_DIR
        self.embeddings = get_embedding_model()
        self._vectorstore = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """Loads FAISS index from disk if available."""
        index_file = os.path.join(self.index_dir, "index.faiss")
        pkl_file = os.path.join(self.index_dir, "index.pkl")
        
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                from langchain_community.vectorstores import FAISS
                self._vectorstore = FAISS.load_local(
                    self.index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[VectorStore] Successfully loaded FAISS index from {self.index_dir}")
            except Exception as e:
                print(f"[VectorStore] Could not load existing FAISS index: {e}")
                self._vectorstore = None
        else:
            self._vectorstore = None

    def add_documents(self, documents: List[Document]) -> int:
        """Adds LangChain documents to the FAISS index and persists to disk."""
        if not documents:
            return 0

        try:
            from langchain_community.vectorstores import FAISS
            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                self._vectorstore.add_documents(documents)
            
            # Save index to disk
            os.makedirs(self.index_dir, exist_ok=True)
            self._vectorstore.save_local(self.index_dir)
            print(f"[VectorStore] Indexed {len(documents)} chunks into FAISS.")
            return len(documents)
        except Exception as e:
            print(f"[VectorStore] Error adding documents: {e}")
            raise e

    def similarity_search(
        self,
        query: str,
        k: int = settings.TOP_K_RETRIEVAL,
        threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> List[Tuple[Document, float]]:
        """
        Executes semantic search against FAISS and returns documents with relevance scores.
        Scores are normalized where 1.0 is maximum similarity.
        """
        if self._vectorstore is None:
            self._load_vectorstore()
            if self._vectorstore is None:
                return []

        try:
            # Over-fetch candidates to ensure top-k unique chunks after deduplication
            fetch_k = max(k * 3, 10)
            results_with_scores = self._vectorstore.similarity_search_with_score(query, k=fetch_k)
            
            scored_docs: List[Tuple[Document, float]] = []
            seen_content = set()

            for doc, distance in results_with_scores:
                # Deduplication key: normalized content snippet or chunk id
                content_key = " ".join(doc.page_content.strip().split())[:120]
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)

                # Convert distance to similarity score in [0.0, 1.0]
                sim_score = max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
                
                # Filter by similarity threshold
                if sim_score >= threshold:
                    doc.metadata["score"] = round(sim_score, 4)
                    scored_docs.append((doc, sim_score))

                if len(scored_docs) >= k:
                    break
            
            # Sort descending by similarity score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs
        except Exception as e:
            print(f"[VectorStore] Search error: {e}")
            return []

    def get_total_indexed_documents(self) -> int:
        """Returns the number of vectors indexed in FAISS."""
        if self._vectorstore is not None and hasattr(self._vectorstore, "index"):
            return self._vectorstore.index.ntotal
        return 0

    def reset_index(self):
        """Clears the FAISS index on disk."""
        import shutil
        if os.path.exists(self.index_dir):
            for file in os.listdir(self.index_dir):
                file_path = os.path.join(self.index_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"[VectorStore] Error clearing {file_path}: {e}")
        self._vectorstore = None


# Singleton instance
vector_store_manager = VectorStoreManager()
