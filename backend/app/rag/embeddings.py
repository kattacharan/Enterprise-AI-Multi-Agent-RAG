import os
import hashlib
import numpy as np
from typing import List
from backend.app.config import settings


try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:
        pass


class LightweightFallbackEmbeddings(Embeddings):
    """
    Zero-dependency deterministic semantic vectorizer fallback.
    Ensures 100% crash-free bootstrap even if HuggingFace/PyTorch
    is not yet downloaded or in restricted environments.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def __call__(self, text: str) -> List[float]:
        return self._text_to_vector(text)

    def _text_to_vector(self, text: str) -> List[float]:
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vec.tolist()
        
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "to", "in", "for", "of", "what", "are", "our", "how", "can"}

        # Word unigrams
        for w in tokens:
            weight = 0.2 if w in stop_words else (1.0 + min(len(w) * 0.1, 1.5))
            h = int(hashlib.sha256(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += weight
        
        # Word bigrams for multi-word phrase matching
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]}_{tokens[i+1]}"
            h = int(hashlib.sha256(bg.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 2.0
        
        # L2 normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)


import warnings

def get_embedding_model():
    """
    Returns Hugging Face sentence-transformers embedding model.
    Falls back gracefully if dependencies are missing.
    """
    warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings was deprecated.*")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            print(f"[Embeddings] Notice: HuggingFaceEmbeddings not available ({e}). Using robust fallback embeddings.")
            return LightweightFallbackEmbeddings(dimension=384)
