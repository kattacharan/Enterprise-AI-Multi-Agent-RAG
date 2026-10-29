import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

# Determine base directory (saicharan root directory)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Agent Enterprise Knowledge Assistant"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM Settings (OpenAI / Fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.2

    # Hugging Face Embeddings Settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # FAISS Vector Database Storage
    FAISS_INDEX_DIR: str = str(BASE_DIR / "data" / "vectorstore")
    # TOP_K_RETRIEVAL: final number of chunks passed to Synthesizer
    TOP_K_RETRIEVAL: int = 4
    # CANDIDATE_K: larger initial pool fetched from FAISS before contextual verification
    CANDIDATE_K: int = 12
    SIMILARITY_THRESHOLD: float = 0.30

    # Verifier thresholds
    # Minimum verifier score for a chunk to pass as evidence
    VERIFIER_ACCEPT_THRESHOLD: float = 0.45
    # Score above which chunk is rated "direct answer" (high confidence)
    VERIFIER_DIRECT_THRESHOLD: float = 0.70

    # Evidence ranking composite score weights (must sum to ~1.0)
    RANK_WEIGHT_SEMANTIC: float = 0.30      # FAISS similarity contribution
    RANK_WEIGHT_CONTEXT: float = 0.35       # Context / entity match contribution
    RANK_WEIGHT_DIRECT: float = 0.25        # Direct-answer match contribution
    RANK_WEIGHT_VERIFIER: float = 0.10      # Overall verifier score contribution

    # Ingestion: extract PDF page numbers
    INGEST_PDF_PAGE_NUMBERS: bool = True
    # Ingestion: chunk size (chars) and overlap
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 120

    # SQLite Database Storage
    SQLITE_DB_PATH: str = f"sqlite:///{BASE_DIR / 'data' / 'enterprise_assistant.db'}"

    # Sample Documents Directory
    SAMPLE_DOCS_DIR: str = str(BASE_DIR / "data" / "sample_docs")

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure required directories exist
os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
os.makedirs(Path(settings.SAMPLE_DOCS_DIR), exist_ok=True)
os.makedirs(BASE_DIR / "data", exist_ok=True)
