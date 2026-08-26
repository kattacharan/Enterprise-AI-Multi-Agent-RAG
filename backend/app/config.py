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
    TOP_K_RETRIEVAL: int = 4
    SIMILARITY_THRESHOLD: float = 0.30

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
