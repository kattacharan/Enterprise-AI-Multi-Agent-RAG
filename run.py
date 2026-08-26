import os
import sys
import uvicorn
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.config import settings

if __name__ == "__main__":
    print("""
  ╔═══════════════════════════════════════════════════════════════╗
  ║         MULTI-AGENT ENTERPRISE KNOWLEDGE ASSISTANT            ║
  ║  LangGraph • FastAPI • FAISS • HuggingFace • SQLite • LangChain║
  ╚═══════════════════════════════════════════════════════════════╝
    """)
    print(f"[*] Starting API Server at http://{settings.HOST}:{settings.PORT}")
    print(f"[*] Interactive Swagger Docs: http://localhost:{settings.PORT}/docs")
    print(f"[*] Web UI Dashboard: http://localhost:{settings.PORT}")
    print(f"[*] Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"[*] LLM Provider: {'OpenAI (' + settings.OPENAI_MODEL_NAME + ')' if settings.OPENAI_API_KEY else 'Smart Grounded Fallback'}")
    print("-----------------------------------------------------------------")
    
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
