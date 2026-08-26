import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.config import settings, BASE_DIR
from backend.app.database.session import init_db
from backend.app.rag.ingest import seed_sample_documents
from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.api.history import router as history_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initialize database and seed sample documents."""
    print("================================================================")
    print(f"[*] Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    print("================================================================")
    
    # 1. Initialize SQLite tables
    init_db()
    print("[Startup] Database tables initialized.")

    # 2. Seed knowledge base with initial sample enterprise documents
    seed_sample_documents()

    yield
    print("[Shutdown] Enterprise Knowledge Assistant stopped.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Multi-Agent Enterprise Knowledge Assistant powered by LangGraph, LangChain, FAISS, Hugging Face, and SQLite.",
    lifespan=lifespan
)

# CORS middleware for open web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(history_router)


@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    from backend.app.rag.vectorstore import vector_store_manager
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "llm_provider": "OpenAI" if (settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-")) else "Local Fallback / Hybrid",
        "llm_model": settings.OPENAI_MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "total_faiss_vectors": vector_store_manager.get_total_indexed_documents()
    }


# Mount Frontend UI Static Files
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))
