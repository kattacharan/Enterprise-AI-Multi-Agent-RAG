from backend.app.api.chat import router as chat_router
from backend.app.api.documents import router as documents_router
from backend.app.api.history import router as history_router

__all__ = ["chat_router", "documents_router", "history_router"]
