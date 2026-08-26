from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: str
    created_at: str


class DocumentListResponse(BaseModel):
    total_documents: int
    total_vectors_in_faiss: int
    documents: List[DocumentResponse]


class IngestionResponse(BaseModel):
    success: bool
    message: str
    document: DocumentResponse
