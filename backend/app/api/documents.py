import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.database.session import get_db
from backend.app.database.models import DocumentRecord, DocumentChunkRecord
from backend.app.schemas.document import DocumentResponse, DocumentListResponse, IngestionResponse
from backend.app.rag.ingest import ingest_document, seed_sample_documents
from backend.app.rag.vectorstore import vector_store_manager

router = APIRouter(prefix="/api/documents", tags=["Document Ingestion & Management"])

UPLOAD_DIR = Path(settings.SAMPLE_DOCS_DIR) / "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=IngestionResponse)
async def upload_and_ingest_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a document (.pdf, .txt, .md), chunks it, computes Hugging Face embeddings,
    indexes into FAISS vector database, and stores records in SQLite.
    """
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown", ".csv", ".json"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {list(allowed_extensions)}"
        )

    # Save file locally
    saved_file_path = UPLOAD_DIR / file.filename
    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {str(e)}"
        )

    # Ingest document
    try:
        result = ingest_document(str(saved_file_path), original_filename=file.filename, db=db)
        
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == result["document_id"]).first()
        doc_resp = DocumentResponse(
            id=doc_record.id,
            filename=doc_record.filename,
            file_type=doc_record.file_type,
            file_size_bytes=doc_record.file_size_bytes,
            chunk_count=doc_record.chunk_count,
            status=doc_record.status,
            created_at=doc_record.created_at.isoformat()
        )
        return IngestionResponse(
            success=True,
            message=f"Successfully ingested '{file.filename}' ({result['chunk_count']} chunks indexed).",
            document=doc_resp
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    """
    Returns all indexed documents, chunk statistics, and total vectors in FAISS.
    """
    docs = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()
    total_faiss = vector_store_manager.get_total_indexed_documents()

    doc_list = [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_type=d.file_type,
            file_size_bytes=d.file_size_bytes,
            chunk_count=d.chunk_count,
            status=d.status,
            created_at=d.created_at.isoformat()
        )
        for d in docs
    ]

    return DocumentListResponse(
        total_documents=len(docs),
        total_vectors_in_faiss=total_faiss,
        documents=doc_list
    )


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Deletes a document from the database and triggers index rebuild."""
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    file_path = doc.file_path
    db.delete(doc)
    db.commit()

    # If physical file exists in uploads, delete it
    if os.path.exists(file_path) and "uploads" in file_path:
        try:
            os.remove(file_path)
        except Exception:
            pass

    # Rebuild index with remaining documents
    _rebuild_faiss_index(db)

    return {"success": True, "message": f"Document '{doc.filename}' deleted and FAISS index updated."}


@router.post("/reindex")
def trigger_reindex(db: Session = Depends(get_db)):
    """Re-indexes all sample and uploaded documents."""
    _rebuild_faiss_index(db)
    return {
        "success": True,
        "message": "Knowledge base re-indexed successfully.",
        "total_vectors": vector_store_manager.get_total_indexed_documents()
    }


def _rebuild_faiss_index(db: Session):
    """Internal helper to clear and re-populate FAISS from SQLite chunks."""
    from langchain_core.documents import Document

    vector_store_manager.reset_index()
    chunks = db.query(DocumentChunkRecord).all()
    
    if not chunks:
        seed_sample_documents()
        return

    docs_to_index = []
    for chunk in chunks:
        docs_to_index.append(
            Document(
                page_content=chunk.content,
                metadata=chunk.metadata_json or {}
            )
        )
    
    if docs_to_index:
        vector_store_manager.add_documents(docs_to_index)
