import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Simple fallback text splitter if needed
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=600, chunk_overlap=100, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    if end >= len(text):
                        break
                    start += self.chunk_size - self.chunk_overlap
                return chunks

from backend.app.config import settings
from backend.app.database.session import SessionLocal
from backend.app.database.models import DocumentRecord, DocumentChunkRecord
from backend.app.rag.vectorstore import vector_store_manager


def load_file_content(file_path: str) -> str:
    """Reads content from .txt, .md, or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md", ".markdown", ".csv", ".json"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text_pages)
        except Exception as e:
            print(f"[Ingest] Error reading PDF {file_path}: {e}")
            return ""
    else:
        # Fallback to UTF-8 read
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def ingest_document(
    file_path: str,
    original_filename: Optional[str] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Chunks document, extracts metadata, indexes vectors in FAISS, and persists records in SQLite.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        filename = original_filename or os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_ext = os.path.splitext(filename)[1].lower()

        # 1. Read document text
        content = load_file_content(file_path)
        if not content.strip():
            raise ValueError(f"Document {filename} is empty or unreadable.")

        # 2. Split into semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        raw_chunks = text_splitter.split_text(content)

        # 3. Clean up any existing record for this filename to prevent duplicate entries
        existing_doc = db.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()
        if existing_doc:
            db.delete(existing_doc)
            db.flush()

        # Create fresh document record in SQLite
        doc_id = str(uuid.uuid4())
        doc_record = DocumentRecord(
            id=doc_id,
            filename=filename,
            file_path=file_path,
            file_type=file_ext.replace(".", "") or "txt",
            file_size_bytes=file_size,
            chunk_count=len(raw_chunks),
            status="indexed"
        )
        db.add(doc_record)
        db.flush()

        # 4. Prepare LangChain Document objects and DB chunk records
        langchain_docs: List[Document] = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks)
            }
            
            # DB record
            chunk_record = DocumentChunkRecord(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=idx,
                content=chunk_text,
                metadata_json=chunk_metadata,
                faiss_doc_id=f"{doc_id}_{idx}"
            )
            db.add(chunk_record)

            # LangChain Doc for FAISS
            langchain_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
            )

        # 5. Index into FAISS
        vector_store_manager.add_documents(langchain_docs)

        db.commit()
        print(f"[Ingest] Ingested '{filename}' -> {len(langchain_docs)} chunks indexed.")
        return {
            "document_id": doc_id,
            "filename": filename,
            "chunk_count": len(raw_chunks),
            "status": "indexed"
        }

    except Exception as e:
        db.rollback()
        print(f"[Ingest] Ingestion failed for {file_path}: {e}")
        raise e
    finally:
        if should_close_db:
            db.close()


def seed_sample_documents():
    """
    Checks if documents exist in SQLite; if not, ingests all sample documents
    from data/sample_docs/ on application bootstrap.
    """
    db = SessionLocal()
    try:
        existing_count = db.query(DocumentRecord).count()
        if existing_count > 0:
            print(f"[Seed] Database already contains {existing_count} indexed documents.")
            return

        sample_dir = Path(settings.SAMPLE_DOCS_DIR)
        if not sample_dir.exists():
            return

        sample_files = list(sample_dir.glob("*.md")) + list(sample_dir.glob("*.txt"))
        if not sample_files:
            print(f"[Seed] No sample documents found in {sample_dir}.")
            return

        print(f"[Seed] Initializing knowledge base with {len(sample_files)} enterprise documents...")
        for file_path in sample_files:
            ingest_document(str(file_path), db=db)

        print("[Seed] Knowledge base seeding complete!")
    except Exception as e:
        print(f"[Seed] Error seeding sample documents: {e}")
    finally:
        db.close()
