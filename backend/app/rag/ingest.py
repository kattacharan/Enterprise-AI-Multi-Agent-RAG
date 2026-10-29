import os
import re
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
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=600, chunk_overlap=120, separators=None):
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_section_heading(text: str) -> Optional[str]:
    """
    Returns the first Markdown heading found in a chunk, or None.
    Example: '## Remuneration Policy Objectives' → 'Remuneration Policy Objectives'
    """
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^#{1,4}\s+(.+)', line)
        if m:
            return m.group(1).strip()
    return None


def _load_pdf_with_pages(file_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {page_number (1-indexed), text} dicts for each PDF page.
    Falls back to concatenated text if pypdf is unavailable.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page_number": i + 1, "text": text})
        return pages
    except Exception as e:
        print(f"[Ingest] PDF page extraction error ({e}); falling back to full-text.")
        with open(file_path, "rb") as f:
            try:
                from pypdf import PdfReader
                reader = PdfReader(f)
                full = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                full = ""
        return [{"page_number": 1, "text": full}]


def load_file_content(file_path: str) -> str:
    """Reads content from .txt, .md, or .pdf files (plain text, for seed docs)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md", ".markdown", ".csv", ".json"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        pages = _load_pdf_with_pages(file_path)
        return "\n\n".join(p["text"] for p in pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def ingest_document(
    file_path: str,
    original_filename: Optional[str] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Chunks document, extracts rich metadata (page number, section heading),
    indexes vectors in FAISS, and persists records in SQLite.
    """
    should_close_db = False
    if db is None:
        db = SessionLocal()
        should_close_db = True

    try:
        filename = original_filename or os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_ext = os.path.splitext(filename)[1].lower()

        # 1. Read document — PDF gets per-page metadata; others get flat text
        is_pdf = file_ext == ".pdf"

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )

        # Build a list of (chunk_text, page_number, section) tuples
        chunks_with_meta: List[tuple] = []

        if is_pdf and settings.INGEST_PDF_PAGE_NUMBERS:
            pdf_pages = _load_pdf_with_pages(file_path)
            if not pdf_pages:
                raise ValueError(f"Document {filename} is empty or unreadable.")
            for page_data in pdf_pages:
                page_num = page_data["page_number"]
                page_text = page_data["text"]
                raw = text_splitter.split_text(page_text)
                for chunk_text in raw:
                    section = _extract_section_heading(chunk_text)
                    chunks_with_meta.append((chunk_text, page_num, section))
        else:
            content = load_file_content(file_path)
            if not content.strip():
                raise ValueError(f"Document {filename} is empty or unreadable.")
            raw = text_splitter.split_text(content)
            for chunk_text in raw:
                section = _extract_section_heading(chunk_text)
                chunks_with_meta.append((chunk_text, None, section))

        if not chunks_with_meta:
            raise ValueError(f"No usable chunks produced from {filename}.")

        # 2. Clean up any existing record for this filename
        existing_doc = db.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()
        if existing_doc:
            db.delete(existing_doc)
            db.flush()

        # 3. Create fresh document record
        doc_id = str(uuid.uuid4())
        doc_record = DocumentRecord(
            id=doc_id,
            filename=filename,
            file_path=file_path,
            file_type=file_ext.replace(".", "") or "txt",
            file_size_bytes=file_size,
            chunk_count=len(chunks_with_meta),
            status="indexed"
        )
        db.add(doc_record)
        db.flush()

        # 4. Prepare LangChain Document objects and DB chunk records
        langchain_docs: List[Document] = []
        for idx, (chunk_text, page_num, section) in enumerate(chunks_with_meta):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks_with_meta),
                "page_number": page_num,
                "section": section,
            }

            chunk_record = DocumentChunkRecord(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=idx,
                content=chunk_text,
                metadata_json=chunk_metadata,
                faiss_doc_id=f"{doc_id}_{idx}"
            )
            db.add(chunk_record)

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
            "chunk_count": len(chunks_with_meta),
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
