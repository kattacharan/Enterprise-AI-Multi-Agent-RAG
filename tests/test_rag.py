import os
import unittest
from pathlib import Path
from langchain_core.documents import Document
from backend.app.rag.vectorstore import VectorStoreManager
from backend.app.rag.embeddings import get_embedding_model
from backend.app.database.session import SessionLocal, init_db
from backend.app.database.models import DocumentRecord, DocumentChunkRecord
from backend.app.rag.ingest import ingest_document


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_embedding_model_initialization(self):
        """Verify embedding model generates dense vector embeddings."""
        model = get_embedding_model()
        vec = model.embed_query("Enterprise cloud security KMS AES-256")
        self.assertIsInstance(vec, list)
        self.assertTrue(len(vec) > 0)

    def test_vectorstore_indexing_and_search(self):
        """Verify FAISS vector store indexing and similarity retrieval in isolated test directory."""
        import tempfile
        import shutil
        test_dir = tempfile.mkdtemp()
        try:
            vsm = VectorStoreManager(index_dir=test_dir)
            docs = [
                Document(
                    page_content="Employees receive a one-time home office equipment stipend of $1500.",
                    metadata={"filename": "company_policies.md", "chunk_index": 0}
                ),
                Document(
                    page_content="Code review requires at least two peer approvals before merging.",
                    metadata={"filename": "engineering_standards.md", "chunk_index": 1}
                )
            ]
            indexed_count = vsm.add_documents(docs)
            self.assertEqual(indexed_count, 2)

            # Search for equipment stipend
            results = vsm.similarity_search("equipment reimbursement stipend", k=2, threshold=0.1)
            self.assertTrue(len(results) > 0)
            top_doc, score = results[0]
            self.assertIn("stipend", top_doc.page_content.lower())
            self.assertGreaterEqual(score, 0.1)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_document_ingestion_sqlite_and_faiss(self):
        """Verify full ingestion pipeline records metadata in SQLite."""
        db = SessionLocal()
        sample_path = Path("data/sample_docs/company_policies.md")
        if sample_path.exists():
            res = ingest_document(str(sample_path), db=db)
            self.assertIn("document_id", res)
            self.assertGreater(res["chunk_count"], 0)

            # Query SQLite
            doc_rec = db.query(DocumentRecord).filter(DocumentRecord.id == res["document_id"]).first()
            self.assertIsNotNone(doc_rec)
            self.assertEqual(doc_rec.filename, "company_policies.md")
        db.close()


if __name__ == "__main__":
    unittest.main()
