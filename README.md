# Multi-Agent Enterprise Knowledge Assistant

A high-performance, modular **RAG-based Enterprise Knowledge Assistant** built with **Agentic AI** using **Python, FastAPI, LangGraph, LangChain, FAISS, Hugging Face, and SQLite**.

---

## 🌟 Key Highlights & Architecture

- **Multi-Agent Workflows with LangGraph**:
  - **Planner Agent**: Analyzes user intent, determines if internal retrieval is needed, and rewrites queries for semantic vector search.
  - **Retriever Agent**: Performs semantic similarity search against FAISS with score thresholding.
  - **Verifier / Hallucination Grader Agent**: Evaluates relevance of retrieved passages against the query, filtering noise and guarding against hallucinations.
  - **Synthesizer Agent**: Generates factual, grounded responses with inline numbered citations `[1]`, `[2]` linking back to source documents.
- **Document Ingestion & Semantic Search**:
  - Ingestion pipeline supporting `.pdf`, `.md`, and `.txt` files with `RecursiveCharacterTextSplitter`.
  - Embeddings powered by **Hugging Face** (`sentence-transformers/all-MiniLM-L6-v2`) with fallback vectorizer.
  - Vector indexing and disk persistence using **FAISS** (`faiss-cpu`).
- **REST APIs & Persistence**:
  - Built with **FastAPI** featuring interactive Swagger docs (`/docs`).
  - **SQLite** conversation memory storing chat sessions, messages, citations, and full step-by-step agent execution traces.
- **Modern Interactive Dashboard**:
  - Real-time Multi-Agent Step Execution Visualizer.
  - Interactive Citation Drawer and Knowledge Base document upload manager.

---

## 📐 Multi-Agent Workflow Diagram

```
                       User Query
                           │
                           ▼
                  ┌─────────────────┐
                  │  Planner Agent  │
                  └────────┬────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼ (Internal Search)         ▼ (Conversational)
    ┌─────────────────┐                  │
    │ Retriever Agent │ (FAISS + HF)     │
    └────────┬────────┘                  │
             │                           │
             ▼                           │
    ┌─────────────────┐                  │
    │ Verifier Agent  │ (Relevance /     │
    └────────┬────────┘  Anti-Hallucin.) │
             │                           │
             ▼                           │
    ┌─────────────────┐                  │
    │Synthesizer Agent│◄─────────────────┘
    └────────┬────────┘
             │
             ▼
Grounded Response + Inline Citations + Agent Trace
```

---

## 📁 Project Structure

```
saicharan/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application entrypoint & static mount
│   │   ├── config.py             # Pydantic environment configuration
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── session.py        # SQLite engine & session management
│   │   │   └── models.py         # SQLAlchemy models (Conversations, Messages, Docs)
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py     # Hugging Face embeddings model wrapper
│   │   │   ├── vectorstore.py    # FAISS Vector Store manager
│   │   │   └── ingest.py         # Document parsing, chunking & auto-seeder
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── state.py          # LangGraph AgentState TypedDict
│   │   │   ├── planner.py        # Planner & Query Router Agent
│   │   │   ├── retriever.py      # Semantic FAISS Retriever Agent
│   │   │   ├── verifier.py       # Hallucination / Relevance Grader Agent
│   │   │   ├── synthesizer.py    # Grounded Response & Citation Synthesizer
│   │   │   └── graph.py          # Compiled LangGraph StateGraph workflow
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py           # POST /api/chat
│   │   │   ├── documents.py      # Upload, list, delete documents
│   │   │   └── history.py        # Conversation sessions & memory
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── chat.py           # Pydantic schemas for chat & agent steps
│   │       └── document.py       # Pydantic schemas for documents & stats
├── frontend/
│   ├── index.html                # Modern enterprise AI dashboard
│   ├── styles.css                # Dark-mode styling system
│   └── app.js                    # Chat controller, agent visualizer, file upload
├── data/
│   ├── sample_docs/              # Seed enterprise documentation (.md)
│   │   ├── company_policies.md
│   │   ├── engineering_standards.md
│   │   └── cloud_security.md
│   └── vectorstore/              # FAISS index storage (.faiss, .pkl)
├── tests/
│   ├── __init__.py
│   ├── test_rag.py               # Unit tests for FAISS and embeddings
│   └── test_agents.py            # Unit tests for LangGraph multi-agent flow
├── .env.example
├── requirements.txt
├── run.py                        # Single-command application launcher
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`. If you want to use OpenAI LLMs, provide your API key (the assistant also includes a built-in deterministic grounding fallback for instant offline testing):
```env
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL_NAME=gpt-4o-mini
```

### 3. Launch the Application
```bash
python run.py
```

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 REST API Specifications

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Executes LangGraph multi-agent workflow, returns answer, citations, and agent execution trace. |
| `GET` | `/api/documents` | Lists indexed documents, chunk statistics, and total FAISS vectors. |
| `POST` | `/api/documents/upload` | Ingests `.pdf`, `.md`, or `.txt` file, chunks and indexes into FAISS. |
| `DELETE` | `/api/documents/{id}` | Deletes document and reindexes remaining chunks. |
| `POST` | `/api/documents/reindex`| Rebuilds FAISS index from stored chunks. |
| `GET` | `/api/conversations` | Lists user conversation threads. |
| `GET` | `/api/conversations/{id}`| Retrieves full conversation message history with citations. |
| `DELETE`| `/api/conversations/{id}`| Deletes conversation thread and messages. |
| `GET` | `/api/health` | System health check and model status. |

---

## 🧪 Running Tests

Run the test suite using `unittest` or `pytest`:
```bash
python -m unittest discover tests
```
