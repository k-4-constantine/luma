# Luma RAG Application

RAG (Retrieval-Augmented Generation) application for hospital researchers with document processing (PDF, DOCX, PPTX), vector storage (Weaviate), and chat interface.

## Running with Docker Compose

```bash
docker compose up --build
```

This starts:
- **Weaviate** (vector store) on `http://localhost:8080`
- **Backend** (FastAPI) on `http://localhost:8000`
- **Frontend** (Streamlit) on `http://localhost:8501`

### Access

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs
- Weaviate: http://localhost:8080

### Stop

```bash
docker compose down
```

### Reset Database

```bash
docker compose down -v
docker compose up --build
```