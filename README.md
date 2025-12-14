# Luma RAG Application

A production-ready RAG (Retrieval-Augmented Generation) application designed for hospital researchers. This application enables intelligent document search and question-answering using AI-powered semantic search and natural language processing.

![1](https://github.com/user-attachments/assets/1c6c3a71-d354-4bdf-a4ce-f3bd19a84ae0)

## 🚀 Features

### Core Capabilities

- **📄 Multi-Format Document Processing**
  - Supports PDF, DOCX, and PPTX files
  - Automatic metadata extraction (author, creation date, keywords)
  - Three chunking strategies: whole file, page-based, and token-based

- **🔍 Intelligent Search & Retrieval**
  - Vector-based semantic search using Weaviate
  - GreenPT embeddings (OpenAI-compatible)
  - Relevance scoring and ranking
  - Keyword-based document linking

- **💬 AI-Powered Chat Interface**
  - RAG-based question answering
  - Context-aware responses using retrieved documents
  - Conversation history support
  - Markdown rendering for formatted responses

 <img width="2193" height="926" alt="2" src="https://github.com/user-attachments/assets/74e2ed72-b8da-4c77-9d12-e248f873a1cb" />


- **📊 Knowledge Graph Visualization**
  - Interactive document network visualization
  - Document relationships based on keyword matching
  - Force-directed graph layout
  - Real-time graph generation

- **🌐 Web-Based UI**
  - Modern, responsive web interface
  - Real-time status monitoring
  - Document metadata display
  - Knowledge graph integration

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.13+)
- **Vector Database**: Weaviate 1.27.6
- **Embeddings**: GreenPT API (OpenAI-compatible)
- **Frontend**: HTML/CSS/JavaScript with ECharts
- **Document Processing**: PyPDF, python-docx, python-pptx
- **Containerization**: Docker & Docker Compose

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.13+ (for local development)
- ngrok (optional, for public access)

## 🏃 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LUMA
```

### 2. Configure Environment

Create a `.env.docker` file (copy from `.env.docker.example` if available):

```env
GREENPT_API_KEY=your_api_key_here
GREENPT_BASE_URL=https://api.greenpt.ai/v1
WEAVIATE_URL=http://weaviate:8080
DOCUMENTS_PATH=/app/Example-Files
```

### 3. Start Services

```bash
docker-compose up -d
```

This starts:
- **Weaviate** (vector store) on `http://localhost:8080`
- **Backend** (FastAPI) on `http://localhost:8000`

### 4. Access the Application

- **Main Application**: http://localhost:8000/webpages/find.html
- **Knowledge Graph**: http://localhost:8000/webpages/graph.html
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage Guide

### Document Processing

Documents in the `Example-Files` directory are automatically processed on startup. The system:

1. Extracts text and metadata from PDF, DOCX, and PPTX files
2. Creates chunks using three strategies:
   - **Whole file**: Entire document as a single chunk
   - **Page-based**: Each page as a separate chunk
   - **Token-based**: Chunks split by token count (max 1000 tokens)
3. Generates embeddings for all chunks
4. Stores everything in Weaviate with metadata

### Using the Chat Interface

1. Open http://localhost:8000/webpages/find.html
2. Enter your question in the search box
3. The system will:
   - Search for relevant documents using semantic similarity
   - Retrieve top 5 most relevant chunks
   - Generate an AI response based on retrieved context
   - Display retrieved documents with metadata

### Knowledge Graph

1. Click "📊 Knowledge Graph" in the header
2. View the interactive network visualization
3. Nodes represent documents, edges represent keyword-based relationships
4. Zoom, pan, and explore document connections

## 🔌 API Endpoints

### Status & Health

- `GET /health` - Health check endpoint
- `GET /api/status` - System status with document counts

### Chat & Search

- `POST /api/chat` - Chat with RAG functionality
  ```json
  {
    "message": "Your question here",
    "conversation_history": []
  }
  ```

### Documents

- `GET /api/documents` - List all processed documents
- `GET /api/graph` - Get knowledge graph data

### Static Files

- `GET /webpages/find.html` - Main application page
- `GET /webpages/graph.html` - Knowledge graph visualization
- `GET /` - Redirects to main application page

## 🌍 Public Access with Ngrok

To expose your local application to the internet:

### 1. Install Ngrok

See `NGROK_SETUP.md` for detailed installation instructions.

### 2. Configure Authtoken

```powershell
.\ngrok.exe config add-authtoken YOUR_AUTHTOKEN
```

Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken

### 3. Start Ngrok Tunnel

```powershell
.\start-ngrok.ps1
```

Or manually:
```powershell
.\ngrok.exe http 8000
```

### 4. Access via Public URL

Ngrok will provide a public URL like:
```
https://xxxx-xx-xx-xx-xx.ngrok-free.app
```

The application automatically detects ngrok URLs and adjusts API endpoints accordingly.

## 📁 Project Structure

```
LUMA/
├── backend/                 # FastAPI backend
│   ├── api/                # API routes
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   │   ├── chat_service.py
│   │   ├── document_processor.py
│   │   ├── embedding_service.py
│   │   ├── knowledge_graph_service.py
│   │   └── vector_store.py
│   └── main.py             # FastAPI app entry point
├── webpages/               # Frontend web pages
│   ├── find.html          # Main chat interface
│   ├── graph.html         # Knowledge graph visualization
│   └── style.css          # Styling
├── Example-Files/          # Documents to process
├── scripts/                # Utility scripts
├── docker-compose.yml      # Docker services configuration
├── Dockerfile.backend      # Backend container definition
└── pyproject.toml          # Python dependencies
```

## 🔧 Configuration

### Environment Variables

- `GREENPT_API_KEY`: Your GreenPT API key
- `GREENPT_BASE_URL`: GreenPT API base URL (default: https://api.greenpt.ai/v1)
- `WEAVIATE_URL`: Weaviate instance URL (default: http://weaviate:8080)
- `DOCUMENTS_PATH`: Path to documents directory (default: /app/Example-Files)

### Docker Compose Services

- **weaviate**: Vector database service
- **backend**: FastAPI application with static file serving

## 🧪 Development

### Local Development Setup

```bash
# Install dependencies
pip install -e .

# Start Weaviate
docker-compose up -d weaviate

# Run backend locally
uvicorn backend.main:app --reload --port 8000
```

### Testing

```bash
# Test document processing
python scripts/test_extraction.py

# Test knowledge graph
python scripts/test_graph.py

# Check vector store
python scripts/show_embeddings.py
```

## 📝 Key Features Explained

### Document Chunking Strategies

1. **Whole File**: Preserves document context, useful for short documents
2. **Page-based**: Maintains page boundaries, good for structured documents
3. **Token-based**: Ensures chunks fit within LLM context windows

### RAG Pipeline

1. **Query Embedding**: Convert user question to vector
2. **Vector Search**: Find similar document chunks in Weaviate
3. **Context Building**: Aggregate top results with metadata
4. **Response Generation**: Use LLM to generate answer from context

### Knowledge Graph

- Nodes represent documents
- Edges represent keyword-based relationships
- Opacity indicates document age (newer = more opaque)
- Categories group documents by author

## 🐛 Troubleshooting

### Backend Not Starting

- Check Docker logs: `docker-compose logs backend`
- Verify Weaviate is healthy: `docker-compose ps`
- Check environment variables in `.env.docker`

### No Documents Found

- Ensure documents are in `Example-Files/` directory
- Check backend logs for processing errors
- Verify document formats are supported (PDF, DOCX, PPTX)

### Graph Not Loading

- Open browser console (F12) to check for errors
- Verify API endpoint is accessible: `curl http://localhost:8000/api/graph`
- Check that documents have been processed

### Ngrok Issues

- Verify ngrok is running: Check http://127.0.0.1:4040
- Ensure authtoken is configured correctly
- Check firewall settings

## 📚 Additional Documentation

- `NGROK_SETUP.md` - Detailed ngrok setup guide
- `QUICK_START.md` - Quick reference for common tasks
- `USAGE_GUIDE.md` - Detailed usage instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

See `LICENSE` file for details.

## 🙏 Acknowledgments

- Weaviate for vector database capabilities
- GreenPT for embeddings API
- FastAPI for the excellent web framework
- ECharts for graph visualization
