# LUMA RAG Application - Architecture Documentation

## Table of Contents
1. [High-Level Architecture Overview](#high-level-architecture-overview)
2. [Component Interaction Diagram](#component-interaction-diagram)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Technology Stack](#technology-stack)

---

## High-Level Architecture Overview

The LUMA (Shoulders of Giants) RAG application follows a modern microservices architecture with clear separation between frontend, backend, vector database, and external AI services.

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI1[Chat Interface<br/>find.html]
        UI2[Knowledge Graph<br/>graph.html]
    end

    subgraph "API Layer"
        API[FastAPI Backend<br/>Port 8000]
        Routes[API Routes<br/>/api/chat, /api/status, /api/graph, /api/transcribe]
    end

    subgraph "Service Layer"
        CS[Chat Service<br/>RAG Pipeline]
        DP[Document Processor<br/>Extract & Chunk]
        ES[Embedding Service<br/>Text → Vectors]
        VS[Vector Store<br/>Weaviate Client]
        KG[Knowledge Graph Service<br/>Network Generation]
        TS[Transcription Service<br/>Audio → Text]
    end

    subgraph "Data Layer"
        WV[(Weaviate<br/>Vector Database<br/>Port 8080)]
        FS[File Storage<br/>Example-Files/]
    end

    subgraph "External Services"
        GPT[GreenPT API<br/>OpenAI-Compatible]
        EMB[Embedding Model<br/>text-embedding-3-small]
        LLM[Chat Model<br/>gpt-4o-mini]
        DG[Deepgram<br/>Transcription]
    end

    UI1 --> API
    UI2 --> API
    API --> Routes
    Routes --> CS
    Routes --> DP
    Routes --> KG
    Routes --> TS

    CS --> ES
    CS --> VS
    CS --> LLM

    DP --> ES
    DP --> CS
    DP --> FS

    ES --> EMB
    VS --> WV
    KG --> VS

    TS --> DG
    TS --> FS

    EMB --> GPT
    LLM --> GPT
    DG --> GPT

    style UI1 fill:#e3f2fd
    style UI2 fill:#e3f2fd
    style API fill:#fff3e0
    style CS fill:#f3e5f5
    style DP fill:#f3e5f5
    style ES fill:#f3e5f5
    style VS fill:#f3e5f5
    style KG fill:#f3e5f5
    style TS fill:#f3e5f5
    style WV fill:#e8f5e9
    style FS fill:#e8f5e9
    style GPT fill:#ffebee
    style EMB fill:#ffebee
    style LLM fill:#ffebee
    style DG fill:#ffebee
```

### Key Components

#### Frontend Layer
- **Chat Interface (find.html)**: Main user interface for document search and Q&A
- **Knowledge Graph (graph.html)**: Interactive visualization of document relationships

#### API Layer
- **FastAPI Backend**: Async Python web framework handling all HTTP requests
- **API Routes**: RESTful endpoints for chat, status, graph, and transcription

#### Service Layer
- **Chat Service**: Implements RAG pipeline (retrieve + generate)
- **Document Processor**: Extracts and chunks documents with 3 strategies
- **Embedding Service**: Converts text to vector embeddings
- **Vector Store**: Manages Weaviate database operations
- **Knowledge Graph Service**: Generates document network visualization
- **Transcription Service**: Converts audio to text

#### Data Layer
- **Weaviate**: Vector database storing document chunks with embeddings
- **File Storage**: Physical storage for uploaded documents

#### External Services
- **GreenPT API**: OpenAI-compatible API gateway
- **Embedding Model**: text-embedding-3-small for vector generation
- **Chat Model**: gpt-4o-mini for response generation
- **Deepgram**: Audio transcription service

---

## Component Interaction Diagram

This diagram shows the sequence of interactions for the main user flows.

### RAG Query Flow (Chat)

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Chat UI
    participant API as FastAPI
    participant ChatService
    participant EmbedService
    participant VectorStore
    participant Weaviate
    participant LLM as GreenPT LLM

    User->>Frontend: Enter question
    Frontend->>API: POST /api/chat
    API->>ChatService: chat_with_rag(request)

    ChatService->>EmbedService: embed_query(question)
    EmbedService->>LLM: Generate embedding
    LLM-->>EmbedService: Vector[1536]
    EmbedService-->>ChatService: query_embedding

    ChatService->>VectorStore: search(embedding, limit=5)
    VectorStore->>Weaviate: Vector similarity search
    Weaviate-->>VectorStore: Top 5 chunks
    VectorStore-->>ChatService: retrieved_documents

    ChatService->>ChatService: Build context from docs
    ChatService->>LLM: Generate response with context
    LLM-->>ChatService: AI response

    ChatService-->>API: ChatResponse
    API-->>Frontend: JSON response
    Frontend-->>User: Display answer + sources
```

### Document Processing Flow

```mermaid
sequenceDiagram
    participant System as Startup/Upload
    participant DocProcessor
    participant Extractor as Text Extractor
    participant ChatService
    participant EmbedService
    participant VectorStore
    participant Weaviate

    System->>DocProcessor: process_document(file_path)

    DocProcessor->>Extractor: Extract text by type
    Extractor-->>DocProcessor: full_text, page_texts, metadata

    DocProcessor->>ChatService: generate_summary(content)
    ChatService-->>DocProcessor: Summary (2-3 sentences)

    DocProcessor->>ChatService: extract_keywords(content)
    ChatService-->>DocProcessor: Keywords (5-7 terms)

    DocProcessor->>DocProcessor: chunk_whole_file()
    DocProcessor->>DocProcessor: chunk_by_pages()
    DocProcessor->>DocProcessor: chunk_by_max_tokens(512)

    DocProcessor-->>System: ProcessedDocument (all chunks)

    System->>EmbedService: embed_texts(all_chunks)
    EmbedService-->>System: embeddings[]

    System->>VectorStore: add_chunks(chunks, embeddings)
    VectorStore->>Weaviate: Batch insert
    Weaviate-->>VectorStore: Success
    VectorStore-->>System: Stored
```

### Audio Transcription Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Chat UI
    participant API as FastAPI
    participant TranscriptService
    participant Deepgram
    participant FS as File Storage
    participant DocProcessor
    participant VectorStore

    User->>Frontend: Upload audio file
    Frontend->>API: POST /api/transcribe

    API->>TranscriptService: transcribe_audio(bytes)
    TranscriptService->>Deepgram: Transcribe request
    Deepgram-->>TranscriptService: Transcript text

    TranscriptService->>FS: Save as .txt file
    FS-->>TranscriptService: File path

    TranscriptService-->>API: transcript, file_path

    API->>DocProcessor: process_document(txt_path)
    DocProcessor-->>API: ProcessedDocument

    API->>VectorStore: Store chunks + embeddings
    VectorStore-->>API: Success

    API-->>Frontend: Transcript + chunk count
    Frontend-->>User: Display transcript
```

### Knowledge Graph Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Graph UI
    participant API as FastAPI
    participant KGService as KnowledgeGraphService
    participant VectorStore
    participant Weaviate

    User->>Frontend: Open graph.html
    Frontend->>API: GET /api/graph

    API->>KGService: generate_graph()

    KGService->>VectorStore: Get all documents
    VectorStore->>Weaviate: Fetch all chunks
    Weaviate-->>VectorStore: All chunks
    VectorStore-->>KGService: Document chunks

    KGService->>KGService: Group by file_path
    KGService->>KGService: Extract unique documents
    KGService->>KGService: Calculate keyword matches
    KGService->>KGService: Build nodes (documents)
    KGService->>KGService: Build edges (relationships)
    KGService->>KGService: Create categories (authors)

    KGService-->>API: Graph data (nodes, links, categories)
    API-->>Frontend: JSON graph data
    Frontend->>Frontend: Render ECharts visualization
    Frontend-->>User: Interactive graph
```

---

## Data Flow Diagram

This diagram illustrates how data flows through the system from input to output.

```mermaid
graph LR
    subgraph "Input Sources"
        DOC[Documents<br/>PDF, DOCX, PPTX, TXT]
        AUDIO[Audio Files<br/>MP3, MP4, WAV, M4A]
        QUERY[User Queries<br/>Natural Language]
    end

    subgraph "Processing Pipeline"
        EXTRACT[Text Extraction<br/>pypdf, python-docx, pptx]
        TRANSCRIBE[Audio Transcription<br/>Deepgram API]
        CHUNK[Chunking<br/>3 Strategies:<br/>Whole/Page/Token]
        SUMMARIZE[Summarization<br/>LLM Generation]
        KEYWORD[Keyword Extraction<br/>LLM Analysis]
        EMBED[Embedding Generation<br/>text-embedding-3-small]
    end

    subgraph "Storage & Retrieval"
        VECTOR[(Vector Database<br/>Weaviate<br/>DocumentChunk Collection)]
        FILES[File System<br/>Example-Files/]
    end

    subgraph "Query Processing"
        QEMBED[Query Embedding<br/>Convert to vector]
        SEARCH[Vector Search<br/>Similarity matching]
        CONTEXT[Context Building<br/>Top 5 chunks]
        GENERATE[Response Generation<br/>LLM with context]
    end

    subgraph "Output & Visualization"
        ANSWER[Chat Response<br/>AI-generated answer]
        SOURCES[Source Documents<br/>Retrieved chunks]
        GRAPH[Knowledge Graph<br/>Document network]
    end

    DOC --> EXTRACT
    AUDIO --> TRANSCRIBE

    EXTRACT --> SUMMARIZE
    EXTRACT --> KEYWORD
    EXTRACT --> CHUNK
    TRANSCRIBE --> CHUNK

    SUMMARIZE --> METADATA[Metadata]
    KEYWORD --> METADATA
    METADATA --> EMBED
    CHUNK --> EMBED

    EMBED --> VECTOR
    CHUNK --> FILES

    QUERY --> QEMBED
    QEMBED --> SEARCH
    SEARCH --> VECTOR
    VECTOR --> CONTEXT
    CONTEXT --> GENERATE

    GENERATE --> ANSWER
    CONTEXT --> SOURCES
    VECTOR --> GRAPH

    ANSWER --> USER[User]
    SOURCES --> USER
    GRAPH --> USER

    style DOC fill:#e3f2fd
    style AUDIO fill:#e3f2fd
    style QUERY fill:#e3f2fd
    style EXTRACT fill:#fff3e0
    style TRANSCRIBE fill:#fff3e0
    style CHUNK fill:#fff3e0
    style SUMMARIZE fill:#fff3e0
    style KEYWORD fill:#fff3e0
    style EMBED fill:#fff3e0
    style VECTOR fill:#e8f5e9
    style FILES fill:#e8f5e9
    style QEMBED fill:#f3e5f5
    style SEARCH fill:#f3e5f5
    style CONTEXT fill:#f3e5f5
    style GENERATE fill:#f3e5f5
    style ANSWER fill:#ffebee
    style SOURCES fill:#ffebee
    style GRAPH fill:#ffebee
    style USER fill:#ffebee
```

### Data Flow Details

#### Document Ingestion Flow
```
Documents → Text Extraction → Chunking (3 strategies) → Metadata Generation (Summary + Keywords)
→ Embedding Generation → Storage in Weaviate + File System
```

**Chunking Strategies:**
1. **Whole File**: Entire document as single chunk
2. **Page-based**: Each page/slide as separate chunk
3. **Token-based**: 512-token chunks with 25% overlap

**Metadata Generation:**
- Summary: 2-3 sentence overview using LLM
- Keywords: 5-7 key terms extracted by LLM
- Author, created date, file type extracted from document

#### Query Processing Flow
```
User Query → Query Embedding → Vector Similarity Search → Retrieve Top 5 Chunks
→ Build Context → LLM Generation with Context → Response + Sources
```

**RAG Pipeline Steps:**
1. Embed user query into vector space
2. Search Weaviate for similar document chunks
3. Retrieve top 5 most relevant chunks
4. Build context from retrieved chunks
5. Generate response using LLM with context
6. Return answer + source documents

#### Audio Transcription Flow
```
Audio Upload → Deepgram Transcription → Save as TXT → Process as Document
→ Chunk & Embed → Store in Vector Database
```

#### Knowledge Graph Flow
```
All Document Chunks → Group by File → Extract Metadata → Calculate Keyword Matches
→ Build Nodes (Documents) → Build Edges (Relationships) → Render ECharts Graph
```

**Graph Elements:**
- **Nodes**: Documents (sized by connection count)
- **Edges**: Keyword-based relationships (weighted by match count)
- **Categories**: Author groups
- **Opacity**: Time-based (newer docs more visible)

---

## Technology Stack

### Frontend
- **HTML5/CSS3/JavaScript**: Core web technologies
- **marked.js**: Markdown rendering for chat responses
- **Apache ECharts 5.4.3**: Interactive graph visualization
- **jQuery 3.6.0**: AJAX requests

### Backend
- **FastAPI 0.115.0+**: Async Python web framework
- **Python 3.13+**: Core runtime
- **uvicorn**: ASGI server
- **pydantic-settings**: Configuration management

### AI/ML Services
- **GreenPT API**: OpenAI-compatible AI gateway
  - **Embeddings**: text-embedding-3-small (1536 dimensions)
  - **Chat**: gpt-4o-mini
  - **Transcription**: Deepgram (green-s model)
- **tiktoken 0.8.0+**: Token counting (cl100k_base)

### Vector Database
- **Weaviate 1.27.6**: Vector search engine
  - HTTP endpoint: Port 8080
  - gRPC endpoint: Port 50051
  - Collection: DocumentChunk

### Document Processing
- **pypdf 5.1.0+**: PDF text extraction
- **python-docx 1.1.2+**: DOCX processing
- **python-pptx 1.0.2+**: PowerPoint processing

### Infrastructure
- **Docker + Docker Compose**: Containerization
- **ngrok** (optional): Public access tunneling

### Development
- **uv**: Python package installer
- **Custom scripts**: Testing and database management

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Environment"
        subgraph "Backend Container"
            FASTAPI[FastAPI App<br/>Port 8000]
            SERVICES[All Services]
        end

        subgraph "Weaviate Container"
            WEAVIATE[Weaviate DB<br/>Port 8080/50051]
            PERSIST[(Persistent Volume<br/>weaviate_data)]
        end

        FASTAPI <--> WEAVIATE
        WEAVIATE --> PERSIST
    end

    subgraph "External"
        NGROK[ngrok Tunnel<br/>Optional]
        GREENPT[GreenPT API<br/>api.greenpt.ai]
    end

    NGROK --> FASTAPI
    SERVICES --> GREENPT

    subgraph "Client"
        BROWSER[Web Browser]
    end

    BROWSER --> NGROK
    BROWSER --> FASTAPI

    style FASTAPI fill:#fff3e0
    style WEAVIATE fill:#e8f5e9
    style PERSIST fill:#e8f5e9
    style GREENPT fill:#ffebee
    style BROWSER fill:#e3f2fd
```

### Container Configuration

**Backend Service:**
- Base: Python 3.13
- Exposed Port: 8000
- Volumes: ./Example-Files mounted
- Depends on: Weaviate (healthy)

**Weaviate Service:**
- Image: cr.weaviate.io/semitechnologies/weaviate:1.27.6
- Exposed Ports: 8080 (HTTP), 50051 (gRPC)
- Volumes: weaviate_data (persistent)
- Health Check: Every 5 seconds

**Network:**
- Custom bridge network: luma-network
- Internal service communication
- External access via localhost:8000 or ngrok

---

## Key Design Decisions

### 1. Multiple Chunking Strategies
The system implements three different chunking strategies to optimize retrieval:
- **Whole File**: Best for short documents or when full context is needed
- **Page-based**: Preserves page structure, good for presentations and reports
- **Token-based**: Optimal for long documents, ensures consistent chunk sizes

### 2. RAG Pipeline
Uses Retrieval-Augmented Generation to ground AI responses in actual documents:
- Reduces hallucinations
- Provides source attribution
- Enables knowledge base updates without retraining

### 3. Async Architecture
FastAPI's async capabilities enable:
- High concurrency
- Non-blocking I/O operations
- Efficient resource utilization

### 4. Vector Database (Weaviate)
Chosen for:
- Native vector similarity search
- Hybrid search capabilities
- Scalable architecture
- Rich metadata support

### 5. Knowledge Graph
Provides document discovery through:
- Keyword-based relationships
- Author clustering
- Time-based relevance
- Interactive exploration

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to main chat interface |
| `/health` | GET | Health check endpoint |
| `/api/status` | GET | System status and document counts |
| `/api/chat` | POST | RAG-powered Q&A endpoint |
| `/api/documents` | GET | List processed documents |
| `/api/graph` | GET | Knowledge graph data |
| `/api/transcribe` | POST | Audio transcription and processing |
| `/webpages/*` | GET | Static file serving (HTML/CSS/JS) |
| `/Example-Files/*` | GET | Document file serving |

---

## Performance Considerations

1. **Embedding Caching**: Embeddings are stored in Weaviate to avoid recomputation
2. **Batch Processing**: Documents are processed asynchronously on startup
3. **Connection Pooling**: Persistent connections to Weaviate and GreenPT API
4. **Lazy Loading**: Services initialized only once on startup
5. **Vector Indexing**: Weaviate maintains HNSW index for fast similarity search

---

## Security Notes

1. **CORS**: Configured to allow all origins (development mode)
2. **API Keys**: GreenPT API key stored in environment variables
3. **File Upload**: Validated file types for transcription
4. **Anonymous Access**: Weaviate configured for anonymous access (development mode)

**Production Recommendations:**
- Restrict CORS origins
- Implement authentication/authorization
- Secure Weaviate with API keys
- Add rate limiting
- Use HTTPS/TLS encryption
- Implement input validation and sanitization
