# backend/main.py
"""FastAPI application for Luma RAG backend."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# Import router directly to avoid circular imports
from backend.api.routes import router as api_router
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.chat_service import ChatService
from backend.services.document_processor import DocumentProcessor
from backend.config import settings
from pathlib import Path
import asyncio
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    await vector_store.connect()
    
    chat_service = ChatService(embedding_service, vector_store)
    document_processor = DocumentProcessor(embedding_service, chat_service)
    
    # Store services in app state
    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.chat_service = chat_service
    app.state.document_processor = document_processor
    
    # Auto-process Example-Files on startup (run in background to avoid blocking startup)
    print("🚀 Starting auto-processing of Example-Files in background...")
    
    # Create background task for auto-processing
    async def startup_processing():
        try:
            await process_all_documents(app.state)
            print("✅ Auto-processing completed!")
        except Exception as e:
            print(f"❌ Auto-processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run in background (non-blocking)
    asyncio.create_task(startup_processing())
    
    yield
    
    # Cleanup on shutdown
    await vector_store.disconnect()
    await embedding_service.close()
    await chat_service.close()


async def process_all_documents(app_state):
    """Process all documents in Example-Files directory."""
    document_processor = app_state.document_processor
    vector_store = app_state.vector_store
    embedding_service = app_state.embedding_service
    
    documents_path = Path(settings.documents_path)
    
    # Find all supported files
    supported_extensions = [".pdf", ".docx", ".pptx"]
    files = [
        f for f in documents_path.iterdir()
        if f.suffix.lower() in supported_extensions
    ]
    
    print(f"📁 Found {len(files)} documents to process")
    
    for file_path in files:
        try:
            print(f"📄 Processing: {file_path.name}")
            processed_doc = await document_processor.process_document(file_path)
            
            if processed_doc:
                # Collect ALL chunks (all 3 strategies)
                all_chunks = (
                    processed_doc.whole_file_chunks +
                    processed_doc.page_chunks +
                    processed_doc.token_chunks
                )
                
                # Generate embeddings
                chunk_texts = [chunk.content for chunk in all_chunks]
                embeddings = await embedding_service.embed_texts(chunk_texts)
                
                # Store in Weaviate
                await vector_store.add_chunks(all_chunks, embeddings)
                print(f"✅ Processed {file_path.name}: {len(all_chunks)} chunks")
            else:
                print(f"❌ Failed to process {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    # Get final count
    total_chunks = await vector_store.get_document_count()
    print(f"📊 Total chunks in vector store: {total_chunks}")


# Create FastAPI app
app = FastAPI(
    title="Luma RAG API",
    description="Backend API for Luma Research Assistant",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint - redirect to main page
@app.get("/")
async def root():
    """Root endpoint - redirects to main application page."""
    return RedirectResponse(url="/webpages/find.html")

# Include API router
app.include_router(api_router, prefix="/api")

# Mount static files for webpages
webpages_path = Path(__file__).parent.parent / "webpages"
if webpages_path.exists():
    app.mount("/webpages", StaticFiles(directory=str(webpages_path)), name="webpages")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)