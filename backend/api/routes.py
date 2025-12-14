# backend/api/routes.py
"""API routes for Luma RAG backend."""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from backend.models.schemas import ChatRequest, ChatResponse, StatusResponse
from backend.services.vector_store import VectorStore
from backend.services.knowledge_graph_service import KnowledgeGraphService
from typing import List


router = APIRouter(tags=["api"])


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Get system status and document count."""
    vector_store = request.app.state.vector_store
    
    try:
        # Check if Weaviate is connected
        weaviate_connected = vector_store.client is not None
        
        # Get document count
        total_chunks = await vector_store.get_document_count() if weaviate_connected else 0
        
        # Count processed documents (approximate by counting unique document IDs)
        # For now, we'll use total_chunks as a proxy
        processed_documents = total_chunks // 3 if total_chunks > 0 else 0  # Average 3 chunks per document
        
        return StatusResponse(
            status="healthy",
            processed_documents=processed_documents,
            total_chunks=total_chunks,
            weaviate_connected=weaviate_connected
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {e}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """Chat endpoint with RAG functionality."""
    chat_service = request.app.state.chat_service
    
    try:
        response = await chat_service.chat_with_rag(chat_request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {e}")


@router.get("/documents")
async def list_documents(request: Request):
    """List all processed documents (for debugging)."""
    vector_store = request.app.state.vector_store
    
    try:
        if not vector_store.client:
            await vector_store.connect()
            
        collection = vector_store.client.collections.get(vector_store.collection_name)
        
        # Get all documents (limit to 100 for performance)
        response = collection.query.fetch_objects(limit=100)
        
        documents = []
        for item in response.objects:
            props = item.properties
            documents.append({
                "title": props["title"],
                "author": props["author"],
                "file_type": props["file_type"],
                "file_path": props["file_path"],
                "chunking_strategy": props["chunking_strategy"],
                "token_count": props["token_count"]
            })
        
        return {"documents": documents, "count": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {e}")


@router.get("/graph")
async def get_knowledge_graph(request: Request):
    """Get knowledge graph data for document network visualization."""
    vector_store = request.app.state.vector_store
    
    try:
        graph_service = KnowledgeGraphService(vector_store)
        graph_data = await graph_service.generate_graph()
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating knowledge graph: {str(e)}")


@router.post("/graph/filtered")
async def get_filtered_knowledge_graph(request: Request, file_paths: List[str]):
    """Get knowledge graph data only for specified file paths and their connections."""
    vector_store = request.app.state.vector_store
    
    try:
        graph_service = KnowledgeGraphService(vector_store)
        graph_data = await graph_service.generate_filtered_graph(file_paths)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating filtered knowledge graph: {str(e)}")


@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    """Transcribe audio file and save as text."""
    transcription_service = request.app.state.transcription_service
    document_processor = request.app.state.document_processor
    vector_store = request.app.state.vector_store
    embedding_service = request.app.state.embedding_service
    
    # Validate file type
    valid_extensions = [".mp3", ".mp4", ".wav", ".m4a"]
    if not file.filename.lower().endswith(tuple(valid_extensions)):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: .mp3, .mp4, .wav, .m4a")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Transcribe audio
        transcript = await transcription_service.transcribe_audio(audio_bytes)
        
        # Save transcript
        txt_path = await transcription_service.save_transcript(transcript, file.filename)
        
        # Auto-process into RAG (reuse existing pattern from main.py)
        processed_doc = await document_processor.process_document(txt_path)
        
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
            chunks_added = len(all_chunks)
        else:
            chunks_added = 0
        
        return {
            "filename": file.filename,
            "transcript": transcript,
            "txt_file": txt_path.name,
            "chunks_added": chunks_added
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")