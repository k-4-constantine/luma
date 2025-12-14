# backend/api/routes.py
"""API routes for Luma RAG backend."""

from fastapi import APIRouter, Request, HTTPException
from backend.models.schemas import ChatRequest, ChatResponse, StatusResponse
from backend.services.vector_store import VectorStore
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