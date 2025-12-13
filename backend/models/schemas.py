# backend/models/schemas.py
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class RetrievedDocument(BaseModel):
    chunk_id: UUID
    title: str
    summary: str
    keywords: List[str]
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    file_type: str
    file_path: str
    content: str
    relevance_score: float
    chunking_strategy: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[dict] = []

class ChatResponse(BaseModel):
    message: str
    retrieved_documents: List[RetrievedDocument] = []

class StatusResponse(BaseModel):
    status: str
    processed_documents: int
    total_chunks: int
    weaviate_connected: bool