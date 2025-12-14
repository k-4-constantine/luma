# backend/models/document.py
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"

class ChunkingStrategy(str, Enum):
    WHOLE_FILE = "whole_file"
    PAGES = "pages"
    MAX_TOKENS = "max_tokens"

class DocumentMetadata(BaseModel):
    title: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    file_type: FileType
    file_path: str
    file_size: int

class DocumentChunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int
    chunking_strategy: ChunkingStrategy
    token_count: int
    title: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    file_type: FileType
    file_path: str
    summary: str
    keywords: List[str]

class ProcessedDocument(BaseModel):
    document_id: UUID
    metadata: DocumentMetadata
    full_content: str
    whole_file_chunks: List[DocumentChunk]
    page_chunks: List[DocumentChunk]
    token_chunks: List[DocumentChunk]
    summary: str
    keywords: List[str]