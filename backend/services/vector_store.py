# backend/services/vector_store.py
"""Vector store service for Weaviate operations."""

from typing import List
from uuid import UUID
import weaviate
import weaviate.classes as wvc
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from backend.models.document import DocumentChunk
from backend.models.schemas import RetrievedDocument
from backend.config import settings


class VectorStore:
    def __init__(self):
        """Initialize vector store with Weaviate client."""
        self.client = None
        self.collection_name = "DocumentChunk"

    async def connect(self):
        """Connect to Weaviate instance."""
        # Parse URL to extract host and port
        url = settings.weaviate_url.replace("http://", "").replace("https://", "")
        if ":" in url:
            host, port = url.split(":")
            port = int(port)
        else:
            host = url
            port = 80
        
        # Parse GRPC URL for gRPC connection
        grpc_url = settings.weaviate_grpc_url
        if ":" in grpc_url:
            grpc_host, grpc_port = grpc_url.split(":")
            grpc_port = int(grpc_port)
        else:
            grpc_host = grpc_url
            grpc_port = 50051
        
        self.client = weaviate.WeaviateClient(
            connection_params=ConnectionParams(
                http=ProtocolParams(
                    host=host,
                    port=port,
                    secure=False
                ),
                grpc=ProtocolParams(
                    host=grpc_host,
                    port=grpc_port,
                    secure=False
                )
            )
        )
        
        # Connect the client
        self.client.connect()
        return self.client

    async def disconnect(self):
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()

    async def create_schema(self):
        """Create Weaviate schema for DocumentChunk collection."""
        if self.client is None:
            await self.connect()

        # Delete existing collection if it exists
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        # Create new collection
        collection = self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.UUID),
                Property(name="document_id", data_type=DataType.UUID),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="chunking_strategy", data_type=DataType.TEXT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="author", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
                Property(name="file_type", data_type=DataType.TEXT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            ],
        )
        return collection

    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add document chunks with embeddings to Weaviate."""
        if self.client is None:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        # Prepare DataObject instances with properties and vectors
        data_objects = []
        for chunk, embedding in zip(chunks, embeddings):
            properties = {
                "chunk_id": str(chunk.chunk_id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunking_strategy": chunk.chunking_strategy.value,
                "token_count": chunk.token_count,
                "title": chunk.title,
                "author": chunk.author,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                "file_type": chunk.file_type.value,
                "file_path": chunk.file_path,
                "summary": chunk.summary,
                "keywords": chunk.keywords,
            }

            # Create DataObject with properties and vector
            data_objects.append(
                wvc.data.DataObject(
                    properties=properties,
                    vector=embedding
                )
            )

        # Insert DataObjects
        collection.data.insert_many(data_objects)

    async def search(self, query_embedding: List[float], limit: int = 5) -> List[RetrievedDocument]:
        """Search for similar documents using vector search."""
        if self.client is None:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )

        retrieved_docs = []
        for item in response.objects:
            props = item.properties
            # Handle chunk_id - it might be a UUID object or string
            chunk_id_value = props["chunk_id"]
            if isinstance(chunk_id_value, str):
                chunk_id = UUID(chunk_id_value)
            else:
                chunk_id = chunk_id_value
            
            retrieved_docs.append(RetrievedDocument(
                chunk_id=chunk_id,
                title=props["title"],
                summary=props["summary"],
                keywords=props["keywords"],
                author=props["author"],
                created_at=props["created_at"],
                file_type=props["file_type"],
                file_path=props["file_path"],
                content=props["content"],
                relevance_score=1 - item.metadata.distance,
                chunking_strategy=props["chunking_strategy"],
            ))

        return retrieved_docs

    async def get_document_count(self) -> int:
        """Get total number of chunks in the vector store."""
        if self.client is None:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count

    async def clear_all(self):
        """Clear all data from the vector store."""
        if self.client is None:
            await self.connect()

        if self.client.collections.exists(self.collection_name):
            # Delete and recreate the collection (fastest way to clear all data)
            self.client.collections.delete(self.collection_name)
            await self.create_schema()