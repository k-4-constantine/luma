"""Embedding service for generating text embeddings using GreenPT API."""

from typing import List
from openai import AsyncOpenAI
from backend.config import settings


class EmbeddingService:
    def __init__(self):
        """Initialize embedding service with OpenAI-compatible client."""
        self.client = AsyncOpenAI(
            api_key=settings.greenpt_api_key,
            base_url=settings.greenpt_base_url,
        )
        self.model = settings.greenpt_embedding_model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
