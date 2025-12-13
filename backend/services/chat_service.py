"""Chat service for LLM completions using GreenPT API."""

from openai import AsyncOpenAI
from backend.config import settings


class ChatService:
    def __init__(self, embedding_service=None, vector_store=None):
        """Initialize chat service with OpenAI-compatible client."""
        self.client = AsyncOpenAI(
            api_key=settings.greenpt_api_key,
            base_url=settings.greenpt_base_url,
        )
        self.model = settings.greenpt_chat_model
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def generate_completion(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate a completion for the given prompt."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    async def close(self):
        """Close the OpenAI client connection."""
        if self.client:
            await self.client.close()
