"""Chat service for LLM completions using GreenPT API."""

from typing import List
from openai import AsyncOpenAI
from backend.config import settings
from backend.models.schemas import ChatRequest, ChatResponse, RetrievedDocument


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

    async def chat_with_rag(self, chat_request: ChatRequest) -> ChatResponse:
        """Complete RAG pipeline: embed query, search Weaviate, build context, generate answer."""
        # Step 1: Embed query
        query_embedding = await self.embedding_service.embed_query(chat_request.message)

        # Step 2: Retrieve documents from Weaviate
        retrieved_docs = await self.vector_store.search(query_embedding, limit=5)

        # Step 3: Build context from retrieved documents
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {idx}: {doc.title}]\n"
                f"Summary: {doc.summary}\n"
                f"Content: {doc.content[:1000]}...\n"
                f"Keywords: {', '.join(doc.keywords)}\n"
            )
        context = "\n\n".join(context_parts)

        # Step 4: Generate response using RAG
        system_prompt = """You are an AI assistant for hospital researchers.
Answer questions based ONLY on the provided documents.
Cite which documents you reference."""

        messages = [
            {"role": "system", "content": system_prompt},
            *chat_request.conversation_history[-10:],  # Keep last 10 messages
            {
                "role": "user", 
                "content": f"Question: {chat_request.message}\n\nDocuments:\n{context}"
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        return ChatResponse(
            message=response.choices[0].message.content,
            retrieved_documents=retrieved_docs,
        )

    async def close(self):
        """Close the OpenAI client connection."""
        if self.client:
            await self.client.close()
