# backend/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    greenpt_api_key: str
    greenpt_base_url: str
    greenpt_embedding_model: str = "text-embedding-3-small"
    greenpt_chat_model: str = "gpt-4o-mini"

    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_url: str = "localhost:50051"

    documents_path: Path = Path("/home/pjotterb/repos/luma/Example-Files")
    max_tokens_chunk_size: int = 512

    model_config = {"env_file": ".env"}

settings = Settings()