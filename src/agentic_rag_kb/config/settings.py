"""Application settings for Agentic RAG.

TODO:
- Add validation for production and local environments.
- Split nested settings into model, qdrant, retrieval, graph, and evaluation groups.
- Load `.env` automatically in local development.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration used across the Agentic RAG application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_CHAT_MODEL")
    ollama_embed_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBED_MODEL")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="agentic_rag_chunks", alias="QDRANT_COLLECTION")
    qdrant_collection_child: str = Field(default="kb_child_chunks", alias="QDRANT_COLLECTION_CHILD")
    qdrant_collection_parent: str = Field(default="kb_parent_chunks", alias="QDRANT_COLLECTION_PARENT")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
