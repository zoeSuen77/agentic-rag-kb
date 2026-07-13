"""Configuration layer for the Agentic RAG system.

The config module centralizes environment variables, model names, Qdrant settings,
retrieval parameters, graph limits, and UI options so other modules do not read
environment state directly.
"""

from agentic_rag_kb.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]

