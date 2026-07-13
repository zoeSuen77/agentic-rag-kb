"""Dense embedding models for indexing and retrieval.

Production indexing uses sentence-transformers. Tests can inject the deterministic
embedding model to avoid model downloads and keep assertions stable.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:#-]+|[\u4e00-\u9fff]")


class EmbeddingModel(Protocol):
    """Embedding model interface used by the Qdrant indexer."""

    @property
    def vector_size(self) -> int:
        """Dense vector dimension."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document strings."""


class SentenceTransformerEmbeddingModel:
    """sentence-transformers embedding model wrapper."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install sentence-transformers to build dense embeddings.") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._vector_size = int(self.model.get_sentence_embedding_dimension())

    @property
    def vector_size(self) -> int:
        """Dense vector dimension."""

        return self._vector_size

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""

        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document strings."""

        return self.model.encode(texts, normalize_embeddings=True).tolist()


class DeterministicEmbeddingModel:
    """Deterministic hashing embedding for tests."""

    def __init__(self, vector_size: int = 64) -> None:
        self._vector_size = vector_size

    @property
    def vector_size(self) -> int:
        """Dense vector dimension."""

        return self._vector_size

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string."""

        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document strings."""

        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._vector_size
        for token in TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self._vector_size
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
