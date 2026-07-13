"""Qdrant storage adapter.

This module isolates Qdrant-specific operations from the indexing pipeline. The
production adapter uses `qdrant-client`, while tests can use `InMemoryVectorStore`
with the same minimal interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


DEFAULT_COLLECTION_NAME = "agentic_rag_chunks"


@dataclass(slots=True)
class VectorPoint:
    """Vector point sent to Qdrant."""

    point_id: str
    vector: list[float]
    payload: dict


@dataclass(slots=True)
class VectorSearchResult:
    """Search result returned by the vector store."""

    point_id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    """Minimal vector store interface used by the indexer."""

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Delete and recreate a collection."""

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Upsert points into a collection."""

    def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search by dense vector."""


class QdrantStore:
    """Qdrant-backed vector store implementation."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None) -> None:
        self.url = url
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Create the qdrant-client lazily."""

        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Install qdrant-client to use QdrantStore.") from exc
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Delete old collection if present, then create a fresh dense-vector collection."""

        from qdrant_client import models

        try:
            self.client.delete_collection(collection_name=collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Upsert vector points into Qdrant."""

        from qdrant_client import models

        qdrant_points = [
            models.PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
            for point in points
        ]
        self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search Qdrant by dense vector."""

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except AttributeError:  # pragma: no cover - newer qdrant-client API
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            results = response.points
        return [
            VectorSearchResult(point_id=str(item.id), score=float(item.score), payload=dict(item.payload or {}))
            for item in results
        ]


class InMemoryVectorStore:
    """Small vector store used by tests and local smoke checks."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, VectorPoint]] = {}

    def recreate_collection(self, collection_name: str, vector_size: int) -> None:
        """Create an empty in-memory collection."""

        self.collections[collection_name] = {}

    def upsert(self, collection_name: str, points: list[VectorPoint]) -> None:
        """Store points in memory."""

        self.collections.setdefault(collection_name, {})
        for point in points:
            self.collections[collection_name][point.point_id] = point

    def search(self, collection_name: str, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Search by cosine similarity in memory."""

        points = self.collections.get(collection_name, {}).values()
        scored = [
            VectorSearchResult(
                point_id=point.point_id,
                score=_cosine_similarity(query_vector, point.vector),
                payload=point.payload,
            )
            for point in points
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return numerator / (left_norm * right_norm)
