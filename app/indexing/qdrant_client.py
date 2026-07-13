from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QdrantConnectionConfig:
    url: str
    api_key: str | None = None


class QdrantUnavailable(RuntimeError):
    pass


def create_qdrant_client(config: QdrantConnectionConfig):
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # pragma: no cover
        raise QdrantUnavailable("qdrant-client is not installed") from exc
    return QdrantClient(url=config.url, api_key=config.api_key)

