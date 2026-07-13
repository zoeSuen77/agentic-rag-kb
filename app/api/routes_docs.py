from __future__ import annotations

from pathlib import Path

from app.ingestion.pipeline import ingest_path
from app.indexing.hybrid_indexer import HybridIndexer


def ingest_documents(indexer: HybridIndexer, path: Path) -> dict:
    parents, children = ingest_path(path)
    indexer.index(parents, children)
    return {"parents": len(parents), "children": len(children)}

