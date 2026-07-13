"""CLI for indexing child chunks into Qdrant.

Usage:
    python scripts/index_chunks.py --child data/chunks/child_chunks.jsonl --parent data/chunks/parent_chunks.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.config import get_settings  # noqa: E402
from agentic_rag_kb.indexing.docstore import ParentDocStore  # noqa: E402
from agentic_rag_kb.indexing.indexer import KnowledgeBaseIndexer  # noqa: E402
from agentic_rag_kb.indexing.io import read_child_chunks_jsonl, read_parent_chunks_jsonl  # noqa: E402
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, QdrantStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Index child chunks into Qdrant.")
    parser.add_argument("--child", type=Path, default=Path("data/chunks/child_chunks.jsonl"))
    parser.add_argument("--parent", type=Path, default=Path("data/chunks/parent_chunks.jsonl"))
    parser.add_argument("--docstore", type=Path, default=Path("data/docstore/parent_chunks.jsonl"))
    parser.add_argument("--collection", default=None)
    return parser.parse_args()


def main() -> None:
    """Run indexing."""

    args = parse_args()
    settings = get_settings()
    parents = read_parent_chunks_jsonl(args.parent)
    children = read_child_chunks_jsonl(args.child)
    collection_name = args.collection or settings.qdrant_collection or DEFAULT_COLLECTION_NAME
    indexer = KnowledgeBaseIndexer(
        vector_store=QdrantStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        parent_docstore=ParentDocStore(args.docstore),
        collection_name=collection_name,
    )
    indexer.build_index(parents, children)
    print(
        f"Indexed {len(children)} child chunks into Qdrant collection "
        f"`{collection_name}` and stored {len(parents)} parents in {args.docstore}"
    )


if __name__ == "__main__":
    main()
