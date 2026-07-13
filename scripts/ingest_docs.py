from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion.pipeline import ingest_path
from app.indexing.hybrid_indexer import HybridIndexer
from app.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    parents, children = ingest_path(args.path)
    indexer = HybridIndexer(settings.embedding_dimensions)
    indexer.index(parents, children)
    indexer.save(settings.index_dir)
    print(f"indexed parents={len(parents)} children={len(children)}")


if __name__ == "__main__":
    main()

