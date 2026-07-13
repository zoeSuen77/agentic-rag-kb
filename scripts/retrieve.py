"""CLI for hybrid retrieval and reranking.

Usage:
    python scripts/retrieve.py --query "如何配置数据库连接池？"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.config import get_settings  # noqa: E402
from agentic_rag_kb.indexing.docstore import ParentDocStore  # noqa: E402
from agentic_rag_kb.indexing.embeddings import SentenceTransformerEmbeddingModel  # noqa: E402
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, QdrantStore  # noqa: E402
from agentic_rag_kb.rerank import CrossEncoderReranker  # noqa: E402
from agentic_rag_kb.retrieval import HybridRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Run Dense+Sparse hybrid retrieval.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--docstore", type=Path, default=Path("data/docstore/parent_chunks.jsonl"))
    parser.add_argument("--top-k-dense", type=int, default=10)
    parser.add_argument("--top-k-sparse", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run retrieval and print JSON results."""

    args = parse_args()
    settings = get_settings()
    collection_name = args.collection or settings.qdrant_collection or DEFAULT_COLLECTION_NAME
    retriever = HybridRetriever(
        vector_store=QdrantStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        parent_docstore=ParentDocStore(args.docstore),
        collection_name=collection_name,
    )
    results = retriever.retrieve(
        query=args.query,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
        final_k=args.final_k,
    )
    if not args.no_rerank:
        reranker = CrossEncoderReranker(settings.reranker_model)
        results = reranker.rerank(args.query, results, args.rerank_top_n)
    output = {
        "query": args.query,
        "results": [result.to_json_dict() for result in results],
        "debug": retriever.get_debug_info().to_json_dict(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
