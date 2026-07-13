"""CLI for hybrid retrieval followed by Cross-Encoder reranking.

Usage:
    python scripts/rerank.py --query "如何配置数据库连接池？"
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
from agentic_rag_kb.rerank import CrossEncoderReranker, RerankConfig  # noqa: E402
from agentic_rag_kb.retrieval import HybridRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Retrieve and rerank parent contexts.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--docstore", type=Path, default=Path("data/docstore/parent_chunks.jsonl"))
    parser.add_argument("--top-k-dense", type=int, default=30)
    parser.add_argument("--top-k-sparse", type=int, default=30)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--final-context-k", type=int, default=None)
    parser.add_argument("--disable-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run hybrid retrieval and Cross-Encoder reranking."""

    args = parse_args()
    settings = get_settings()
    collection_name = args.collection or settings.qdrant_collection or DEFAULT_COLLECTION_NAME
    config = RerankConfig(
        enable_rerank=(settings.enable_rerank and not args.disable_rerank),
        rerank_top_n=args.rerank_top_n or settings.rerank_top_n,
        final_context_k=args.final_context_k or settings.final_context_k,
    )
    retriever = HybridRetriever(
        vector_store=QdrantStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        parent_docstore=ParentDocStore(args.docstore),
        collection_name=collection_name,
    )
    candidates = retriever.retrieve(
        query=args.query,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
        final_k=config.rerank_top_n,
    )
    reranker = CrossEncoderReranker(settings.reranker_model, config=config)
    contexts = reranker.rerank(args.query, candidates)
    print(
        json.dumps(
            {
                "query": args.query,
                "rerank_config": {
                    "enable_rerank": config.enable_rerank,
                    "rerank_top_n": config.rerank_top_n,
                    "final_context_k": config.final_context_k,
                },
                "contexts": [context.to_json_dict() for context in contexts],
                "retrieval_debug": retriever.get_debug_info().to_json_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
