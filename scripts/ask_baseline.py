"""CLI for the non-Agentic baseline RAG chain.

Usage:
    python scripts/ask_baseline.py --query "如何配置数据库连接池？"
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

from agentic_rag_kb.baseline import BaselineRAGChain  # noqa: E402
from agentic_rag_kb.config import get_settings  # noqa: E402
from agentic_rag_kb.indexing.docstore import ParentDocStore  # noqa: E402
from agentic_rag_kb.indexing.embeddings import SentenceTransformerEmbeddingModel  # noqa: E402
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, QdrantStore  # noqa: E402
from agentic_rag_kb.llm import OllamaLLMClient  # noqa: E402
from agentic_rag_kb.rerank import CrossEncoderReranker, RerankConfig  # noqa: E402
from agentic_rag_kb.retrieval import HybridRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="Ask the non-Agentic baseline RAG chain.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--docstore", type=Path, default=Path("data/docstore/parent_chunks.jsonl"))
    parser.add_argument("--top-k-dense", type=int, default=30)
    parser.add_argument("--top-k-sparse", type=int, default=30)
    parser.add_argument("--rerank-top-n", type=int, default=None)
    parser.add_argument("--final-context-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the baseline RAG chain."""

    args = parse_args()
    settings = get_settings()
    collection_name = args.collection or settings.qdrant_collection or DEFAULT_COLLECTION_NAME
    rerank_config = RerankConfig(
        enable_rerank=settings.enable_rerank,
        rerank_top_n=args.rerank_top_n or settings.rerank_top_n,
        final_context_k=args.final_context_k or settings.final_context_k,
    )
    retriever = HybridRetriever(
        vector_store=QdrantStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        embedding_model=SentenceTransformerEmbeddingModel(settings.embedding_model),
        parent_docstore=ParentDocStore(args.docstore),
        collection_name=collection_name,
    )
    reranker = CrossEncoderReranker(settings.reranker_model, config=rerank_config)
    llm_client = OllamaLLMClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
    )
    chain = BaselineRAGChain(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm_client,
        rerank_config=rerank_config,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
    )
    result = chain.ask(args.query)
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
