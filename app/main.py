from __future__ import annotations

from app.graphs.main_graph import build_main_graph
from app.indexing.hybrid_indexer import HybridIndexer
from app.settings import load_settings


def create_app_graph():
    settings = load_settings()
    indexer = HybridIndexer.load_or_create(settings.index_dir)
    return build_main_graph(indexer=indexer, settings=settings)


if __name__ == "__main__":
    graph = create_app_graph()
    result = graph.invoke({"session_id": "local", "user_id": "cli", "raw_query": "How does this RAG system work?"})
    print(result["final_answer"])
