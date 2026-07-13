from __future__ import annotations

from app.graphs.main_graph import build_main_graph
from app.indexing.hybrid_indexer import HybridIndexer
from app.indexing.schemas import ChildChunk, ParentChunk
from app.settings import load_settings


def test_main_graph_decomposes_and_answers_with_citations() -> None:
    parent = ParentChunk(
        parent_id="p1",
        doc_id="d1",
        text="For HTTP 502, check ingress gateway upstream timeout, service discovery endpoints, and pod readiness.",
        title="gateway",
        section_path=["gateway"],
        metadata={"source": "gateway.md"},
    )
    child = ChildChunk(
        child_id="c1",
        parent_id="p1",
        doc_id="d1",
        text=parent.text,
        title="gateway",
        section_path=["gateway"],
        metadata={"source": "gateway.md"},
    )
    indexer = HybridIndexer()
    indexer.index([parent], [child])
    graph = build_main_graph(indexer, load_settings())
    state = graph.invoke(
        {
            "session_id": "test",
            "user_id": "tester",
            "raw_query": "502 如何排查，服务发现和网关分别要看什么？",
        }
    )
    assert state["sub_results"]
    assert "引用" in state["final_answer"]

