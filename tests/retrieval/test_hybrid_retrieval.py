from __future__ import annotations

from app.indexing.hybrid_indexer import HybridIndexer
from app.indexing.schemas import ChildChunk, ParentChunk
from app.retrieval.hybrid_retriever import HybridRetriever


def test_hybrid_retrieval_finds_keyword_and_semantic_context() -> None:
    parent = ParentChunk(
        parent_id="p1",
        doc_id="d1",
        text="Kubernetes CoreDNS CrashLoopBackOff troubleshooting checks pods and kube-dns service.",
        title="k8s",
        section_path=["k8s"],
        metadata={"source": "k8s.md"},
    )
    child = ChildChunk(
        child_id="c1",
        parent_id="p1",
        doc_id="d1",
        text=parent.text,
        title="k8s",
        section_path=["k8s"],
        metadata={"source": "k8s.md"},
    )
    indexer = HybridIndexer()
    indexer.index([parent], [child])
    retriever = HybridRetriever(indexer)
    results = retriever.retrieve("CrashLoopBackOff CoreDNS", dense_top_k=5, sparse_top_k=5, fusion_top_k=5)
    assert results
    assert results[0].parent_id == "p1"

