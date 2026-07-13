"""Tests for Dense+Sparse hybrid retrieval and reranking."""

from __future__ import annotations

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.indexing.docstore import ParentDocStore
from agentic_rag_kb.indexing.embeddings import DeterministicEmbeddingModel
from agentic_rag_kb.indexing.indexer import KnowledgeBaseIndexer
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, InMemoryVectorStore
from agentic_rag_kb.rerank import LexicalReranker
from agentic_rag_kb.retrieval import HybridRetriever
from agentic_rag_kb.retrieval.hybrid import reciprocal_rank_fusion


def test_dense_only_retrieval_returns_parent_context(tmp_path) -> None:
    """Dense retrieval should return parent-expanded context."""

    retriever = _build_retriever(tmp_path)

    results = retriever.retrieve(
        "database connection pool timeout",
        top_k_dense=1,
        top_k_sparse=0,
        final_k=1,
    )

    assert len(results) == 1
    assert results[0].child_id == "child_db"
    assert results[0].score_dense is not None
    assert results[0].text == results[0].parent["text"]
    assert "数据库连接池完整上下文" in results[0].text


def test_sparse_only_retrieval_finds_exact_error_code(tmp_path) -> None:
    """Sparse retrieval should catch exact technical terms."""

    retriever = _build_retriever(tmp_path)

    results = retriever.retrieve(
        "ORA-12514",
        top_k_dense=0,
        top_k_sparse=2,
        final_k=1,
    )

    assert len(results) == 1
    assert results[0].child_id == "child_ora"
    assert results[0].score_sparse is not None
    assert "ORA-12514 完整上下文" in results[0].text


def test_hybrid_retrieval_uses_rrf_and_debug_info(tmp_path) -> None:
    """Hybrid retrieval should fuse dense and sparse rankings and expose debug info."""

    retriever = _build_retriever(tmp_path)

    results = retriever.retrieve(
        "qdrant hybrid dense sparse retrieval",
        top_k_dense=3,
        top_k_sparse=3,
        final_k=3,
    )
    debug = retriever.get_debug_info()

    assert results
    assert all(result.score_fused > 0 for result in results)
    assert debug.dense_hits
    assert debug.sparse_hits
    assert debug.fused_ranking
    assert debug.parent_contexts


def test_parent_level_duplicate_control(tmp_path) -> None:
    """One parent should not fill the entire result set when per-parent limit is one."""

    retriever = _build_retriever(tmp_path, max_children_per_parent=1)

    results = retriever.retrieve(
        "database pool timeout connection",
        top_k_dense=4,
        top_k_sparse=4,
        final_k=4,
    )

    parent_ids = [result.parent_id for result in results]
    assert parent_ids.count("parent_db") == 1


def test_rrf_combines_dense_and_sparse_scores() -> None:
    """RRF should deduplicate child IDs and sum rank contributions."""

    fused = reciprocal_rank_fusion(
        dense_hits=[
            {"child_id": "a", "parent_id": "p1", "text": "a", "score_dense": 0.9, "metadata": {}},
            {"child_id": "b", "parent_id": "p2", "text": "b", "score_dense": 0.8, "metadata": {}},
        ],
        sparse_hits=[
            {"child_id": "b", "parent_id": "p2", "text": "b", "score_sparse": 4.0, "metadata": {}},
            {"child_id": "c", "parent_id": "p3", "text": "c", "score_sparse": 3.0, "metadata": {}},
        ],
        rrf_k=60,
    )

    assert [item["child_id"] for item in fused][:2] == ["b", "a"]
    assert fused[0]["score_dense"] == 0.8
    assert fused[0]["score_sparse"] == 4.0


def test_lexical_reranker_orders_parent_contexts(tmp_path) -> None:
    """Second-stage reranking should sort retrieved parent contexts."""

    retriever = _build_retriever(tmp_path)
    candidates = retriever.retrieve(
        "ORA-12514 listener service name",
        top_k_dense=3,
        top_k_sparse=3,
        final_k=3,
    )

    reranked = LexicalReranker().rerank("ORA-12514 listener service name", candidates, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].parent_id == "parent_ora"
    assert reranked[0].rerank_score is not None


def _build_retriever(tmp_path, max_children_per_parent: int = 2) -> HybridRetriever:
    parents, children = _sample_chunks()
    vector_store = InMemoryVectorStore()
    embedding_model = DeterministicEmbeddingModel(vector_size=96)
    docstore = ParentDocStore(tmp_path / "parent_chunks.jsonl")
    indexer = KnowledgeBaseIndexer(
        vector_store=vector_store,
        embedding_model=embedding_model,
        parent_docstore=docstore,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    indexer.build_index(parents, children)
    return HybridRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        parent_docstore=docstore,
        collection_name=DEFAULT_COLLECTION_NAME,
        max_children_per_parent=max_children_per_parent,
    )


def _sample_chunks() -> tuple[list[ParentChunk], list[ChildChunk]]:
    parents = [
        ParentChunk(
            parent_id="parent_db",
            doc_id="doc_db",
            source_path="data/raw/db.md",
            title="数据库手册",
            title_path="数据库手册 > 连接池",
            chunk_index=0,
            text="数据库连接池完整上下文：connection pool timeout max pool size idle timeout retry policy。",
            metadata={"section_title": "连接池"},
        ),
        ParentChunk(
            parent_id="parent_ora",
            doc_id="doc_ora",
            source_path="data/raw/oracle.md",
            title="Oracle 故障",
            title_path="Oracle 故障 > ORA-12514",
            chunk_index=0,
            text="ORA-12514 完整上下文：listener service name registration database instance。",
            metadata={"section_title": "ORA-12514"},
        ),
        ParentChunk(
            parent_id="parent_search",
            doc_id="doc_search",
            source_path="data/raw/search.md",
            title="检索设计",
            title_path="检索设计 > 混合召回",
            chunk_index=0,
            text="混合检索完整上下文：qdrant hybrid dense sparse retrieval rrf rerank。",
            metadata={"section_title": "混合召回"},
        ),
    ]
    children = [
        _child("child_db", "parent_db", "doc_db", "database connection pool timeout", "连接池"),
        _child("child_db_2", "parent_db", "doc_db", "pool max size idle timeout", "连接池"),
        _child("child_ora", "parent_ora", "doc_ora", "ORA-12514 listener service name", "ORA-12514"),
        _child("child_search", "parent_search", "doc_search", "qdrant hybrid dense sparse retrieval", "混合召回"),
    ]
    return parents, children


def _child(child_id: str, parent_id: str, doc_id: str, text: str, section_title: str) -> ChildChunk:
    return ChildChunk(
        child_id=child_id,
        parent_id=parent_id,
        doc_id=doc_id,
        source_path=f"data/raw/{doc_id}.md",
        title=doc_id,
        title_path=f"{doc_id} > {section_title}",
        chunk_index=0,
        text=text,
        metadata={
            "doc_id": doc_id,
            "parent_id": parent_id,
            "source_path": f"data/raw/{doc_id}.md",
            "section_title": section_title,
            "chunk_index": 0,
        },
    )
