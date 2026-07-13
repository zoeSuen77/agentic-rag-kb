"""Tests for Qdrant indexing abstractions and parent docstore."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_kb.chunking.models import ChildChunk, ParentChunk
from agentic_rag_kb.indexing.docstore import ParentDocStore
from agentic_rag_kb.indexing.embeddings import DeterministicEmbeddingModel
from agentic_rag_kb.indexing.indexer import KnowledgeBaseIndexer
from agentic_rag_kb.indexing.io import read_child_chunks_jsonl, read_parent_chunks_jsonl
from agentic_rag_kb.indexing.qdrant_store import DEFAULT_COLLECTION_NAME, InMemoryVectorStore


def test_index_after_build_can_search_and_hydrate_parent(tmp_path: Path) -> None:
    """Indexed child chunks should be searchable and expandable to parent context."""

    parents, children = _sample_chunks()
    docstore = ParentDocStore(tmp_path / "docstore" / "parent_chunks.jsonl")
    vector_store = InMemoryVectorStore()
    indexer = KnowledgeBaseIndexer(
        vector_store=vector_store,
        embedding_model=DeterministicEmbeddingModel(vector_size=64),
        parent_docstore=docstore,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    indexer.build_index(parents, children)
    results = indexer.search("qdrant health check", top_k=2)

    assert results
    assert results[0]["parent_id"] == "parent_ops"
    assert results[0]["parent"]["parent_id"] == "parent_ops"
    assert "完整上下文" in results[0]["parent"]["text"]


def test_index_payload_metadata_is_complete(tmp_path: Path) -> None:
    """Qdrant payload should include child IDs, parent IDs, metadata, and sparse terms."""

    parents, children = _sample_chunks()
    vector_store = InMemoryVectorStore()
    indexer = KnowledgeBaseIndexer(
        vector_store=vector_store,
        embedding_model=DeterministicEmbeddingModel(vector_size=32),
        parent_docstore=ParentDocStore(tmp_path / "parents.jsonl"),
    )

    indexer.build_index(parents, children)

    points = vector_store.collections[DEFAULT_COLLECTION_NAME]
    payload = points["child_ops"].payload
    assert payload["child_chunk_id"] == "child_ops"
    assert payload["parent_id"] == "parent_ops"
    assert payload["doc_id"] == "doc_ops"
    assert payload["text"]
    assert payload["metadata"]["doc_id"] == "doc_ops"
    assert payload["metadata"]["parent_id"] == "parent_ops"
    assert payload["metadata"]["source_path"] == "data/raw/ops.md"
    assert payload["metadata"]["section_title"] == "运维"
    assert payload["metadata"]["chunk_index"] == 0
    assert "qdrant" in payload["sparse_terms"]
    assert payload["sparse_term_freq"]["qdrant"] >= 1


def test_parent_docstore_get_parent_and_get_many(tmp_path: Path) -> None:
    """ParentDocStore should return one or many parent chunks."""

    parents, _ = _sample_chunks()
    docstore = ParentDocStore(tmp_path / "parent_chunks.jsonl")
    docstore.save(parents)

    assert docstore.get_parent("parent_ops")["doc_id"] == "doc_ops"
    many = docstore.get_many(["parent_missing", "parent_design", "parent_ops"])
    assert [item["parent_id"] for item in many] == ["parent_design", "parent_ops"]


def test_chunk_indexing_io_reads_jsonl(tmp_path: Path) -> None:
    """Indexing IO should read chunk JSONL files produced by the chunking stage."""

    parents, children = _sample_chunks()
    parent_path = tmp_path / "parent_chunks.jsonl"
    child_path = tmp_path / "child_chunks.jsonl"
    parent_path.write_text(
        "\n".join(json.dumps(parent.to_json_dict(), ensure_ascii=False) for parent in parents) + "\n",
        encoding="utf-8",
    )
    child_path.write_text(
        "\n".join(json.dumps(child.to_json_dict(), ensure_ascii=False) for child in children) + "\n",
        encoding="utf-8",
    )

    loaded_parents = read_parent_chunks_jsonl(parent_path)
    loaded_children = read_child_chunks_jsonl(child_path)

    assert loaded_parents[0].parent_id == "parent_ops"
    assert loaded_children[0].child_id == "child_ops"


def _sample_chunks() -> tuple[list[ParentChunk], list[ChildChunk]]:
    parents = [
        ParentChunk(
            parent_id="parent_ops",
            doc_id="doc_ops",
            source_path="data/raw/ops.md",
            title="运维手册",
            title_path="运维手册 > 运维",
            chunk_index=0,
            text="完整上下文：Qdrant health check 需要检查 collection 状态和服务日志。",
            metadata={"section_title": "运维", "chunk_index": 0},
        ),
        ParentChunk(
            parent_id="parent_design",
            doc_id="doc_design",
            source_path="data/raw/design.md",
            title="系统设计",
            title_path="系统设计 > 检索",
            chunk_index=0,
            text="完整上下文：混合检索包含 dense search sparse search 和 rerank。",
            metadata={"section_title": "检索", "chunk_index": 0},
        ),
    ]
    children = [
        ChildChunk(
            child_id="child_ops",
            parent_id="parent_ops",
            doc_id="doc_ops",
            source_path="data/raw/ops.md",
            title="运维手册",
            title_path="运维手册 > 运维",
            chunk_index=0,
            text="Qdrant health check collection logs",
            metadata={
                "doc_id": "doc_ops",
                "parent_id": "parent_ops",
                "source_path": "data/raw/ops.md",
                "section_title": "运维",
                "chunk_index": 0,
            },
        ),
        ChildChunk(
            child_id="child_design",
            parent_id="parent_design",
            doc_id="doc_design",
            source_path="data/raw/design.md",
            title="系统设计",
            title_path="系统设计 > 检索",
            chunk_index=0,
            text="dense sparse hybrid retrieval rerank",
            metadata={
                "doc_id": "doc_design",
                "parent_id": "parent_design",
                "source_path": "data/raw/design.md",
                "section_title": "检索",
                "chunk_index": 0,
            },
        ),
    ]
    return parents, children
