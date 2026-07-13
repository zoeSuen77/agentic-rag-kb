"""Tests for parent-child hierarchical chunking."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agentic_rag_kb.chunking.io import read_documents_jsonl
from agentic_rag_kb.chunking.parent_child import ParentChildChunker
from agentic_rag_kb.chunking.text_splitter import estimate_tokens


def test_parent_child_chunking_relationships_and_metadata(tmp_path: Path) -> None:
    """Every child chunk should map to an existing parent and carry required metadata."""

    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(_sample_documents_jsonl(), encoding="utf-8")
    documents = read_documents_jsonl(input_path)

    chunker = ParentChildChunker(
        parent_min_tokens=60,
        parent_target_tokens=120,
        parent_max_tokens=180,
        child_min_tokens=20,
        child_target_tokens=45,
        child_max_tokens=70,
    )
    parents, children = chunker.split(documents)

    assert parents
    assert children
    parent_ids = {parent.parent_id for parent in parents}
    assert all(child.parent_id in parent_ids for child in children)
    assert all(parent.text.strip() for parent in parents)
    assert all(child.text.strip() for child in children)
    assert max(estimate_tokens(parent.text) for parent in parents) <= 180
    assert max(estimate_tokens(child.text) for child in children) <= 70

    for child in children:
        assert child.metadata["doc_id"] == child.doc_id
        assert child.metadata["parent_id"] == child.parent_id
        assert child.metadata["source_path"]
        assert "section_title" in child.metadata
        assert isinstance(child.metadata["chunk_index"], int)


def test_title_path_preserves_heading_hierarchy(tmp_path: Path) -> None:
    """Chunk metadata should preserve readable title paths."""

    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(_sample_documents_jsonl(), encoding="utf-8")
    documents = read_documents_jsonl(input_path)
    chunker = ParentChildChunker(parent_min_tokens=60, parent_target_tokens=120, parent_max_tokens=180)

    parents, _ = chunker.split(documents)

    assert any(
        "系统设计" in parent.title_path and "检索模块" in parent.title_path
        for parent in parents
    )


def test_chunking_report_contains_expected_statistics(tmp_path: Path) -> None:
    """ChunkingReport should summarize document and chunk counts."""

    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(_sample_documents_jsonl(), encoding="utf-8")
    documents = read_documents_jsonl(input_path)
    chunker = ParentChildChunker(parent_min_tokens=60, parent_target_tokens=120, parent_max_tokens=180)
    parents, children = chunker.split(documents)

    report = chunker.build_report(len(documents), parents, children)

    assert report.document_count == len(documents)
    assert report.parent_chunk_count == len(parents)
    assert report.child_chunk_count == len(children)
    assert report.max_parent_tokens >= report.min_parent_tokens
    assert report.max_child_tokens >= report.min_child_tokens


def test_build_chunks_cli_writes_outputs(tmp_path: Path) -> None:
    """The CLI should write parent chunks, child chunks, and report files."""

    input_path = tmp_path / "documents.jsonl"
    output_dir = tmp_path / "chunks"
    input_path.write_text(_sample_documents_jsonl(), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_chunks.py",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Built" in result.stdout
    parent_rows = _read_jsonl(output_dir / "parent_chunks.jsonl")
    child_rows = _read_jsonl(output_dir / "child_chunks.jsonl")
    report = json.loads((output_dir / "chunking_report.json").read_text(encoding="utf-8"))
    assert parent_rows
    assert child_rows
    assert report["parent_chunk_count"] == len(parent_rows)
    assert report["child_chunk_count"] == len(child_rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sample_documents_jsonl() -> str:
    repeated_design = " ".join(
        [
            "混合召回需要同时使用 dense semantic search sparse keyword search rerank parent context"
            for _ in range(45)
        ]
    )
    repeated_ops = " ".join(
        [
            "部署排查需要检查 qdrant ollama langgraph gradio ragas 日志 指标 配置"
            for _ in range(25)
        ]
    )
    rows = [
        {
            "doc_id": "doc_design",
            "source_path": "data/raw/design.md",
            "title": "系统设计",
            "page_number": None,
            "section_title": "检索模块",
            "text": (
                "# 系统设计\n\n"
                "## 检索模块\n\n"
                "### 混合召回\n\n"
                f"{repeated_design}\n\n"
                "### 重排\n\n"
                f"{repeated_ops}"
            ),
            "metadata": {"file_type": "markdown"},
        }
    ]
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
