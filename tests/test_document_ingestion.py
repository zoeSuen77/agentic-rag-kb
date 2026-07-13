"""Tests for document ingestion parsing and cleaning."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_rag_kb.document_loader.cleaners import clean_document_text
from agentic_rag_kb.document_loader.markdown_parser import MarkdownDocumentParser
from agentic_rag_kb.document_loader.router import DocumentLoaderRouter
from agentic_rag_kb.document_loader.txt_parser import TxtDocumentParser
from scripts.ingest_documents import write_jsonl


def test_clean_document_text_removes_noise_and_preserves_code_and_tables() -> None:
    """Cleaning should remove obvious noise while keeping code fences and tables."""

    raw = """Company Confidential

# Install Guide

Company Confidential

```python
print("hello")
```

| key | value |
| --- | --- |
| port | 6333 |

Page 1

Company Confidential
"""

    cleaned = clean_document_text(raw)

    assert "Company Confidential" not in cleaned
    assert "Page 1" not in cleaned
    assert "# Install Guide" in cleaned
    assert 'print("hello")' in cleaned
    assert "| port | 6333 |" in cleaned


def test_txt_parser_returns_standard_document(tmp_path: Path) -> None:
    """TXT parser should emit a standardized Document object."""

    file_path = tmp_path / "runbook.txt"
    file_path.write_text("Runbook\n\n\nCheck Qdrant health.", encoding="utf-8")

    documents = TxtDocumentParser().load(file_path)

    assert len(documents) == 1
    assert documents[0].doc_id.startswith("doc_")
    assert documents[0].source_path == file_path
    assert documents[0].title == "runbook"
    assert documents[0].text == "Runbook\n\nCheck Qdrant health."
    assert documents[0].metadata["file_type"] == "txt"


def test_markdown_parser_preserves_sections_code_and_tables(tmp_path: Path) -> None:
    """Markdown parser should preserve headings, code blocks, and table text."""

    file_path = tmp_path / "guide.md"
    file_path.write_text(
        """# Agentic RAG

Intro text.

## Deploy

```bash
ollama serve
```

| item | value |
| --- | --- |
| qdrant | 6333 |
""",
        encoding="utf-8",
    )

    documents = MarkdownDocumentParser().load(file_path)

    assert len(documents) == 2
    assert documents[0].title == "Agentic RAG"
    assert documents[0].section_title == "Agentic RAG"
    assert documents[1].section_title == "Deploy"
    assert "```bash" in documents[1].text
    assert "| qdrant | 6333 |" in documents[1].text


def test_router_discovers_supported_files_and_ignores_unknown(tmp_path: Path) -> None:
    """Router should select parsers by extension."""

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Beta\ncontent", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x,y", encoding="utf-8")

    router = DocumentLoaderRouter()
    documents = router.load(tmp_path)

    assert len(documents) == 2
    assert {document.metadata["file_type"] for document in documents} == {"txt", "markdown"}


def test_write_jsonl_outputs_document_rows(tmp_path: Path) -> None:
    """JSONL writer should serialize one document per line."""

    output_path = tmp_path / "documents.jsonl"
    rows = [{"doc_id": "doc_1", "text": "hello"}, {"doc_id": "doc_2", "text": "world"}]

    write_jsonl(output_path, rows)

    loaded = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert loaded == rows
