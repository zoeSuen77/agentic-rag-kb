# Agentic RAG Knowledge Base QA

企业内部技术文档 Agentic RAG 知识库问答系统。

当前阶段只搭建项目骨架，不一次性实现全部功能。代码采用 `src/` 分层结构，每个模块先定义职责、接口、类名、函数名和 TODO，后续按模块逐步实现。

## Tech Stack

- Python 3.10+
- LangGraph
- LangChain
- Qdrant client
- sentence-transformers
- rank-bm25 或 fastembed sparse
- cross-encoder reranker
- Ollama
- Gradio
- RAGAS
- pytest

## Project Structure

```text
agentic-rag-kb/
├── src/
│   └── agentic_rag_kb/
│       ├── config/
│       ├── document_loader/
│       ├── chunking/
│       ├── indexing/
│       ├── retrieval/
│       ├── rerank/
│       ├── graph/
│       ├── agents/
│       ├── memory/
│       ├── evaluation/
│       └── ui/
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

## Architecture Draft

```text
Document Upload
  -> Document Loader
  -> Cleaning
  -> Parent / Child Chunking
  -> Dense + Sparse Indexing
  -> Qdrant

User Query
  -> Main LangGraph
  -> Ambiguity Detection
  -> Human Clarification when needed
  -> Query Decomposition
  -> Parallel Retrieval Subagents through Send API
  -> Hybrid Retrieval
  -> Parent Expansion
  -> Cross-Encoder Rerank
  -> Subanswer Generation
  -> Aggregation
  -> Final Answer with Citations
```

## Development Plan

1. Build project skeleton and interfaces.
2. Implement document loading and parent-child chunking.
3. Implement dense, sparse, and hybrid indexing.
4. Implement retrieval and parent expansion.
5. Add cross-encoder reranking.
6. Build LangGraph main graph and retrieval subgraph.
7. Add human-in-the-loop clarification and loop fallback.
8. Add memory compression.
9. Add Gradio streaming UI.
10. Add RAGAS evaluation.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Document Ingestion

Parse PDF, Markdown, TXT, and DOCX files from `data/raw` into standardized JSONL:

```bash
python scripts/ingest_documents.py --input data/raw --output data/processed/documents.jsonl
```

Each output row contains:

- `doc_id`
- `source_path`
- `title`
- `page_number`
- `section_title`
- `text`
- `metadata`

## Parent-Child Chunking

Build hierarchical parent and child chunks from parsed documents:

```bash
python scripts/build_chunks.py --input data/processed/documents.jsonl --output data/chunks
```

Outputs:

- `data/chunks/parent_chunks.jsonl`
- `data/chunks/child_chunks.jsonl`
- `data/chunks/chunking_report.json`

## Qdrant Indexing

Index child chunks into Qdrant and store parent chunks locally for context expansion:

```bash
python scripts/index_chunks.py --child data/chunks/child_chunks.jsonl --parent data/chunks/parent_chunks.jsonl
```

Defaults:

- Qdrant collection: `agentic_rag_chunks`
- Parent docstore: `data/docstore/parent_chunks.jsonl`
