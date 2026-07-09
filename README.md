# Agentic RAG Knowledge Base QA

Enterprise-grade Agentic RAG system for internal technical documentation.

This project is designed as a production-oriented knowledge base QA system, not a toy RAG demo. It combines hierarchical parent-child indexing, dense and sparse hybrid retrieval, cross-encoder reranking, LangGraph-based multi-agent orchestration, human-in-the-loop clarification, long-context memory compression, streaming UI, and RAGAS evaluation.

## Goals

- Upload, parse, clean, and chunk enterprise technical documents.
- Build parent-child hierarchical indexes for precise retrieval and complete generation context.
- Support dense + sparse hybrid retrieval with rank fusion.
- Apply cross-encoder reranking before answer generation.
- Use a LangGraph main graph plus subgraph agent architecture.
- Decompose complex questions automatically.
- Use LangGraph `Send` API to dispatch multiple retrieval subagents in parallel.
- Let each retrieval subgraph complete an independent retrieval lifecycle.
- Aggregate multiple subanswers into one grounded final answer.
- Detect ambiguous queries and trigger human clarification.
- Add loop limits and fallback behavior for empty, weak, or conflicting retrieval.
- Compress long conversation history into durable memory.
- Provide a Gradio streaming chat interface.
- Evaluate the system with RAGAS metrics:
  - AnswerCorrectness
  - ContextRecall
  - Faithfulness
  - ContextPrecision

## Tech Stack

- Python
- LangGraph
- LangChain
- Qdrant
- Ollama
- RAGAS
- Gradio
- FastAPI, optional

## Architecture

```text
User Query
   |
   v
Main LangGraph Agent
   |
   +--> Load conversation memory
   +--> Compress long context
   +--> Detect ambiguity
   +--> Human clarification, when needed
   +--> Classify query type
   +--> Detect complexity
   +--> Decompose complex question
   +--> Dispatch sub retrieval agents in parallel with Send API
   |
   +--> Sub Retrieval Agent 1
   +--> Sub Retrieval Agent 2
   +--> Sub Retrieval Agent N
   |
   v
Aggregate Subanswers
   |
   v
Faithfulness / Consistency Check
   |
   +--> Fallback or retry when needed
   |
   v
Final Grounded Answer with Citations
```

## Planned Directory Structure

```text
agentic-rag-kb/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   ├── app.yaml
│   ├── model.yaml
│   ├── qdrant.yaml
│   ├── retriever.yaml
│   └── ragas.yaml
├── data/
│   ├── raw/
│   ├── parsed/
│   ├── cleaned/
│   ├── chunks/
│   └── eval/
├── app/
│   ├── main.py
│   ├── settings.py
│   ├── api/
│   ├── ui/
│   ├── ingestion/
│   ├── indexing/
│   ├── retrieval/
│   ├── llm/
│   ├── graphs/
│   ├── memory/
│   ├── evaluation/
│   ├── observability/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── graph/
│   ├── retrieval/
│   └── evaluation/
└── scripts/
    ├── ingest_docs.py
    ├── build_index.py
    ├── run_api.py
    ├── run_eval.py
    └── run_gradio.py
```

## Core Modules

### Ingestion

The ingestion pipeline converts raw enterprise documents into structured parent-child chunks.

```text
raw file
-> loader
-> parser
-> cleaner
-> parent splitter
-> child splitter
-> metadata enrichment
-> indexable document records
```

Supported document types will include:

- PDF
- Markdown
- HTML
- TXT
- DOCX
- exported wiki pages
- API documentation
- troubleshooting manuals

### Parent-Child Hierarchical Index

The system stores both child chunks and parent chunks.

```text
Document
└── Parent Section
    ├── Child Chunk 1
    ├── Child Chunk 2
    └── Child Chunk 3
```

Child chunks are used for precise dense and sparse retrieval. Parent chunks are used for context expansion before reranking and answer generation.

Example record:

```json
{
  "doc_id": "doc_001",
  "parent_id": "parent_001",
  "child_id": "child_001",
  "title": "Kubernetes Troubleshooting Guide",
  "section_path": ["Networking", "DNS Resolution Failure"],
  "child_text": "CoreDNS pods may fail DNS resolution when upstream DNS is unreachable...",
  "parent_text": "Full section text...",
  "metadata": {
    "source": "k8s_troubleshooting.pdf",
    "page": 12,
    "version": "2026-01",
    "department": "infra",
    "tags": ["kubernetes", "dns", "coredns"]
  }
}
```

### Indexing

The indexing layer writes vectors, sparse signals, text, and metadata into Qdrant.

Planned indexes:

- Dense vector index for semantic retrieval.
- Sparse vector or BM25-style index for keyword, error code, config key, and identifier retrieval.
- Parent-child mapping for context expansion.
- Metadata filters for source, version, department, document type, and ACL.

### Retrieval

Each retrieval subagent executes the full retrieval lifecycle:

```text
sub question
-> query rewrite
-> dense retrieval
-> sparse retrieval
-> reciprocal rank fusion
-> parent expansion
-> cross-encoder reranking
-> context packing
-> local answer generation
```

Dense retrieval handles semantic questions. Sparse retrieval handles exact technical terms, error codes, config names, class names, function names, and log snippets.

### Reranking

The cross-encoder receives `(query, context)` pairs and reranks candidates after hybrid retrieval and parent expansion.

The first implementation should use:

- hybrid top_k: 40 to 60
- rerank top_n: 5 to 10
- context packing with citations and metadata

## LangGraph Design

### Main Graph Responsibilities

- Load session memory.
- Compress long conversation context.
- Detect query ambiguity.
- Trigger human clarification.
- Classify query type.
- Detect question complexity.
- Decompose complex questions.
- Dispatch sub retrieval tasks in parallel with LangGraph `Send`.
- Aggregate subanswers and contexts.
- Check consistency and faithfulness.
- Apply fallback or retry when needed.
- Generate the final cited answer.
- Update memory.

### Main Graph State

```python
class MainGraphState(TypedDict):
    session_id: str
    user_id: str
    raw_query: str
    normalized_query: str
    conversation_history: list[dict]
    compressed_history: str
    ambiguity_result: dict
    clarification_question: str
    human_clarification: str
    query_type: str
    complexity_level: str
    decomposed_questions: list[dict]
    sub_tasks: list[dict]
    sub_results: list[dict]
    aggregated_contexts: list[dict]
    aggregated_answer: str
    final_answer: str
    citations: list[dict]
    loop_count: int
    max_loops: int
    fallback_reason: str
    errors: list[dict]
    evaluation_trace: dict
```

### Sub Retrieval Graph Responsibilities

- Rewrite a single subquery.
- Run dense retrieval.
- Run sparse retrieval.
- Fuse results.
- Expand child hits to parent contexts.
- Rerank contexts with a cross-encoder.
- Build the final context block.
- Generate a local grounded answer.
- Return citations and retrieval quality metadata.

### Sub Retrieval State

```python
class SubRetrievalState(TypedDict):
    sub_query_id: str
    parent_query: str
    sub_query: str
    rewritten_query: str
    intent: str
    dense_results: list[dict]
    sparse_results: list[dict]
    fused_results: list[dict]
    parent_contexts: list[dict]
    reranked_contexts: list[dict]
    local_answer: str
    local_citations: list[dict]
    retrieval_quality: dict
    error: str
```

## Human-in-the-loop

Human clarification is a first-class graph path, not a UI-only feature.

It is triggered when:

- the user query is ambiguous;
- required slots are missing;
- retrieval evidence conflicts;
- the answer would involve risky operational actions;
- faithfulness checks fail;
- loop limits are reached.

Example:

```text
User: How do I roll it back?
System: Which rollback target do you mean: Kubernetes Deployment, database migration, or application release?
User: Kubernetes Deployment.
```

## Fallback and Loop Limits

The graph maintains `loop_count` and `max_loops`.

Fallback levels:

1. Rewrite the query and retry retrieval.
2. Increase top_k or lower score thresholds.
3. Reduce decomposition breadth.
4. Ask the user for clarification.
5. Return a conservative answer that clearly states missing evidence.

## Long Conversation Memory

Memory is split into:

- recent messages;
- compressed session summary;
- confirmed facts;
- open questions;
- user and environment constraints.

The compression node preserves operationally important facts, such as system version, deployment environment, confirmed symptoms, and rejected hypotheses.

## Gradio UI

The Gradio interface should support:

- streaming chat;
- document upload;
- index rebuild;
- query clarification;
- subquestion trace display;
- retrieved context display;
- citations;
- RAGAS evaluation entry point.

## RAGAS Evaluation

Evaluation will run on curated QA samples and generated traces.

Required metrics:

- AnswerCorrectness
- ContextRecall
- Faithfulness
- ContextPrecision

Dataset shape:

```json
{
  "question": "How should CoreDNS intermittent resolution failures be investigated?",
  "ground_truth": "Check CoreDNS pods, kube-dns service, network policies, NodeLocal DNSCache, and upstream DNS health.",
  "answer": "Generated system answer",
  "contexts": ["Retrieved context 1", "Retrieved context 2"],
  "reference": "Expert answer"
}
```

Reports should include:

- average metric scores;
- low-score sample list;
- per-query-type breakdown;
- retrieval failure analysis;
- faithfulness failure analysis.

## Implementation Roadmap

### Phase 1: Project Skeleton

- Create package layout.
- Add configuration loading.
- Add logging.
- Connect Ollama.
- Connect Qdrant.

Tests:

- settings load correctly;
- Ollama health check works;
- Qdrant health check works.

### Phase 2: Ingestion and Parent-Child Chunking

- Parse PDF, Markdown, TXT, and HTML.
- Clean noisy text.
- Build parent and child chunks.
- Attach metadata and stable IDs.

Tests:

- chunk sizes are bounded;
- parent-child relationships are valid;
- metadata is preserved.

### Phase 3: Dense and Sparse Indexing

- Generate embeddings.
- Build sparse retrieval signals.
- Write child chunks and parent chunks into Qdrant.

Tests:

- indexed counts match input chunks;
- payload schema is valid;
- child hits can resolve parent chunks.

### Phase 4: Hybrid Retrieval

- Implement dense retriever.
- Implement sparse retriever.
- Implement reciprocal rank fusion.
- Implement parent expansion.

Tests:

- semantic queries are retrieved by dense search;
- error-code queries are retrieved by sparse search;
- fusion removes duplicates;
- parent expansion restores complete context.

### Phase 5: Cross-Encoder Reranking

- Add reranker interface.
- Rerank fused candidates.
- Pack final contexts.

Tests:

- relevant contexts move upward;
- irrelevant contexts are filtered out;
- rerank scores are logged.

### Phase 6: Sub Retrieval Graph

- Build LangGraph subgraph.
- Implement full single-subquery lifecycle.

Tests:

- subgraph returns local answer;
- citations are present;
- retrieval quality metadata is present.

### Phase 7: Main Graph

- Add memory loading.
- Add ambiguity detection.
- Add query classification.
- Add complexity detection.
- Add decomposition.

Tests:

- simple questions skip decomposition;
- complex questions produce multiple subquestions;
- ambiguous questions route to clarification.

### Phase 8: Parallel Multi-Agent Retrieval

- Use LangGraph `Send` API.
- Dispatch multiple sub retrieval graphs in parallel.
- Aggregate sub-results.

Tests:

- all subqueries execute;
- failed subtask does not crash the graph;
- aggregated results preserve subquery IDs.

### Phase 9: Aggregation, Faithfulness, and Fallback

- Merge subanswers.
- Deduplicate citations.
- Check answer support.
- Add retry and fallback behavior.

Tests:

- empty retrieval triggers fallback;
- conflicting evidence triggers clarification;
- loop limit prevents infinite retries.

### Phase 10: Streaming UI

- Build Gradio chat.
- Stream graph progress events.
- Show citations and retrieval trace.
- Support document upload.

Tests:

- streaming works;
- clarification loop works;
- uploaded documents can be indexed.

### Phase 11: RAGAS

- Build evaluation dataset tools.
- Run required RAGAS metrics.
- Generate reports.

Tests:

- evaluation runs on a small dataset;
- reports are generated;
- weak retrieval produces lower context scores.

## Resume Positioning

Suggested resume bullet:

```text
Built an enterprise-grade Agentic RAG knowledge base QA system with LangGraph, Qdrant, Ollama, and RAGAS, supporting parent-child hierarchical indexing, dense/sparse hybrid retrieval, cross-encoder reranking, parallel multi-agent retrieval through LangGraph Send API, human-in-the-loop clarification, long-context memory compression, streaming Gradio UI, and automated RAGAS evaluation.
```

Suggested interview explanation:

```text
This was not a single-chain RAG system. I designed a main graph and retrieval subgraph architecture. The main graph handles ambiguity detection, question decomposition, task routing, fallback, and final aggregation. Each subgraph independently performs query rewriting, hybrid retrieval, parent expansion, reranking, and local answer generation. For complex questions, the main graph dispatches multiple subgraphs in parallel through LangGraph Send API, then merges the grounded subanswers into one cited final answer.
```

## Git Policy

Every meaningful project change should be committed with Git.

Recommended commit style:

```text
docs: add architecture readme
feat: add ingestion pipeline
feat: add parent child chunk builder
feat: add hybrid retriever
feat: add retrieval subgraph
feat: add main agent graph
test: add ragas smoke evaluation
```
