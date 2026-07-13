"""Non-Agentic RAG baseline chain.

This chain is intentionally simple and linear:

query -> hybrid retrieve -> rerank -> build prompt -> Ollama LLM -> answer

It provides a baseline for comparing future Agentic RAG workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from agentic_rag_kb.baseline.prompts import build_baseline_prompt
from agentic_rag_kb.llm import LLMClient
from agentic_rag_kb.rerank import RerankConfig
from agentic_rag_kb.retrieval.models import RetrievedParentContext


class RetrieverLike(Protocol):
    """Protocol for hybrid retrievers used by the baseline chain."""

    def retrieve(
        self,
        query: str,
        top_k_dense: int,
        top_k_sparse: int,
        final_k: int,
    ) -> list[RetrievedParentContext]:
        """Return parent-expanded candidate contexts."""

    def get_debug_info(self):
        """Return retrieval debug info."""


class RerankerLike(Protocol):
    """Protocol for rerankers used by the baseline chain."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedParentContext],
        top_n: int | None = None,
        final_context_k: int | None = None,
    ) -> list[RetrievedParentContext]:
        """Rerank contexts."""


@dataclass(slots=True)
class BaselineRAGResult:
    """Stable result object returned by the baseline RAG chain."""

    answer: str
    contexts: list[dict]
    citations: list[dict]
    retrieval_debug: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        """Return JSON-serializable output."""

        return asdict(self)


class BaselineRAGChain:
    """Linear, non-Agentic RAG baseline."""

    def __init__(
        self,
        retriever: RetrieverLike,
        reranker: RerankerLike,
        llm_client: LLMClient,
        rerank_config: RerankConfig | None = None,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.rerank_config = rerank_config or RerankConfig()
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse

    def ask(self, query: str) -> BaselineRAGResult:
        """Answer a query with the baseline RAG flow."""

        candidates = self.retriever.retrieve(
            query=query,
            top_k_dense=self.top_k_dense,
            top_k_sparse=self.top_k_sparse,
            final_k=self.rerank_config.rerank_top_n,
        )
        contexts = self.reranker.rerank(
            query,
            candidates,
            top_n=self.rerank_config.rerank_top_n,
            final_context_k=self.rerank_config.final_context_k,
        )
        citations = build_citations(contexts)

        if not contexts:
            answer = "不知道。当前上下文不足，无法基于知识库回答该问题。\n\n引用来源：无"
        else:
            prompt = build_baseline_prompt(query, contexts)
            answer = self.llm_client.generate(prompt)

        debug_info = self.retriever.get_debug_info()
        return BaselineRAGResult(
            answer=answer,
            contexts=[context.to_json_dict() for context in contexts],
            citations=citations,
            retrieval_debug=debug_info.to_json_dict() if hasattr(debug_info, "to_json_dict") else dict(debug_info),
        )


def build_citations(contexts: list[RetrievedParentContext]) -> list[dict]:
    """Build citation records from parent-expanded contexts."""

    citations: list[dict] = []
    for index, context in enumerate(contexts, start=1):
        parent = context.parent or {}
        source = context.metadata.get("source_path") or parent.get("source_path")
        citations.append(
            {
                "citation_id": index,
                "source_path": source,
                "parent_id": context.parent_id,
                "child_id": context.child_id,
                "title_path": parent.get("title_path") or context.metadata.get("title_path"),
                "rerank_score": context.rerank_score,
                "score_fused": context.score_fused,
            }
        )
    return citations
