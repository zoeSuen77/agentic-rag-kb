"""Tests for the non-Agentic baseline RAG chain."""

from __future__ import annotations

from typing import Iterator

from agentic_rag_kb.baseline import BaselineRAGChain
from agentic_rag_kb.baseline.prompts import build_baseline_prompt
from agentic_rag_kb.rerank import LexicalReranker, RerankConfig
from agentic_rag_kb.retrieval.models import RetrievedParentContext, RetrievalDebugInfo


class FakeRetriever:
    """Deterministic retriever for baseline tests."""

    def __init__(self, contexts: list[RetrievedParentContext]) -> None:
        self.contexts = contexts
        self.debug = RetrievalDebugInfo(
            dense_hits=[{"child_id": context.child_id} for context in contexts],
            sparse_hits=[],
            fused_ranking=[{"child_id": context.child_id, "rank": index + 1} for index, context in enumerate(contexts)],
            parent_contexts=[{"parent_id": context.parent_id, "parent_found": True} for context in contexts],
        )

    def retrieve(self, query: str, top_k_dense: int, top_k_sparse: int, final_k: int):
        return self.contexts[:final_k]

    def get_debug_info(self):
        return self.debug


class FakeLLMClient:
    """Fake LLM that records prompts and returns a stable answer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "source=data/raw/db.md" in prompt:
            return "数据库连接池需要配置 max_pool_size 和 idle_timeout。\n\n引用来源：data/raw/db.md"
        return "不知道。当前上下文不足。\n\n引用来源：无"

    def stream_generate(self, prompt: str) -> Iterator[str]:
        yield self.generate(prompt)


def test_empty_context_does_not_hallucinate() -> None:
    """When retrieval returns no contexts, baseline should not call LLM or invent facts."""

    llm = FakeLLMClient()
    chain = BaselineRAGChain(
        retriever=FakeRetriever([]),
        reranker=LexicalReranker(RerankConfig(final_context_k=5)),
        llm_client=llm,
        rerank_config=RerankConfig(rerank_top_n=5, final_context_k=5),
    )

    result = chain.ask("如何配置数据库连接池？")

    assert "不知道" in result.answer
    assert result.contexts == []
    assert result.citations == []
    assert llm.prompts == []


def test_context_answer_includes_source_citation() -> None:
    """With context, baseline prompt should include source and output citations."""

    llm = FakeLLMClient()
    chain = BaselineRAGChain(
        retriever=FakeRetriever([_db_context()]),
        reranker=LexicalReranker(RerankConfig(final_context_k=1)),
        llm_client=llm,
        rerank_config=RerankConfig(rerank_top_n=3, final_context_k=1),
    )

    result = chain.ask("如何配置数据库连接池？")

    assert "data/raw/db.md" in result.answer
    assert result.citations[0]["source_path"] == "data/raw/db.md"
    assert "只能基于给定上下文回答" in llm.prompts[0]
    assert "source=data/raw/db.md" in llm.prompts[0]


def test_output_format_is_stable() -> None:
    """Baseline result should expose answer, contexts, citations, and retrieval_debug."""

    chain = BaselineRAGChain(
        retriever=FakeRetriever([_db_context()]),
        reranker=LexicalReranker(RerankConfig(final_context_k=1)),
        llm_client=FakeLLMClient(),
        rerank_config=RerankConfig(rerank_top_n=3, final_context_k=1),
    )

    result = chain.ask("如何配置数据库连接池？").to_json_dict()

    assert set(result) == {"answer", "contexts", "citations", "retrieval_debug"}
    assert isinstance(result["answer"], str)
    assert isinstance(result["contexts"], list)
    assert isinstance(result["citations"], list)
    assert isinstance(result["retrieval_debug"], dict)


def test_prompt_requires_grounded_answer_and_unknown_when_insufficient() -> None:
    """Prompt should encode grounding and unknown-answer requirements."""

    prompt = build_baseline_prompt("怎么配置？", [_db_context()])

    assert "只能基于给定上下文回答" in prompt
    assert "如果上下文不足" in prompt
    assert "引用来源" in prompt


def _db_context() -> RetrievedParentContext:
    return RetrievedParentContext(
        child_id="child_db",
        parent_id="parent_db",
        text="数据库连接池上下文：max_pool_size idle_timeout connection timeout。",
        score_dense=0.9,
        score_sparse=2.0,
        score_fused=0.08,
        metadata={"source_path": "data/raw/db.md", "title_path": "数据库 > 连接池"},
        parent={
            "parent_id": "parent_db",
            "source_path": "data/raw/db.md",
            "title_path": "数据库 > 连接池",
            "text": "数据库连接池上下文：max_pool_size idle_timeout connection timeout。",
        },
        rerank_score=0.7,
    )
