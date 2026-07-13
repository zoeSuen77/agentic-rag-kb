"""Tests for query rewrite and ambiguity detection agent nodes."""

from __future__ import annotations

import json
from typing import Iterator

from agentic_rag_kb.agents import ambiguity_detection_node, query_rewrite_node


class FakeLLM:
    """Simple fake LLM returning queued JSON payloads."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.responses.pop(0), ensure_ascii=False)

    def stream_generate(self, prompt: str) -> Iterator[str]:
        yield self.generate(prompt)


def test_clear_question_rewrite_and_ambiguity_none() -> None:
    """A clear technical question should rewrite cleanly and not be ambiguous."""

    rewrite_llm = FakeLLM(
        [
            {
                "rewritten_query": "如何配置 PostgreSQL 数据库连接池的 max_pool_size？",
                "reason": "问题已经包含实体和配置项。",
                "missing_info": [],
            }
        ]
    )
    ambiguity_llm = FakeLLM(
        [
            {
                "is_ambiguous": False,
                "ambiguity_type": "none",
                "clarification_question": "",
                "confidence": 0.93,
            }
        ]
    )

    state = query_rewrite_node(
        {
            "original_query": "如何配置 PostgreSQL 数据库连接池的 max_pool_size？",
            "chat_history": [],
            "compression_summary": "",
        },
        rewrite_llm,
    )
    state = ambiguity_detection_node(state, ambiguity_llm)

    assert state["rewritten_query"] == "如何配置 PostgreSQL 数据库连接池的 max_pool_size？"
    assert state["query_rewrite_result"]["missing_info"] == []
    assert state["ambiguity_result"]["is_ambiguous"] is False
    assert state["ambiguity_result"]["ambiguity_type"] == "none"


def test_pronoun_reference_is_rewritten_from_existing_context_and_marked_clear() -> None:
    """省略指代问题 should use existing chat context instead of inventing entities."""

    state = query_rewrite_node(
        {
            "original_query": "这个怎么配置？",
            "chat_history": [
                {"role": "user", "content": "我们在看 PostgreSQL 数据库连接池。"},
                {"role": "assistant", "content": "已确认对象是 PostgreSQL 数据库连接池。"},
            ],
            "compression_summary": "",
        }
    )
    state = ambiguity_detection_node(state)

    assert "这个怎么配置" in state["rewritten_query"]
    assert "PostgreSQL 数据库连接池" in state["rewritten_query"]
    assert state["ambiguity_result"]["is_ambiguous"] is False


def test_pronoun_reference_without_context_is_ambiguous() -> None:
    """指代词 without prior context must be identified as missing_entity."""

    state = query_rewrite_node(
        {
            "original_query": "它有什么问题？",
            "chat_history": [],
            "compression_summary": "",
        }
    )
    state = ambiguity_detection_node(state)

    assert state["query_rewrite_result"]["missing_info"] == ["指代对象"]
    assert state["ambiguity_result"]["is_ambiguous"] is True
    assert state["ambiguity_result"]["ambiguity_type"] == "missing_entity"
    assert "具体" in state["ambiguity_result"]["clarification_question"]


def test_missing_time_range_question_is_ambiguous() -> None:
    """Questions about recent trends should ask for a time range."""

    state = ambiguity_detection_node(
        {
            "rewritten_query": "最近 API 延迟有什么变化？",
        }
    )

    assert state["ambiguity_result"]["is_ambiguous"] is True
    assert state["ambiguity_result"]["ambiguity_type"] == "missing_time"
    assert "时间范围" in state["ambiguity_result"]["clarification_question"]


def test_broad_scope_question_is_ambiguous() -> None:
    """Overly broad questions should ask the user to narrow scope."""

    state = ambiguity_detection_node({"rewritten_query": "介绍一下系统"})

    assert state["ambiguity_result"]["is_ambiguous"] is True
    assert state["ambiguity_result"]["ambiguity_type"] == "broad_scope"
    assert "范围" in state["ambiguity_result"]["clarification_question"]


def test_unclear_metric_question_is_ambiguous() -> None:
    """Metric questions with unclear metric names should be flagged."""

    state = ambiguity_detection_node({"rewritten_query": "这个服务质量是否正常？"})

    assert state["ambiguity_result"]["is_ambiguous"] is True
    assert state["ambiguity_result"]["ambiguity_type"] in {"missing_entity", "unclear_metric"}
