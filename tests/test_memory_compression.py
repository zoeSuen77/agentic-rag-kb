"""Tests for long conversation memory compression."""

from __future__ import annotations

from agentic_rag_kb.graph.main_graph import MainGraphDependencies, memory_check_node, memory_update_node
from agentic_rag_kb.graph.schema import default_main_graph_state
from agentic_rag_kb.memory import CompressionTrigger, ConversationCompressor


class NoopSubgraph:
    """No-op retrieval subgraph dependency."""

    def invoke(self, state: dict) -> dict:
        return state


def _dependencies(trigger: CompressionTrigger | None = None) -> MainGraphDependencies:
    return MainGraphDependencies(
        retrieval_subgraph=NoopSubgraph(),
        memory_trigger=trigger or CompressionTrigger(max_token_estimate=500, max_turns=8),
        memory_compressor=ConversationCompressor(),
    )


def _long_history() -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    payloads = [
        "项目目标是构建企业内部技术文档的 Agentic RAG 知识库问答系统，不要简化成普通 RAG。",
        "必须包含父子分层索引、Dense + Sparse 混合检索、Cross-Encoder 重排和 LangGraph Send API。",
        "已确认技术栈使用 Python、LangGraph、Qdrant、Ollama、Gradio、RAGAS 和 pytest。",
        "不能遗忘的约束：不得编造引用，每个重要结论后面必须带 source。",
        "待澄清问题：线上部署时 Qdrant collection 命名和 Ollama 模型版本需要确认？",
    ]
    for index in range(10):
        turns.append({"role": "user", "content": f"第 {index + 1} 轮用户补充：{payloads[index % len(payloads)]}"})
        turns.append({"role": "assistant", "content": f"已确认第 {index + 1} 轮事实，并会保留这些工程约束。"})
    return turns


def test_memory_check_does_not_compress_below_threshold() -> None:
    state = default_main_graph_state("short question")
    state["chat_history"] = [{"role": "user", "content": "如何配置 Qdrant？"}]

    result = memory_check_node(
        state,
        _dependencies(CompressionTrigger(max_token_estimate=5000, max_turns=20)),
    )

    assert result["compression_summary"] == ""
    assert result["chat_history"] == state["chat_history"]
    assert result["memory_debug"]["memory_check"]["should_compress"] is False


def test_memory_check_compresses_when_threshold_reached() -> None:
    state = default_main_graph_state("继续实现压缩")
    state["chat_history"] = _long_history()

    result = memory_check_node(state, _dependencies())

    assert result["memory_debug"]["memory_check"]["should_compress"] is True
    assert result["compression_summary"]
    assert len(result["chat_history"]) == 4
    assert result["compression_stats"]["original_token_estimate"] > result["compression_stats"]["compressed_token_estimate"]


def test_summary_preserves_important_constraints() -> None:
    state = default_main_graph_state("继续")
    state["chat_history"] = _long_history()

    result = memory_check_node(state, _dependencies())
    summary = result["compression_summary"]

    assert "不能遗忘的约束" in summary
    assert "不得编造引用" in summary
    assert "source" in summary
    assert "LangGraph" in summary
    assert "Qdrant" in summary


def test_compression_ratio_in_target_range() -> None:
    state = default_main_graph_state("继续")
    state["chat_history"] = _long_history()

    result = memory_check_node(state, _dependencies())
    ratio = result["compression_stats"]["compression_ratio"]

    assert 0.55 <= ratio <= 0.7
    assert result["compression_stats"]["status"] == "ok"


def test_memory_update_appends_current_turn() -> None:
    state = default_main_graph_state("什么是父子分层索引？")
    state["final_answer"] = "父子分层索引用父块保留上下文，用子块提升检索精度。"

    result = memory_update_node(state)

    assert result["chat_history"][-2]["role"] == "user"
    assert result["chat_history"][-1]["role"] == "assistant"
    assert result["memory_debug"]["memory_update"]["turn_count"] == 2
