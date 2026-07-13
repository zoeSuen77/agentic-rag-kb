"""Demo for long conversation memory compression.

Usage:
    python scripts/demo_memory_compression.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.graph.main_graph import MainGraphDependencies, memory_check_node  # noqa: E402
from agentic_rag_kb.graph.schema import default_main_graph_state  # noqa: E402
from agentic_rag_kb.memory import CompressionTrigger, ConversationCompressor, estimate_turn_tokens  # noqa: E402


class DemoNoopSubgraph:
    """No-op dependency for memory-only demo."""

    def invoke(self, state: dict) -> dict:
        return state


def main() -> None:
    """Simulate 10 long turns and show compression output."""

    state = default_main_graph_state("继续实现 memory/context_compression 模块")
    state["chat_history"] = _long_chat_history()
    before_tokens = estimate_turn_tokens(state["chat_history"])
    result = memory_check_node(
        state,
        MainGraphDependencies(
            retrieval_subgraph=DemoNoopSubgraph(),
            memory_trigger=CompressionTrigger(max_token_estimate=500, max_turns=8),
            memory_compressor=ConversationCompressor(),
        ),
    )
    print(
        json.dumps(
            {
                "turns_before": len(state["chat_history"]),
                "turns_after": len(result["chat_history"]),
                "token_estimate_before": before_tokens,
                "compression_summary": result["compression_summary"],
                "compression_stats": result["compression_stats"],
                "memory_debug": result["memory_debug"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _long_chat_history() -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    topics = [
        "项目目标是构建企业内部技术文档的 Agentic RAG 知识库问答系统，不能简化成普通 RAG。",
        "必须包含父子分层索引、Dense + Sparse 混合检索、Cross-Encoder 重排和 LangGraph Send API。",
        "用户长期目标是把项目写进简历，并能在面试中讲清楚 Agentic RAG 架构价值。",
        "已确认使用 Python、LangGraph、LangChain、Qdrant、Ollama、Gradio、RAGAS 和 pytest。",
        "不能遗忘的约束：不得编造引用，每个重要结论必须带 source，测试必须覆盖关键路径。",
    ]
    for index in range(10):
        turns.append({"role": "user", "content": f"第 {index + 1} 轮：{topics[index % len(topics)]}"})
        turns.append(
            {
                "role": "assistant",
                "content": (
                    f"已确认第 {index + 1} 轮要求，并继续按模块化工程实现。"
                    "如果问题涉及这个模块，需要先澄清具体组件。"
                ),
            }
        )
    return turns


if __name__ == "__main__":
    main()
