"""Demo for LangGraph Send-style parallel retrieval fan-out.

Usage:
    python scripts/demo_send_parallel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.graph.main_graph import (  # noqa: E402
    MAIN_GRAPH_MERMAID,
    MainGraphDependencies,
    build_main_graph,
    dispatch_retrieval_subgraphs,
)
from agentic_rag_kb.graph.schema import default_main_graph_state  # noqa: E402


class DemoSubRetrievalGraph:
    """Offline subgraph double for demonstrating map-reduce behavior."""

    def invoke(self, state: dict) -> dict:
        sub_query = state["sub_query"]
        parent_id = f"parent_{state['sub_task_id']}"
        source = f"demo/{state['sub_task_id']}.md"
        return {
            **state,
            "reranked_contexts": [
                {
                    "child_id": f"child_{state['sub_task_id']}",
                    "parent_id": parent_id,
                    "text": f"{sub_query} 的相关上下文片段。",
                    "score_dense": 0.9,
                    "score_sparse": 0.7,
                    "score_fused": 0.05,
                    "rerank_score": 0.86,
                    "metadata": {"source_path": source},
                    "parent": {"source_path": source, "title_path": "Demo > Agentic RAG"},
                }
            ],
            "sub_answer": f"{sub_query}：该机制提升 RAG 的召回、精排或上下文完整性。\n\n引用来源：{source}",
            "confidence": 0.86,
            "insufficient_context": False,
            "debug": {"demo_subgraph": {"sub_task_id": state["sub_task_id"]}},
            "error_messages": [],
        }


def main() -> None:
    """Run a deterministic Send fan-out demo."""

    query = "请解释父子分层索引、混合检索和 Cross-Encoder 重排分别解决什么问题，并说明它们如何共同提升 RAG 效果。"
    state = default_main_graph_state(query)
    state["rewritten_query"] = query
    state["decomposed_tasks"] = [
        {
            "sub_task_id": "task_1",
            "sub_query": "父子分层索引解决什么问题",
            "purpose": "definition",
            "priority": 1,
            "dependencies": [],
        },
        {
            "sub_task_id": "task_2",
            "sub_query": "混合检索解决什么问题",
            "purpose": "definition",
            "priority": 2,
            "dependencies": [],
        },
        {
            "sub_task_id": "task_3",
            "sub_query": "Cross-Encoder 重排如何提升 RAG 效果",
            "purpose": "procedure",
            "priority": 3,
            "dependencies": [],
        },
    ]
    sends = dispatch_retrieval_subgraphs(state)
    graph = build_main_graph(MainGraphDependencies(retrieval_subgraph=DemoSubRetrievalGraph()))
    result = graph.invoke(state)
    print(MAIN_GRAPH_MERMAID)
    print(
        json.dumps(
            {
                "query": query,
                "send_count": len(sends),
                "send_tasks": [send.arg["sub_task_id"] for send in sends],
                "sub_answers": result["sub_answers"],
                "final_answer": result["final_answer"],
                "retrieval_debug": result["retrieval_debug"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
