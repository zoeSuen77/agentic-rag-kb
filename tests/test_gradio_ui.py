"""Tests for Gradio UI backend service."""

from __future__ import annotations

import json

from agentic_rag_kb.evaluation import RagasEvaluator
from agentic_rag_kb.ui import GradioRAGService


def test_ui_service_builds_index_and_streams_answer(tmp_path) -> None:
    document = tmp_path / "kb.txt"
    document.write_text(
        "Qdrant hybrid search combines dense semantic retrieval and sparse keyword retrieval. "
        "Cross-Encoder reranking improves final context precision.",
        encoding="utf-8",
    )
    service = GradioRAGService()
    state = service.empty_state()

    state, status_text, status_json = service.build_index([str(document)], state)

    assert "索引构建完成" in status_text
    assert status_json["child_chunk_count"] >= 1

    history, state, clarification, debug, status = service.start_chat("Qdrant hybrid search 解决什么问题？", [], state)
    assert status
    assert clarification == ""
    outputs = list(service.stream_answer("Qdrant hybrid search 解决什么问题？", "", history, state))

    final_history, final_state, _, final_debug, final_status = outputs[-1]
    assert final_status == "完成。"
    assert "引用来源" in final_history[-1]["content"]
    assert final_debug["decomposed_tasks"]
    assert final_state.last_debug["final_contexts"]


def test_ragas_evaluator_fallback_metrics_from_jsonl(tmp_path) -> None:
    dataset = tmp_path / "eval_dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "hybrid retrieval?",
                "answer": "hybrid retrieval combines dense and sparse retrieval",
                "ground_truth": "dense sparse retrieval",
                "contexts": ["dense sparse retrieval context"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = RagasEvaluator().evaluate_jsonl(dataset)

    assert result.sample_count == 1
    assert set(result.metrics) == {
        "AnswerCorrectness",
        "ContextRecall",
        "Faithfulness",
        "ContextPrecision",
    }
    assert result.metrics["ContextRecall"] > 0
