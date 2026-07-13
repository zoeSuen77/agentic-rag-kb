"""Tests for Human-in-the-loop clarification behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agentic_rag_kb.agents.clarification import MAX_CLARIFICATION_LOOPS, fallback_node
from agentic_rag_kb.graph.hitl_demo import LocalHITLDemo, LocalInterrupt


def test_local_hitl_demo_interrupts_and_resumes() -> None:
    """The local demo should interrupt, accept clarification, and continue."""

    demo = LocalHITLDemo()
    interrupted = demo.start("这个怎么部署？")

    assert isinstance(interrupted, LocalInterrupt)
    assert interrupted.clarification_question == "你指的是哪个模块的部署？"

    final_state = demo.resume(interrupted, "Qdrant 和 Gradio")

    assert final_state["user_clarification"] == "Qdrant 和 Gradio"
    assert "Qdrant 和 Gradio" in final_state["rewritten_query"]
    assert final_state["ambiguity_result"]["is_ambiguous"] is False
    assert "继续进入后续任务拆解" in final_state["final_answer"]


def test_fallback_node_after_loop_limit() -> None:
    """Fallback should provide templates after too many clarification attempts."""

    state = fallback_node(
        {
            "original_query": "这个怎么弄？",
            "loop_count": MAX_CLARIFICATION_LOOPS,
            "error_messages": [],
        }
    )

    assert "问题仍然不够明确" in state["final_answer"]
    assert "请问【系统/模块】" in state["final_answer"]
    assert "clarification_loop_limit_exceeded" in state["error_messages"]


def test_demo_hitl_script_runs() -> None:
    """The demo script should show interrupt and resume output."""

    result = subprocess.run(
        [sys.executable, "scripts/demo_hitl.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Assistant interrupt: 你指的是哪个模块的部署？" in result.stdout
    assert "User clarification: Qdrant 和 Gradio" in result.stdout
    assert "Resumed." in result.stdout
