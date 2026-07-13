"""Minimal Human-in-the-loop clarification demo.

Run:
    python scripts/demo_hitl.py

The demo uses a local simulator when LangGraph is not installed. The production
graph builder in `agentic_rag_kb.graph.hitl_demo` uses LangGraph interrupt.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agentic_rag_kb.graph.hitl_demo import LocalHITLDemo, LocalInterrupt, build_hitl_demo_graph  # noqa: E402


def main() -> None:
    """Run a deterministic HITL clarification demo."""

    original_query = "这个怎么部署？"
    user_clarification = "Qdrant 和 Gradio"
    print(f"User: {original_query}")

    try:
        build_hitl_demo_graph()
        print("LangGraph is installed. Use the compiled graph with Command(resume=...) in an app runtime.")
    except RuntimeError:
        pass

    demo = LocalHITLDemo()
    interrupted = demo.start(original_query)
    if isinstance(interrupted, LocalInterrupt):
        print(f"Assistant interrupt: {interrupted.clarification_question}")
        print(f"User clarification: {user_clarification}")
        final_state = demo.resume(interrupted, user_clarification)
    else:
        final_state = interrupted

    print("Resumed.")
    print(f"rewritten_query: {final_state.get('rewritten_query')}")
    print(f"final_answer: {final_state.get('final_answer')}")


if __name__ == "__main__":
    main()

