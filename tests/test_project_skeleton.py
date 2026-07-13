"""Smoke tests for the Agentic RAG project skeleton."""

from agentic_rag_kb.config import get_settings
from agentic_rag_kb.graph.state import MainGraphState, RetrievalSubGraphState


def test_settings_can_be_created() -> None:
    """The project should expose loadable settings from the src package."""

    settings = get_settings()
    assert settings.app_env
    assert settings.qdrant_collection_child


def test_graph_state_types_are_importable() -> None:
    """Graph state contracts should be importable before implementation begins."""

    main_state: MainGraphState = {"session_id": "test", "user_query": "hello"}
    sub_state: RetrievalSubGraphState = {"sub_query_id": "sq_1", "sub_query": "hello"}
    assert main_state["session_id"] == "test"
    assert sub_state["sub_query_id"] == "sq_1"
