"""LangGraph orchestration layer.

The graph module coordinates the main Agentic RAG workflow and retrieval subgraphs.
It will use LangGraph StateGraph, conditional routing, Send API fan-out, loop limits,
fallback edges, and human-in-the-loop interrupts.
"""

from agentic_rag_kb.graph.main_graph import build_main_graph
from agentic_rag_kb.graph.schema import (
    MAIN_GRAPH_NODE_IO,
    RETRIEVAL_SUBGRAPH_NODE_IO,
    default_main_graph_state,
    default_retrieval_subgraph_state,
)
from agentic_rag_kb.graph.state import MainGraphState, RetrievalSubGraphState

__all__ = [
    "MAIN_GRAPH_NODE_IO",
    "RETRIEVAL_SUBGRAPH_NODE_IO",
    "MainGraphState",
    "RetrievalSubGraphState",
    "build_main_graph",
    "default_main_graph_state",
    "default_retrieval_subgraph_state",
]
