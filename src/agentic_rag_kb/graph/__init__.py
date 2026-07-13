"""LangGraph orchestration layer.

The graph module coordinates the main Agentic RAG workflow and retrieval subgraphs.
It will use LangGraph StateGraph, conditional routing, Send API fan-out, loop limits,
fallback edges, and human-in-the-loop interrupts.
"""

from agentic_rag_kb.graph.main_graph import build_main_graph

__all__ = ["build_main_graph"]

