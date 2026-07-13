from __future__ import annotations

from app.graphs.main_graph import MainGraph


def chat(graph: MainGraph, query: str, session_id: str = "default") -> dict:
    return graph.invoke({"session_id": session_id, "user_id": "api", "raw_query": query})

