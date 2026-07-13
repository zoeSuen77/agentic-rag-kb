"""Agent layer.

Agents wrap graph nodes with domain-specific responsibilities such as query
decomposition, ambiguity detection, retrieval planning, answer aggregation, and
fallback decisions.
"""

from agentic_rag_kb.agents.base import Agent

__all__ = ["Agent"]

