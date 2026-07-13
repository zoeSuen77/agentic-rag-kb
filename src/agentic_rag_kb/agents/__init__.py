"""Agent layer.

Agents wrap graph nodes with domain-specific responsibilities such as query
decomposition, ambiguity detection, retrieval planning, answer aggregation, and
fallback decisions.
"""

from agentic_rag_kb.agents.ambiguity import AmbiguityDetectionAgent, ambiguity_detection_node
from agentic_rag_kb.agents.base import Agent
from agentic_rag_kb.agents.clarification import clarification_node, fallback_node
from agentic_rag_kb.agents.query_decomposer import QueryDecomposerAgent, task_decomposition_node
from agentic_rag_kb.agents.query_rewrite import QueryRewriteAgent, query_rewrite_node

__all__ = [
    "Agent",
    "AmbiguityDetectionAgent",
    "QueryDecomposerAgent",
    "QueryRewriteAgent",
    "ambiguity_detection_node",
    "clarification_node",
    "fallback_node",
    "task_decomposition_node",
    "query_rewrite_node",
]
