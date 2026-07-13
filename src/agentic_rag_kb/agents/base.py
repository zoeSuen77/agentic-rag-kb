"""Base agent protocol for graph nodes."""

from typing import Protocol


class Agent(Protocol):
    """Protocol implemented by all Agentic RAG agents."""

    def run(self, state: dict) -> dict:
        """Process graph state and return state updates."""

