"""Conversation memory storage interface.

TODO:
- Implement local JSON store for development.
- Add persistent store integration for production.
- Separate raw turns, compressed summaries, and confirmed facts.
"""


class ConversationMemoryStore:
    """Store and retrieve conversation memory by session."""

    def load(self, session_id: str) -> dict:
        """Load memory for one session."""

        raise NotImplementedError("Memory loading will be implemented in the memory phase.")

    def save(self, session_id: str, memory: dict) -> None:
        """Save memory for one session."""

        raise NotImplementedError("Memory saving will be implemented in the memory phase.")

