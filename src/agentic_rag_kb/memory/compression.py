"""Long conversation compression.

TODO:
- Summarize long histories.
- Preserve confirmed facts, system versions, constraints, and unresolved questions.
- Feed compressed memory into graph state.
"""


class ConversationCompressor:
    """Compress long conversation history into durable memory."""

    def compress(self, messages: list[dict]) -> dict:
        """Return compressed conversation memory."""

        raise NotImplementedError("Conversation compression will be implemented with Ollama/LangChain.")

