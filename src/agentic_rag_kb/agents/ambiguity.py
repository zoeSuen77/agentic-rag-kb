"""Ambiguity detection agent.

TODO:
- Detect missing entities, versions, systems, and environments.
- Generate concise clarification questions for human-in-the-loop flow.
"""


class AmbiguityDetectionAgent:
    """Agent that decides whether the user query needs clarification."""

    def run(self, state: dict) -> dict:
        """Return ambiguity result and optional clarification question."""

        raise NotImplementedError("Ambiguity detection will be implemented with rules plus LLM validation.")

