"""JSON helpers for LLM structured output."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw LLM output.

    LLMs sometimes wrap JSON in Markdown fences or add short prose. This helper
    extracts the first object-looking span and parses it.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from LLM output.")
    return value


def ensure_string_list(value: Any) -> list[str]:
    """Normalize arbitrary JSON values into a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]

