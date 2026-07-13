"""JSON helpers for LLM structured output."""

from __future__ import annotations

import json
import re
from typing import Any

from agentic_rag_kb.llm import LLMClient


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


def generate_json_with_retry(
    llm_client: LLMClient,
    prompt: str,
    *,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], list[str]]:
    """Generate and parse JSON, retrying once with a stricter repair prompt."""

    errors: list[str] = []
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        raw = llm_client.generate(current_prompt)
        try:
            return parse_json_object(raw), errors
        except Exception as exc:
            errors.append(f"json_parse_attempt_{attempt}_failed: {exc}")
            current_prompt = (
                f"{prompt}\n\n上一次输出不是合法 JSON。请重新输出一个 JSON object，"
                "不要输出 Markdown、解释文字或代码块。"
            )
    raise ValueError("; ".join(errors))
