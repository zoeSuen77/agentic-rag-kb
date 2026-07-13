from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return fallback or {}

