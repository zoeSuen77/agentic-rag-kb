from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(value: Any, prefix: str = "") -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}" if prefix else digest

