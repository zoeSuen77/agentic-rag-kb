from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any


@dataclass(slots=True)
class Event:
    name: str
    payload: dict[str, Any]
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time()

