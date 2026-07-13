from __future__ import annotations

from app.observability.events import Event


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def add(self, name: str, **payload) -> None:
        self.events.append(Event(name=name, payload=payload))

    def to_dict(self) -> list[dict]:
        return [{"name": event.name, "payload": event.payload, "timestamp": event.timestamp} for event in self.events]

