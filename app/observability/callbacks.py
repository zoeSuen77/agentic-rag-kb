from __future__ import annotations

from app.observability.trace import TraceRecorder


def default_recorder() -> TraceRecorder:
    return TraceRecorder()

