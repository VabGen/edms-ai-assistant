# edms_ai_assistant/api/sse.py
"""
Server-Sent Events helpers for the v2 chat API.

Wire format (text/event-stream):

    event: <kind>
    data: <single-line JSON>

    event: <kind>
    data: <single-line JSON>

Recognised event kinds:
    - ``token``     — partial AI message content delta
    - ``message``   — fully formed assistant message (final)
    - ``interrupt`` — HITL request, payload follows InterruptPayload schema
    - ``state``     — synchronisation snapshot (used on reconnect)
    - ``done``      — terminal marker for a stream
    - ``error``     — terminal error
"""

from __future__ import annotations

import json
from typing import Any, Final

SSE_KEEPALIVE: Final[bytes] = b": keepalive\n\n"


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Render a single SSE event line.

    ``data`` is serialised as a single JSON line — multi-line ``data:``
    fields are allowed by the spec but break a lot of naive clients.
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


__all__ = ["SSE_KEEPALIVE", "format_sse"]
