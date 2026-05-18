# edms_ai_assistant/api/sse_events.py
"""
Extractors and SSE builders for structured UI-component events.

When a tool (e.g. ``doc_compliance_check``) returns structured JSON we
intercept the ``ToolMessage`` *before* the LLM rewrites it as prose,
and emit a dedicated ``event: ui_component`` SSE frame so the frontend
can render an interactive widget (ComplianceResult, navigate, …).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage


# ── Internal helpers ──────────────────────────────────────────────────────


def _parse_tool_content(content: Any) -> Any:
    """Parse ToolMessage.content into a dict (best-effort)."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                return item
    return None


# ── Public extractors ─────────────────────────────────────────────────────


def extract_compliance_from_tool_message(msg: ToolMessage) -> dict | None:
    """Return compliance payload dict if *msg* looks like a compliance result."""
    data = _parse_tool_content(msg.content)
    if not data or not isinstance(data, dict):
        return None
    if data.get("status") == "success" and "fields" in data and "overall" in data:
        return data
    return None


def extract_navigate_url_from_tool_message(msg: ToolMessage) -> str | None:
    """Return a navigate_url string if *msg* contains one or can derive one.

    Three strategies:
      1. Explicit ``navigate_url`` field in the payload.
      2. Regex fallback over raw string content.
      3. **Auto-derive** from ``document_id`` when status is ``success``
         but the payload is NOT a compliance result (no ``overall``/``fields``).
         This covers ``create_document_from_file`` and similar tools that
         return ``{"status": "success", "document_id": "…"}``.
    """
    data = _parse_tool_content(msg.content)
    if not data or not isinstance(data, dict):
        # Fallback: regex over raw string content
        if isinstance(msg.content, str):
            match = re.search(
                r'"navigate_url"\s*:\s*"(/document-form/[^"]+)"',
                msg.content,
            )
            if match:
                return match.group(1)
        return None

    if data.get("navigate_url"):
        return data["navigate_url"]

    if (
        data.get("status") == "success"
        and data.get("document_id")
        and "overall" not in data
        and "fields" not in data
    ):
        doc_id = data["document_id"]
        return f"/document-form/{doc_id}"

    return None


# ── SSE builders ──────────────────────────────────────────────────────────


def build_compliance_sse_event(data: dict) -> str:
    """Build an ``event: ui_component`` SSE frame for a compliance result."""
    event_data = {
        "type": "compliance_result",
        "overall": data.get("overall"),
        "summary": data.get("summary"),
        "document_id": data.get("document_id"),
        "document_category": data.get("document_category"),
        "fields": data.get("fields", []),
        "stats": data.get("stats"),
        "fix_hint": data.get("fix_hint"),
    }
    payload = json.dumps(event_data, ensure_ascii=False, default=str)
    return f"event: ui_component\ndata: {payload}\n\n"


def build_navigate_sse_event(url: str) -> str:
    """Build an ``event: ui_component`` SSE frame for a navigate directive."""
    event_data = {"type": "navigate", "url": url}
    payload = json.dumps(event_data, ensure_ascii=False)
    return f"event: ui_component\ndata: {payload}\n\n"