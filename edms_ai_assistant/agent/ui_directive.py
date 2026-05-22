# edms_ai_assistant/agent/ui_directive.py
"""
Non-blocking UI directive channel for LangGraph tools.

``UIDirective`` is a structured message sent from a tool to the frontend
**without** pausing the graph.  It rides on LangGraph's ``get_stream_writer()``
custom channel and is emitted as ``event: ui_update`` in the SSE stream.

Use cases:
  - Progress indicators (summarization chunk progress)
  - Partial document previews
  - Inline tool-output cards
  - Non-blocking hints and notifications

Unlike ``InterruptPayload``, a ``UIDirective``:
  - Does NOT stop the graph
  - Does NOT require ``Command(resume=...)``
  - Is NOT persisted in the checkpoint (transient)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

logger = logging.getLogger(__name__)

UI_DIRECTIVE_SCHEMA_VERSION: int = 1


class UIDirective(BaseModel):
    """Non-blocking UI message — rendered by the frontend without pausing.

    The ``component`` field names a registered frontend component
    (e.g. ``"DocumentPreviewCard"``, ``"ProgressBar"``).
    ``props`` is an opaque dict typed on the frontend side via TS schemas.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = UI_DIRECTIVE_SCHEMA_VERSION
    directive_id: str = Field(
        description="Unique id for dedup / replacement on the frontend."
    )
    component: str = Field(
        description='Frontend component name, e.g. "SummarizationProgress".'
    )
    props: dict[str, Any] = Field(
        default_factory=dict,
        description="Component-specific props (typed on frontend side).",
    )
    ttl_ms: int | None = Field(
        default=None,
        description="Auto-dismiss after this many milliseconds.",
    )
    replaces: str | None = Field(
        default=None,
        description="directive_id of a previous directive this one replaces.",
    )


UIDirectiveAdapter: TypeAdapter[UIDirective] = TypeAdapter(UIDirective)


# ── Emission helper ────────────────────────────────────────────────────────


def emit_ui(directive: UIDirective) -> None:
    """Send a non-blocking UI directive to the frontend via stream writer.

    Must be called inside a LangGraph tool node (where
    ``get_stream_writer()`` is available).

    Args:
        directive: The UI directive to emit.

    Raises:
        RuntimeError: If called outside a LangGraph execution context.
    """
    try:
        from langgraph.config import get_stream_writer
    except ImportError:
        logger.warning("emit_ui: langgraph.config.get_stream_writer not available")
        return

    try:
        writer = get_stream_writer()
        payload = UIDirectiveAdapter.dump_python(directive, mode="json")
        writer({"ui": payload})
    except Exception:
        logger.exception("emit_ui: failed to write UIDirective to stream")


# ── State helper ───────────────────────────────────────────────────────────


# In AgentState, add:
#   last_ui_directives: Annotated[
#       dict[str, str],   # directive_id -> component
#       operator.or_,
#   ] = {}
#
# When a directive is emitted, also update state for reconnect:
#   return {"last_ui_directives": {directive.directive_id: directive.component}}


__all__ = [
    "UI_DIRECTIVE_SCHEMA_VERSION",
    "UIDirective",
    "UIDirectiveAdapter",
    "emit_ui",
]
