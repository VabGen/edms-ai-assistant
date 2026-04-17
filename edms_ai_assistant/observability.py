# edms_ai_assistant/observability.py
"""
Agent Observability — structured tracing, metrics, and token tracking.

Best Practice (2026): Every agent request produces a trace span with
structured metadata.  This enables:
- Latency analysis per intent / tool
- Token budget monitoring
- Error-rate dashboards
- A/B experiment comparison

Inspired by Anthropic's internal observability stack and OpenTelemetry
semantic conventions for LLM workloads.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Span kinds ────────────────────────────────────────────────────────────────


class SpanKind(str, Enum):
    """Trace span classification."""

    REQUEST = "request"
    AGENT = "agent"
    INTENT_CLASSIFICATION = "intent_classification"
    PROMPT_BUILD = "prompt_build"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    GUARDRAIL = "guardrail"
    ORCHESTRATION = "orchestration"


# ─── Span data ────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """
    A single observability span — lightweight, no external dependency.

    Attributes:
        name: Human-readable span name (e.g. ``"tool_call:doc_get_details"``).
        kind: Classification of the span.
        start_time: Monotonic clock at span creation.
        duration_ms: Elapsed wall-clock milliseconds (set on ``finish()``).
        attributes: Arbitrary key-value metadata attached to the span.
        status: ``"ok"`` | ``"error"`` | ``"unset"``.
    """

    name: str
    kind: SpanKind
    start_time: float = field(default_factory=time.monotonic)
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "unset"

    def finish(self, status: str = "ok", *, metadata: dict[str, Any] | None = None) -> None:
        """Mark the span as completed and compute duration."""
        self.duration_ms = (time.monotonic() - self.start_time) * 1000
        self.status = status
        if metadata:
            self.attributes.update(metadata)

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a key-value pair to the span."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Serialize span for logging / export."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "duration_ms": round(self.duration_ms, 1),
            "status": self.status,
            "attributes": self.attributes,
        }


# ─── Trace — collection of spans for a single request ────────────────────────


@dataclass
class Trace:
    """
    A complete trace for one agent request.

    Collects spans, accumulates token usage, and produces a summary dict
    suitable for structured logging or external export (Jaeger, Datadog, etc.).

    Attributes:
        name: Trace name (e.g. ``"agent.chat"``).
        kind: Primary span kind for the trace.
        request_id: Unique identifier for the request (thread_id or UUID).
        intent: Detected user intent (set after classification).
        metadata: Arbitrary key-value metadata attached to the trace.
        spans: Ordered list of spans.
        token_usage: Accumulated token counts.
    """

    name: str = "agent"
    kind: SpanKind = SpanKind.AGENT
    request_id: str = ""
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    _start_time: float = field(default_factory=time.monotonic, repr=False)
    _finished: bool = field(default=False, repr=False)

    def start_span(self, name: str, kind: SpanKind, **attrs: Any) -> Span:
        """Create and register a new child span."""
        span = Span(name=name, kind=kind, attributes=attrs)
        self.spans.append(span)
        return span

    def add_span(self, name: str, kind: SpanKind, **attrs: Any) -> Span:
        """Alias for ``start_span`` — create and register a new child span."""
        return self.start_span(name=name, kind=kind, **attrs)

    def add_token_usage(self, prompt: int = 0, completion: int = 0) -> None:
        """Accumulate token counts from an LLM response."""
        self.token_usage["prompt_tokens"] += prompt
        self.token_usage["completion_tokens"] += completion
        self.token_usage["total_tokens"] += prompt + completion

    def finish(self, status: str = "ok", *, metadata: dict[str, Any] | None = None) -> None:
        """Mark the trace as completed and log the summary."""
        if self._finished:
            return
        self._finished = True
        if metadata:
            self.metadata.update(metadata)
        self.log_summary()

    def summary(self) -> dict[str, Any]:
        """
        Produce a structured summary for logging.

        Returns:
            Dict with request metadata, span details, and token totals.
        """
        total_ms = (time.monotonic() - self._start_time) * 1000
        return {
            "name": self.name,
            "kind": self.kind.value,
            "request_id": self.request_id,
            "intent": self.intent,
            "total_duration_ms": round(total_ms, 1),
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }

    def log_summary(self) -> None:
        """Emit the trace summary as a structured log entry."""
        logger.info(
            "Agent trace complete",
            extra={"trace": self.summary()},
        )


# ─── Sync context manager for easy span creation ────────────────────────────


@contextmanager
def trace_span(name: str, *, kind: SpanKind = SpanKind.ORCHESTRATION, metadata: dict[str, Any] | None = None):
    """
    Sync context manager that creates a standalone span, yields it, and finishes on exit.

    Useful for wrapping synchronous blocks (e.g. guardrail checks) without
    requiring a parent ``Trace`` object.  The span is logged on exit.

    Usage::

        with trace_span("guardrail.check", kind=SpanKind.GUARDRAIL) as span:
            result = pipeline.run(content)
            span.set_attribute("blocked", result.blocked)
        # span is now finished and logged
    """
    span = Span(name=name, kind=kind, attributes=metadata or {})
    try:
        yield span
        span.finish(status="ok")
    except Exception:
        span.finish(status="error")
        raise
    finally:
        logger.debug(
            "trace_span complete",
            extra={"span": span.to_dict()},
        )


# ─── Async context manager for easy span creation with a parent Trace ───────


@asynccontextmanager
async def async_trace_span(trace: Trace, name: str, kind: SpanKind, **attrs: Any):
    """
    Async context manager that creates a span on a parent trace.

    Usage::

        async with async_trace_span(trace, "tool_call:doc_get_details", SpanKind.TOOL_CALL) as span:
            result = await tool.arun(...)
            span.set_attribute("tool_status", "success")
        # span is now finished and recorded on the trace
    """
    span = trace.start_span(name, kind, **attrs)
    try:
        yield span
        span.finish(status="ok")
    except Exception:
        span.finish(status="error")
        raise