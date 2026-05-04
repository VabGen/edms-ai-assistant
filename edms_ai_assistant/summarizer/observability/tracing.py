"""
Observability — OpenTelemetry spans + cost/latency tracking.

2025 Best Practices:
- Every pipeline stage wrapped in an OTel span
- Token cost computed and attached as span attributes
- Async-safe context propagation via contextvars
- Zero overhead when tracing disabled (no-op tracer)

Usage:
    async with trace_stage("map_reduce.map", attributes={"chunk_count": 5}) as span:
        result = await do_work()
        span.set_attribute("tokens_used", 1200)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, Span, StatusCode

# ---------------------------------------------------------------------------
# Cost Table (USD per 1K tokens, update as pricing changes)
# ---------------------------------------------------------------------------

_COST_TABLE: dict[str, tuple[float, float]] = {
    # model_name: (input_per_1k, output_per_1k)
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-sonnet-4": (0.003, 0.015),
    "ollama": (0.0, 0.0),  # Local inference
    "default": (0.002, 0.006),
}


def get_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate USD cost for a generation.

    Keys matched longest-first: prevents 'gpt-4o' substring-matching 'gpt-4o-mini'.
    """
    model_lower = model.lower()
    sorted_keys = sorted(
        (k for k in _COST_TABLE if k != "default"),
        key=len, reverse=True,
    )
    key = next((k for k in sorted_keys if k in model_lower), "default")
    inp_price, out_price = _COST_TABLE[key]
    return (input_tokens / 1000 * inp_price) + (output_tokens / 1000 * out_price)


# ---------------------------------------------------------------------------
# Request-scoped cost accumulator
# ---------------------------------------------------------------------------


@dataclass
class RequestCostAccumulator:
    """Accumulates token costs across all LLM calls in a single request."""

    request_id: str
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    call_count: int = 0
    total_latency_ms: float = 0.0
    stage_costs: list[dict] = field(default_factory=list)

    def record(
        self,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_latency_ms += latency_ms
        self.call_count += 1
        self.stage_costs.append({
            "stage": stage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": get_cost_usd(self.model, input_tokens, output_tokens),
            "latency_ms": round(latency_ms, 1),
        })

    @property
    def total_cost_usd(self) -> float:
        return get_cost_usd(self.model, self.total_input_tokens, self.total_output_tokens)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "stages": self.stage_costs,
        }


# ContextVar for request-scoped accumulator (async-safe)
_cost_ctx: ContextVar[RequestCostAccumulator | None] = ContextVar(
    "_cost_ctx", default=None
)


def get_current_accumulator() -> RequestCostAccumulator | None:
    return _cost_ctx.get()


def set_current_accumulator(acc: RequestCostAccumulator) -> None:
    _cost_ctx.set(acc)


# ---------------------------------------------------------------------------
# Tracer setup
# ---------------------------------------------------------------------------

_tracer: trace.Tracer | None = None
_in_memory_exporter: InMemorySpanExporter | None = None


def setup_tracing(
    service_name: str = "edms-summarizer",
    *,
    enable_in_memory: bool = False,
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing.

    Call once at application startup.

    Args:
        service_name: Service name embedded in all spans.
        enable_in_memory: Export spans to in-memory buffer (for testing).
        otlp_endpoint: OTLP gRPC endpoint (e.g. 'http://localhost:4317').
                       If None, uses in-memory or no-op exporter.
    """
    global _tracer, _in_memory_exporter

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # OTLP exporter not installed — fall through

    if enable_in_memory:
        _in_memory_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("edms-summarizer")
    return _tracer


def get_finished_spans() -> list:
    """Return all finished spans (only when enable_in_memory=True). Used in tests."""
    if _in_memory_exporter is None:
        return []
    return _in_memory_exporter.get_finished_spans()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Async context manager for pipeline stages
# ---------------------------------------------------------------------------


@asynccontextmanager
async def trace_stage(
    name: str,
    attributes: dict | None = None,
) -> AsyncIterator[Span]:
    """Async context manager that wraps a pipeline stage in an OTel span.

    Records duration automatically. Sets ERROR status on exception.

    Usage:
        async with trace_stage("map_reduce.reduce", {"chunk_count": n}) as span:
            result = await reduce(chunks)
            span.set_attribute("output_tokens", result.tokens)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(f"summarizer.{name}") as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        start_ms = time.monotonic() * 1000
        try:
            yield span
            duration_ms = time.monotonic() * 1000 - start_ms
            span.set_attribute("duration_ms", round(duration_ms, 1))
            span.set_status(StatusCode.OK)
        except Exception as exc:
            duration_ms = time.monotonic() * 1000 - start_ms
            span.set_attribute("duration_ms", round(duration_ms, 1))
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def record_llm_call(
    span: Span,
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """Record LLM call metrics on span and accumulator."""
    cost = get_cost_usd(model, input_tokens, output_tokens)

    if not isinstance(span, NonRecordingSpan):
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.cost_usd", round(cost, 6))
        span.set_attribute("llm.latency_ms", round(latency_ms, 1))

    acc = get_current_accumulator()
    if acc is not None:
        acc.record(stage, input_tokens, output_tokens, latency_ms)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


class Stopwatch:
    """Simple monotonic stopwatch for latency measurement."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    def reset(self) -> None:
        self._start = time.monotonic()