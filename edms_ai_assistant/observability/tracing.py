"""OpenTelemetry Tracing initialization."""
from __future__ import annotations

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from edms_ai_assistant.config import settings

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str, otlp_endpoint: str | None = None) -> None:
    """Инициализирует глобальный TracerProvider."""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": settings.APP_VERSION,
    })

    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP Exporter configured to %s", otlp_endpoint)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed. Traces will not be exported."
            )

    if settings.DEBUG:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console Span Exporter enabled (DEBUG mode).")

    trace.set_tracer_provider(provider)