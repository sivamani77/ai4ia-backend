"""
OpenTelemetry tracing setup, shared by the API process (app.main) and the
Celery worker (app.worker.celery_app).

Enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set (pointing at an OTel
Collector — see k8s/01-configmap.yaml). Without it, setup_tracing() is a
no-op so local dev / tests / CI never need a collector running.
"""
from __future__ import annotations

import os

_configured = False


def setup_tracing(service_name: str) -> None:
    """Configure the global tracer provider and inject trace_id/span_id into
    log records. Safe to call multiple times; only the first call (per
    process) takes effect."""
    global _configured
    if _configured:
        return
    _configured = True

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    # set_logging_format=False: we own the log format strings (see
    # app.main's _PipelineLogFilter) and just want otelTraceID/otelSpanID
    # attached to each LogRecord for them to reference.
    LoggingInstrumentor().instrument(set_logging_format=False)


def get_tracer(name: str):
    from opentelemetry import trace
    return trace.get_tracer(name)
