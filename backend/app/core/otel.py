"""
OpenTelemetry tracing setup, shared by the API process (app.main) and the
Celery worker (app.worker.celery_app).

Exports to the org's existing Jaeger collector (istio-system namespace) via
the Thrift-over-HTTP collector endpoint — same protocol/package versions
already used by other services here, pinned in requirements.txt.

Enabled only when JAEGER_URL is set. Without it, setup_tracing() is a no-op,
so local dev / tests / CI never need Jaeger running.

NOTE: the gate is JAEGER_URL, not OTEL_EXPORTER_OTLP_ENDPOINT. The latter is
set in .env but is read by nothing in this codebase, so setting it alone does
NOT enable tracing. The only place JAEGER_URL was ever set was the Kubernetes
configmap now parked in examples/local-dev/k8s/01-configmap.yaml, which does
not deploy — meaning tracing is currently off in the deployed environment.
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

    jaeger_url = os.environ.get("JAEGER_URL")
    if not jaeger_url:
        return
    jaeger_port = int(os.environ.get("JAEGER_PORT", "14268"))

    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(JaegerExporter(
        collector_endpoint=f"http://{jaeger_url}:{jaeger_port}/api/traces",
    )))
    trace.set_tracer_provider(provider)

    # set_logging_format=False: we own the log format strings (see
    # app.main's _PipelineLogFilter) and just want otelTraceID/otelSpanID
    # attached to each LogRecord for them to reference.
    LoggingInstrumentor().instrument(set_logging_format=False)


def is_enabled() -> bool:
    return bool(os.environ.get("JAEGER_URL"))


def get_tracer(name: str):
    from opentelemetry import trace
    return trace.get_tracer(name)
