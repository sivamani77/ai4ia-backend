"""FastAPI dependencies shared across routers.

require_models() is the gate: attach it to an endpoint and, if the model
host(s) it names aren't ready, the endpoint's body never runs — the
dependency raises before FastAPI calls the handler. See
app/services/model_gate.py for how readiness is actually decided (a real
probe, not warm-up bookkeeping) and app/services/video_job_runner.py for
the equivalent gate on the async video-job pipeline, which waits far
longer since nothing is holding an HTTP connection open there.
"""
from __future__ import annotations

from fastapi import HTTPException

from app.core.settings import settings
from app.services import model_gate


def require_models(*keys: str):
    """Dependency factory: `Depends(require_models(model_gate.EMBEDDING, model_gate.LLM))`.

    If every named target is already ready, this costs one cached Redis
    read per target and returns immediately. If any target is cold, it
    starts (or joins) the backoff warmer and blocks the request for up to
    settings.model_gate_inline_wait_seconds per still-cold target before
    giving up — short enough to stay under typical LB/proxy idle timeouts.
    On timeout it raises 503 with Retry-After so the frontend can show a
    "starting up" state and retry, rather than the request hanging until
    something else kills the connection.
    """

    def _dependency() -> None:
        if not settings.model_gate_enabled:
            return

        readiness = model_gate.ensure_ready(
            *keys, wait_seconds=settings.model_gate_inline_wait_seconds
        )
        not_ready = [key for key, ok in readiness.items() if not ok]
        if not not_ready:
            return

        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_warming",
                "message": (
                    "The AI models needed for this feature are starting up. "
                    "Please retry in a few seconds."
                ),
                "targets": not_ready,
                "states": {key: model_gate.state(key) for key in not_ready},
            },
            headers={"Retry-After": "20"},
        )

    return _dependency


__all__ = ["require_models"]
