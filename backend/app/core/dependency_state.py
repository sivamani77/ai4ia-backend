"""Event-driven health record for the model-serving hosts (vLLM / Ollama).

Nothing in this module ever initiates a request to a model host. Callers
that already talk to one report what happened; /api/health reads that
record instead of probing.

That inversion is the whole point. The model hosts are scale-to-zero GPU
deployments, so a liveness probe against them is not free — it is the thing
that keeps the GPU resident. A 4-second health check every 10 seconds is
enough to make the idle window unreachable and pin an H100 up permanently.
Health therefore has to be a byproduct of real work, never a reason to
generate traffic.

State is shared through Redis because LLM calls come from three separate
process groups:

  * API pods      — serve /api/health, make almost no LLM calls
  * Celery worker — long-lived, calls llm_service
  * scripts/07*   — short-lived subprocesses, exit when a job ends, and
                    account for most of the 120b traffic

A module-global would be invisible across all three, and would die with the
subprocess that recorded it. Redis is already a hard dependency reachable
from every one of them, and one SETEX against a call that takes seconds to
minutes is free.

Redis being down is not an error here: recording degrades to process-local
memory and snapshot() reports what that process happens to know. This
module must never be able to fail a pipeline run.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

from app.core import settings as _settings_module

logger = logging.getLogger("ai4ia.core.dependency_state")

_KEY_PREFIX = "dep_health:"

# Process-local mirror. Always written, read only as a fallback when Redis
# is unreachable — see snapshot().
_local: dict[str, dict[str, Any]] = {}
_local_lock = threading.Lock()

_redis_client = None
_redis_failed_at: float = 0.0
_REDIS_RETRY_SECONDS = 30.0


def _get_redis():
    """Lazily connect to Redis; back off for a while after a failure.

    Mirrors the pattern in services/storyline_cache.py — short socket
    timeouts so a wedged Redis can never add meaningful latency to the LLM
    call that triggered the record.
    """
    global _redis_client, _redis_failed_at
    if _redis_client is not None:
        return _redis_client
    if time.monotonic() - _redis_failed_at < _REDIS_RETRY_SECONDS:
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            _settings_module.settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=2.0,
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:  # noqa: BLE001
        _redis_failed_at = time.monotonic()
        logger.debug("dependency_state: Redis unavailable, recording locally only")
        return None


def host_from_url(url: str) -> str:
    """Network location of `url`, used as the record key.

    Keying on netloc rather than a caller-supplied label means every model
    host is tracked automatically — the 120b, the 20b and Ollama — with no
    per-call-site knowledge of which is which.
    """
    try:
        return urlparse(url).netloc or url
    except Exception:  # noqa: BLE001
        return url


def record(
    host: str,
    ok: bool,
    latency_s: float | None = None,
    error: str | None = None,
) -> None:
    """Record the outcome of a real call to `host`. Never raises."""
    entry = {
        "ok": bool(ok),
        "at": time.time(),
        "latency_s": round(latency_s, 3) if latency_s is not None else None,
        "error": (error or "")[:200] or None,
    }

    with _local_lock:
        _local[host] = entry

    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(
            _KEY_PREFIX + host,
            _stale_after_seconds(),
            json.dumps(entry),
        )
    except Exception:  # noqa: BLE001
        # A failed write must not surface to the caller: this is telemetry
        # attached to an in-flight LLM call, not the call itself.
        logger.debug("dependency_state: failed to publish %s", host, exc_info=True)


def snapshot(stale_after_s: float | None = None) -> dict[str, dict[str, Any]]:
    """Last known state of every model host that has been called. No network
    calls to the hosts themselves.

    A host reports "unknown" when nobody has called it inside the staleness
    window. For a scale-to-zero GPU with no jobs running that is the correct
    answer, not an outage — see the note in /api/health.
    """
    window = _stale_after_seconds() if stale_after_s is None else stale_after_s
    now = time.time()

    with _local_lock:
        merged: dict[str, dict[str, Any]] = dict(_local)

    client = _get_redis()
    if client is not None:
        try:
            keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in client.scan_iter(match=_KEY_PREFIX + "*", count=100)
            ]
            # One MGET rather than a GET per host: /api/health is hit by
            # every probe on every pod, so this runs on a hot path.
            for key, raw in zip(keys, client.mget(keys) if keys else []):
                if not raw:
                    continue
                # Redis wins: it sees every process, the local mirror only
                # sees this one.
                merged[key[len(_KEY_PREFIX):]] = json.loads(raw)
        except Exception:  # noqa: BLE001
            logger.debug("dependency_state: snapshot read failed", exc_info=True)

    out: dict[str, dict[str, Any]] = {}
    for host, entry in merged.items():
        age = now - float(entry.get("at", 0.0))
        if age > window:
            status = "unknown"
        else:
            status = "up" if entry.get("ok") else "down"
        result: dict[str, Any] = {
            "status": status,
            "age_seconds": round(age, 1),
        }
        if entry.get("latency_s") is not None:
            result["last_latency_seconds"] = entry["latency_s"]
        if status == "down" and entry.get("error"):
            result["error"] = entry["error"]
        out[host] = result
    return out


def _stale_after_seconds() -> int:
    return int(
        getattr(_settings_module.settings, "llm_health_stale_after_seconds", 900)
    )


def reset_for_tests() -> None:
    """Drop process-local state. Test helper only."""
    global _redis_client, _redis_failed_at
    with _local_lock:
        _local.clear()
    _redis_client = None
    _redis_failed_at = 0.0
