"""Model readiness gate for the two scale-to-zero GPU hosts the backend
still depends on: Ollama (nomic-embed-text:v1.5) and the 120b vLLM host
(STORYLINE_VLLM_BASE_URL / STORYLINE_VLLM_MODEL — insights, relay, theme
synthesis, speaker comparison, storyline generation). The 20b vLLM
deployment is out of scope; nothing here targets it.

Readiness is decided by probing the hosts directly — NOT by trusting
app/core/dependency_state.py, which deliberately only reports what a real
LLM call happened to observe. A feature request needs a true, current
answer ("is nomic-embed-text loaded right now"), not "nobody called it in
the last 15 minutes". The trade-off this module accepts: it does send
network requests to the model hosts. The invariant that keeps that safe is
that a probe only ever fires from a real login or a real feature request —
never from a timer, a health check, or a liveness probe. If that invariant
is ever broken, this recreates the always-on-GPU problem dependency_state
was built to avoid.

Two Redis-backed pieces of state per target:
  * model_probe:{key}  — short-TTL cache of the last real probe outcome.
    Pure stampede guard (N concurrent requests share one probe), not a
    substitute for probing; see settings.model_probe_cache_seconds.
  * model_state:{key}  — the human-readable status variable ("cold" |
    "warming" | "ready" | "error"), written by the backoff warmer. This is
    reporting only. Nothing gates on it — see is_ready().

Every public function is best-effort against Redis: a Redis outage
degrades to a process-local mirror (this process's own view only), never
raises into a caller, and never blocks a caller longer than the wait it
explicitly asked for.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any

import requests

from app.core import dependency_state
from app.core.config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    STORYLINE_VLLM_BASE_URL,
    STORYLINE_VLLM_MODEL,
)
from app.core.settings import settings
from app.services import session_service

logger = logging.getLogger("ai4ia.services.model_gate")

EMBEDDING = "embedding"
LLM = "llm"
ALL_TARGETS = (EMBEDDING, LLM)

_PROBE_KEY_PREFIX = "model_probe:"
_STATE_KEY_PREFIX = "model_state:"
_LOCK_KEY_PREFIX = "model_warm_lock:"

# Process-local fallback, used only when Redis is unreachable.
_local_probe: dict[str, tuple[bool, float]] = {}
_local_state: dict[str, dict[str, Any]] = {}
_local_lock = threading.Lock()

_redis_client = None
_redis_failed_at: float = 0.0
_REDIS_RETRY_SECONDS = 30.0

# Exponential backoff schedule for the warmer, seconds between attempts.
_BACKOFF_SCHEDULE = (0, 2, 4, 8, 16, 32, 60)


def _get_redis():
    global _redis_client, _redis_failed_at
    if _redis_client is not None:
        return _redis_client
    if time.monotonic() - _redis_failed_at < _REDIS_RETRY_SECONDS:
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url, socket_connect_timeout=0.5, socket_timeout=2.0
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:  # noqa: BLE001
        _redis_failed_at = time.monotonic()
        logger.debug("model_gate: Redis unavailable, using process-local state only")
        return None


# ---------------------------------------------------------------------------
# Real probes — the only functions in this module allowed to talk to a
# model host. Each returns (ok, error_message).
# ---------------------------------------------------------------------------

def _probe_embedding(*, timeout: float = 5) -> tuple[bool, str | None]:
    """A successful /api/pull only means the blob is on disk — the model
    still has to load off disk into memory on the FIRST real inference
    call, which is what this does. On a scale-to-zero host that just woke
    up, that cold load can take much longer than the fast timeout used for
    the interactive-request readiness check (is_ready()), so callers that
    are actively warming (as opposed to just checking) pass a longer
    timeout — see _warm_once."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": "ping",
                "keep_alive": settings.ollama_keep_alive,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" in data or data.get("embeddings"):
            return True, None
        return False, f"unexpected response keys: {list(data.keys())}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def _pull_embedding_model() -> tuple[bool, str | None]:
    """/api/pull only puts the blob on disk — it does not make the model
    resident. Warming always follows this with a real embed (_probe_embedding)
    to force the load."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/pull",
            json={"model": EMBEDDING_MODEL, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def _probe_llm() -> tuple[bool, str | None]:
    try:
        resp = requests.get(
            f"{STORYLINE_VLLM_BASE_URL.rstrip('/')}/v1/models", timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        model_ids = {m.get("id") for m in data.get("data", [])}
        if STORYLINE_VLLM_MODEL in model_ids:
            return True, None
        return False, f"model {STORYLINE_VLLM_MODEL!r} not in {model_ids}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


_PROBES = {EMBEDDING: _probe_embedding, LLM: _probe_llm}


def _call_probe(key: str) -> tuple[bool, str | None]:
    """Every call site goes through this rather than _PROBES[key]() directly
    — guarantees the "never raises" contract even if a probe implementation
    (or a test double) doesn't catch its own exceptions."""
    try:
        return _PROBES[key]()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]


def _host_for(key: str) -> str:
    return dependency_state.host_from_url(
        OLLAMA_BASE_URL if key == EMBEDDING else STORYLINE_VLLM_BASE_URL
    )


# ---------------------------------------------------------------------------
# Probe cache
# ---------------------------------------------------------------------------

def _cached_probe(key: str) -> bool | None:
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(_PROBE_KEY_PREFIX + key)
            if raw is not None:
                return raw in (b"1", "1")
        except Exception:  # noqa: BLE001
            pass
    with _local_lock:
        entry = _local_probe.get(key)
    if entry is not None:
        ok, at = entry
        if time.monotonic() - at < settings.model_probe_cache_seconds:
            return ok
    return None


def _store_probe(key: str, ok: bool) -> None:
    with _local_lock:
        _local_probe[key] = (ok, time.monotonic())
    client = _get_redis()
    if client is not None:
        try:
            client.setex(_PROBE_KEY_PREFIX + key, settings.model_probe_cache_seconds, "1" if ok else "0")
        except Exception:  # noqa: BLE001
            pass


def is_ready(key: str, *, use_cache: bool = True) -> bool:
    """The only readiness question this module answers: is the host
    reachable and serving the expected model, right now (modulo a short
    stampede-guard cache). Never trusts warm-up bookkeeping."""
    if use_cache:
        cached = _cached_probe(key)
        if cached is not None:
            return cached

    t0 = time.time()
    ok, error = _call_probe(key)
    elapsed = time.time() - t0
    _store_probe(key, ok)
    dependency_state.record(_host_for(key), ok, latency_s=elapsed, error=error)
    return ok


# ---------------------------------------------------------------------------
# Warm-up state (status variable only — never gates anything)
# ---------------------------------------------------------------------------

def _write_state(key: str, **fields: Any) -> None:
    with _local_lock:
        entry = _local_state.setdefault(key, {})
        entry.update(fields)
        entry["at"] = time.time()
        snapshot = dict(entry)
    client = _get_redis()
    if client is not None:
        try:
            import json

            client.set(_STATE_KEY_PREFIX + key, json.dumps(snapshot))
        except Exception:  # noqa: BLE001
            pass


def state(key: str) -> dict[str, Any]:
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(_STATE_KEY_PREFIX + key)
            if raw is not None:
                import json

                return json.loads(raw)
        except Exception:  # noqa: BLE001
            pass
    with _local_lock:
        return dict(_local_state.get(key, {"state": "unknown"}))


def all_states() -> dict[str, dict[str, Any]]:
    return {key: state(key) for key in ALL_TARGETS}


# ---------------------------------------------------------------------------
# Backoff warmer
# ---------------------------------------------------------------------------

def _acquire_warm_lock(key: str) -> bool:
    client = _get_redis()
    if client is None:
        # No Redis — best-effort single-process guard only.
        with _local_lock:
            entry = _local_state.get(key, {})
            if entry.get("_locked_until", 0) > time.monotonic():
                return False
            entry = _local_state.setdefault(key, {})
            entry["_locked_until"] = time.monotonic() + settings.model_warm_lock_seconds
        return True
    try:
        return bool(client.set(_LOCK_KEY_PREFIX + key, "1", nx=True, ex=settings.model_warm_lock_seconds))
    except Exception:  # noqa: BLE001
        return True  # fail open on lock acquisition — a duplicate warm is wasteful, not unsafe


def _release_warm_lock(key: str) -> None:
    client = _get_redis()
    if client is not None:
        try:
            client.delete(_LOCK_KEY_PREFIX + key)
        except Exception:  # noqa: BLE001
            pass


def _warm_once(key: str) -> tuple[bool, str | None]:
    if key == EMBEDDING:
        pulled, pull_error = _pull_embedding_model()
        if not pulled:
            return False, pull_error
        # Longer timeout than is_ready()'s fast check: this call is what
        # forces the cold load off disk, which routinely takes longer than
        # the 5s used for a cheap readiness probe. _probe_embedding already
        # catches its own exceptions, so this is safe to call directly.
        return _probe_embedding(timeout=settings.model_embedding_warm_timeout_seconds)
    return _call_probe(key)


def _run_backoff_warmer(key: str) -> None:
    _write_state(key, state="warming", started_at=time.time())
    deadline = time.monotonic() + settings.model_warm_deadline_seconds
    attempt = 0
    last_error: str | None = None
    try:
        while time.monotonic() < deadline:
            ok, error = _warm_once(key)
            _store_probe(key, ok)
            dependency_state.record(_host_for(key), ok, error=error)
            if ok:
                _write_state(key, state="ready", ready_at=time.time(), error=None)
                return
            last_error = error
            delay = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
            delay *= 1 + random.uniform(-0.2, 0.2)
            attempt += 1
            time.sleep(max(delay, 0))
        _write_state(key, state="error", error=last_error or "warm-up deadline exceeded")
    finally:
        _release_warm_lock(key)


def warm_async(key: str) -> bool:
    """Start the backoff warmer for `key` in a background thread if one
    isn't already running. Returns True if this call started a warmer,
    False if another warm attempt (this process or another pod) is already
    in flight. Never raises, never blocks."""
    if not _acquire_warm_lock(key):
        return False
    _write_state(key, state="warming")
    threading.Thread(target=_run_backoff_warmer, args=(key,), daemon=True).start()
    return True


def wait_until_ready(key: str, wait_seconds: float, *, poll_interval: float = 1.0) -> bool:
    """Poll is_ready() for up to `wait_seconds`. Used by the request-time
    gate (inline wait) and the video pipeline (long wait).

    Polls bypass the probe cache: a caller in this loop has already decided
    to wait, so the stampede guard (which protects against many concurrent
    *first* checks) doesn't apply, and honoring it here would mean reporting
    "not ready" for up to model_probe_cache_seconds after the warmer already
    succeeded.
    """
    deadline = time.monotonic() + wait_seconds
    if is_ready(key, use_cache=False):
        return True
    while time.monotonic() < deadline:
        time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0)))
        if is_ready(key, use_cache=False):
            return True
    return False


def ensure_ready(*keys: str, wait_seconds: float = 0) -> dict[str, bool]:
    """For each key: if not ready, kick off (or join) the backoff warmer and
    wait up to `wait_seconds`. Returns {key: ready_bool}. Callers gate
    business logic on this dict, never on state()."""
    result: dict[str, bool] = {}
    for key in keys:
        if is_ready(key):
            result[key] = True
            continue
        warm_async(key)
        result[key] = wait_until_ready(key, wait_seconds) if wait_seconds > 0 else False
    return result


# ---------------------------------------------------------------------------
# Login trigger
# ---------------------------------------------------------------------------

def trigger_warmup(session_id: str) -> None:
    """Fire-and-forget: called right after login/signup issues tokens.
    Starts warming both targets (deduped across concurrent logins via the
    Redis lock) and, independently, waits for both to settle so the
    session row's warmup_state can be updated — that's the status variable
    the plan asked for, and it never gates a request."""
    if not settings.model_gate_enabled or not settings.model_warmup_on_login:
        return

    for key in ALL_TARGETS:
        warm_async(key)

    def _watch() -> None:
        session_service.mark_warmup_started(session_id)
        deadline = time.monotonic() + settings.model_warm_deadline_seconds
        while time.monotonic() < deadline:
            if all(is_ready(k, use_cache=False) for k in ALL_TARGETS):
                session_service.mark_warmup_ready(session_id)
                return
            time.sleep(2)
        errors = ", ".join(
            f"{k}: {state(k).get('error')}" for k in ALL_TARGETS if not is_ready(k, use_cache=False)
        )
        session_service.mark_warmup_error(session_id, errors or "warm-up deadline exceeded")

    threading.Thread(target=_watch, daemon=True).start()


def reset_for_tests() -> None:
    global _redis_client, _redis_failed_at
    with _local_lock:
        _local_probe.clear()
        _local_state.clear()
    _redis_client = None
    _redis_failed_at = 0.0


__all__ = [
    "EMBEDDING",
    "LLM",
    "ALL_TARGETS",
    "is_ready",
    "warm_async",
    "wait_until_ready",
    "ensure_ready",
    "state",
    "all_states",
    "trigger_warmup",
    "reset_for_tests",
]
