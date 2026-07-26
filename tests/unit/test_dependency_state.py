"""Tests for the event-driven model-host health record.

The invariant these protect: recording and reading health must never issue a
request to a model host, and must never raise into the caller — the whole
point is that a scale-to-zero GPU is not probed.
"""
from __future__ import annotations

import time

import pytest

from app.core import dependency_state


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    dependency_state.reset_for_tests()
    # Default to "no Redis" so tests exercise the local-memory path unless
    # they opt into a fake.
    monkeypatch.setattr(dependency_state, "_get_redis", lambda: None)
    yield
    dependency_state.reset_for_tests()


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def setex(self, key, ttl, value):
        self.store[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.store.get(key)

    def mget(self, keys):
        return [self.store.get(k) for k in keys]

    def scan_iter(self, match=None, count=None):
        prefix = (match or "").rstrip("*")
        return [k.encode() for k in self.store if k.startswith(prefix)]


def test_host_from_url_extracts_netloc():
    assert dependency_state.host_from_url(
        "http://vllm-gpt-120.mobiusdtaas.ai/v1/chat/completions"
    ) == "vllm-gpt-120.mobiusdtaas.ai"


def test_record_then_snapshot_reports_up():
    dependency_state.record("vllm-gpt-120", True, latency_s=1.25)

    snap = dependency_state.snapshot()

    assert snap["vllm-gpt-120"]["status"] == "up"
    assert snap["vllm-gpt-120"]["last_latency_seconds"] == 1.25


def test_failure_is_reported_down_with_error():
    dependency_state.record("ollama-host", False, latency_s=0.5, error="timeout")

    entry = dependency_state.snapshot()["ollama-host"]

    assert entry["status"] == "down"
    assert entry["error"] == "timeout"


def test_stale_record_becomes_unknown_not_down():
    """An idle scale-to-zero host must not look like an outage."""
    dependency_state.record("vllm-gpt-120", True)
    # Backdate the entry past the staleness window.
    dependency_state._local["vllm-gpt-120"]["at"] = time.time() - 10_000

    assert dependency_state.snapshot()["vllm-gpt-120"]["status"] == "unknown"


def test_never_called_host_is_absent():
    assert dependency_state.snapshot() == {}


def test_record_survives_redis_failure(monkeypatch):
    class _Broken:
        def setex(self, *a, **kw):
            raise RuntimeError("redis down")

    monkeypatch.setattr(dependency_state, "_get_redis", lambda: _Broken())

    # Must not raise — this runs attached to an in-flight LLM call.
    dependency_state.record("vllm-gpt-120", True, latency_s=0.1)

    assert dependency_state.snapshot()["vllm-gpt-120"]["status"] == "up"


def test_redis_state_is_visible_across_processes(monkeypatch):
    """The API pod answering /api/health is not the process that made the
    call, so the record has to survive the process boundary."""
    shared = _FakeRedis()
    monkeypatch.setattr(dependency_state, "_get_redis", lambda: shared)

    dependency_state.record("vllm-gpt-120", True, latency_s=2.0)

    # Simulate a different process: local memory empty, same Redis.
    dependency_state._local.clear()

    assert dependency_state.snapshot()["vllm-gpt-120"]["status"] == "up"


def test_redis_wins_over_stale_local_copy(monkeypatch):
    shared = _FakeRedis()
    monkeypatch.setattr(dependency_state, "_get_redis", lambda: shared)

    dependency_state.record("vllm-gpt-120", False, error="boom")
    # Another process recorded a success afterwards.
    shared.setex(
        "dep_health:vllm-gpt-120",
        900,
        '{"ok": true, "at": %f, "latency_s": 1.0, "error": null}' % time.time(),
    )

    assert dependency_state.snapshot()["vllm-gpt-120"]["status"] == "up"
