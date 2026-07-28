"""Unit tests for app.services.model_gate.

Redis is forced unavailable (via monkeypatching _get_redis to return None)
so these run purely against the process-local fallback path — the same
code path a Redis outage degrades to in production, per the module's
"never raise into a caller" contract.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.services import model_gate


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    model_gate.reset_for_tests()
    monkeypatch.setattr(model_gate, "_get_redis", lambda: None)
    monkeypatch.setattr(model_gate.dependency_state, "record", lambda *a, **k: None)
    yield
    model_gate.reset_for_tests()


class TestIsReady:
    def test_probes_and_caches_result(self, monkeypatch):
        calls = {"n": 0}

        def _probe():
            calls["n"] += 1
            return True, None

        monkeypatch.setitem(model_gate._PROBES, model_gate.EMBEDDING, _probe)

        assert model_gate.is_ready(model_gate.EMBEDDING) is True
        assert model_gate.is_ready(model_gate.EMBEDDING) is True
        # Second call hit the cache, not the probe again.
        assert calls["n"] == 1

    def test_bypasses_cache_when_requested(self, monkeypatch):
        calls = {"n": 0}

        def _probe():
            calls["n"] += 1
            return True, None

        monkeypatch.setitem(model_gate._PROBES, model_gate.EMBEDDING, _probe)

        model_gate.is_ready(model_gate.EMBEDDING)
        model_gate.is_ready(model_gate.EMBEDDING, use_cache=False)
        assert calls["n"] == 2

    def test_failed_probe_is_not_ready(self, monkeypatch):
        monkeypatch.setitem(
            model_gate._PROBES, model_gate.LLM, lambda: (False, "connection refused")
        )
        assert model_gate.is_ready(model_gate.LLM) is False


class TestWarmLock:
    def test_second_warm_attempt_is_deduped(self, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def _slow_warmer(key):
            started.set()
            release.wait(timeout=2)

        monkeypatch.setattr(model_gate, "_run_backoff_warmer", _slow_warmer)

        assert model_gate.warm_async(model_gate.EMBEDDING) is True
        started.wait(timeout=2)
        # A second warm attempt for the same key while one is in flight must
        # not start another thread — the lock is held.
        assert model_gate.warm_async(model_gate.EMBEDDING) is False
        release.set()

    def test_lock_is_per_key(self, monkeypatch):
        monkeypatch.setattr(model_gate, "_run_backoff_warmer", lambda key: time.sleep(0.05))
        assert model_gate.warm_async(model_gate.EMBEDDING) is True
        assert model_gate.warm_async(model_gate.LLM) is True


class TestBackoffWarmer:
    def test_succeeds_on_first_attempt_updates_state_ready(self, monkeypatch):
        monkeypatch.setattr(model_gate.settings, "model_warm_deadline_seconds", 5)
        monkeypatch.setitem(model_gate._PROBES, model_gate.EMBEDDING, lambda: (True, None))
        monkeypatch.setattr(model_gate, "_pull_embedding_model", lambda: (True, None))

        model_gate._run_backoff_warmer(model_gate.EMBEDDING)

        state = model_gate.state(model_gate.EMBEDDING)
        assert state["state"] == "ready"
        assert model_gate.is_ready(model_gate.EMBEDDING) is True

    def test_gives_up_at_deadline_and_records_error(self, monkeypatch):
        monkeypatch.setattr(model_gate.settings, "model_warm_deadline_seconds", 0.05)
        monkeypatch.setattr(model_gate, "_BACKOFF_SCHEDULE", (0,))
        monkeypatch.setitem(
            model_gate._PROBES, model_gate.LLM, lambda: (False, "still cold")
        )

        model_gate._run_backoff_warmer(model_gate.LLM)

        state = model_gate.state(model_gate.LLM)
        assert state["state"] == "error"
        assert "still cold" in state["error"]

    def test_probe_exception_is_treated_as_failure_not_raised(self, monkeypatch):
        monkeypatch.setattr(model_gate.settings, "model_warm_deadline_seconds", 0.05)
        monkeypatch.setattr(model_gate, "_BACKOFF_SCHEDULE", (0,))

        def _boom():
            raise RuntimeError("network exploded")

        monkeypatch.setitem(model_gate._PROBES, model_gate.LLM, _boom)

        # Must not raise: _call_probe() catches any probe exception and
        # treats it as a failed attempt, so a misbehaving probe can only
        # ever delay readiness, never crash the warmer thread.
        model_gate._run_backoff_warmer(model_gate.LLM)

        state = model_gate.state(model_gate.LLM)
        assert state["state"] == "error"
        assert "network exploded" in state["error"]


class TestEnsureReady:
    def test_returns_true_immediately_when_already_ready(self, monkeypatch):
        monkeypatch.setitem(model_gate._PROBES, model_gate.EMBEDDING, lambda: (True, None))
        result = model_gate.ensure_ready(model_gate.EMBEDDING, wait_seconds=0)
        assert result == {model_gate.EMBEDDING: True}

    def test_returns_false_without_waiting_when_cold_and_no_wait(self, monkeypatch):
        monkeypatch.setattr(model_gate, "_run_backoff_warmer", lambda key: None)
        monkeypatch.setitem(model_gate._PROBES, model_gate.LLM, lambda: (False, "cold"))
        result = model_gate.ensure_ready(model_gate.LLM, wait_seconds=0)
        assert result == {model_gate.LLM: False}

    def test_waits_and_reports_ready_once_warmer_finishes(self, monkeypatch):
        state = {"ready": False}

        def _fake_warmer(key):
            time.sleep(0.1)
            state["ready"] = True

        monkeypatch.setattr(model_gate, "_run_backoff_warmer", lambda key: threading.Thread(
            target=_fake_warmer, args=(key,), daemon=True
        ).start())
        monkeypatch.setitem(
            model_gate._PROBES, model_gate.EMBEDDING, lambda: (state["ready"], None)
        )

        result = model_gate.ensure_ready(
            model_gate.EMBEDDING, wait_seconds=2
        )
        assert result == {model_gate.EMBEDDING: True}


class TestTriggerWarmup:
    def test_noop_when_gate_disabled(self, monkeypatch):
        monkeypatch.setattr(model_gate.settings, "model_gate_enabled", False)
        calls = []
        monkeypatch.setattr(model_gate, "warm_async", lambda key: calls.append(key))
        model_gate.trigger_warmup("session-1")
        assert calls == []

    def test_starts_warmers_for_all_targets(self, monkeypatch):
        monkeypatch.setattr(model_gate.settings, "model_gate_enabled", True)
        monkeypatch.setattr(model_gate.settings, "model_warmup_on_login", True)
        calls = []
        monkeypatch.setattr(model_gate, "warm_async", lambda key: calls.append(key))
        monkeypatch.setattr(model_gate, "is_ready", lambda key, **k: True)
        monkeypatch.setattr(model_gate.session_service, "mark_warmup_started", lambda sid: None)
        monkeypatch.setattr(model_gate.session_service, "mark_warmup_ready", lambda sid: None)

        model_gate.trigger_warmup("session-1")
        time.sleep(0.1)

        assert set(calls) == set(model_gate.ALL_TARGETS)
