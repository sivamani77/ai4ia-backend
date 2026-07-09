"""Unit tests for Prometheus metrics instrumentation — verifies the counters/
histograms/gauges in app.core.metrics actually move when the instrumented
code paths run, using the same mocking patterns as test_video_job_runner.py
and the auth/job repository tests."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.core import metrics


def _counter_value(counter, **labels) -> float:
    child = counter.labels(**labels) if labels else counter
    return child._value.get()


def _gauge_value(gauge, **labels) -> float:
    child = gauge.labels(**labels) if labels else gauge
    return child._value.get()


def _histogram_count(hist, **labels) -> float:
    child = hist.labels(**labels) if labels else hist
    for sample in child.collect()[0].samples:
        if sample.name.endswith("_count"):
            return sample.value
    raise AssertionError("no _count sample found")


# ---------------------------------------------------------------------------
# Job pipeline
# ---------------------------------------------------------------------------

class TestJobPipelineMetrics:
    def test_create_video_job_increments_created_total(self):
        before = _counter_value(metrics.jobs_created_total)
        with patch("app.services.video_job_runner.save_job"), \
             patch("app.services.video_job_runner.append_event"):
            from app.services.video_job_runner import create_video_job
            create_video_job({"prompt": "x", "target_duration": 60}, user_id="u1")
        assert _counter_value(metrics.jobs_created_total) == before + 1

    def test_complete_job_increments_completions_and_duration(self):
        job = {
            "job_id": "j1",
            "created_at": "2026-01-01T00:00:00+00:00",
            "completed_at": None,
        }
        before_total = _counter_value(metrics.job_completions_total, status="completed")
        before_hist = _histogram_count(metrics.job_duration_seconds)
        with patch("app.services.video_job_runner.load_job", return_value=job), \
             patch("app.services.video_job_runner.save_job"):
            from app.services.video_job_runner import _complete_job
            _complete_job("j1")
        assert _counter_value(metrics.job_completions_total, status="completed") == before_total + 1
        assert _histogram_count(metrics.job_duration_seconds) == before_hist + 1

    def test_fail_job_increments_completions_failed(self):
        job = {"job_id": "j2"}
        before = _counter_value(metrics.job_completions_total, status="failed")
        with patch("app.services.video_job_runner.load_job", return_value=job), \
             patch("app.services.video_job_runner.save_job"), \
             patch("app.services.video_job_runner.append_event"):
            from app.services.video_job_runner import _fail_job
            _fail_job("j2", "08_assets", "pipeline_error", "boom")
        assert _counter_value(metrics.job_completions_total, status="failed") == before + 1

    def test_cancel_job_increments_completions_cancelled(self):
        job = {"job_id": "j3", "current_stage": "08_assets"}
        before = _counter_value(metrics.job_completions_total, status="cancelled")
        with patch("app.services.video_job_runner.load_job", return_value=job), \
             patch("app.services.video_job_runner.save_job"), \
             patch("app.services.video_job_runner.append_event"):
            from app.services.video_job_runner import _cancel_job
            _cancel_job("j3", "user requested")
        assert _counter_value(metrics.job_completions_total, status="cancelled") == before + 1

    def test_update_stage_records_transition_and_duration(self):
        job_id = f"jstage_{uuid.uuid4().hex[:8]}"
        stage = "08_assets"
        job = {"job_id": job_id, "stage_status": {}}
        before_transitions = _counter_value(
            metrics.job_stage_transitions_total, stage=stage, status="running"
        )
        before_duration = _histogram_count(metrics.job_stage_duration_seconds, stage=stage)

        with patch("app.services.video_job_runner.load_job", return_value=job), \
             patch("app.services.video_job_runner.save_job"), \
             patch("app.services.video_job_runner.append_event"):
            from app.services.video_job_runner import update_stage
            update_stage(job_id, stage, "running", 0)
            update_stage(job_id, stage, "completed", 100)

        assert _counter_value(
            metrics.job_stage_transitions_total, stage=stage, status="running"
        ) == before_transitions + 1
        assert _histogram_count(metrics.job_stage_duration_seconds, stage=stage) == before_duration + 1

    def test_redis_reachable_sets_gauge(self):
        from app.services import video_job_runner

        with patch("app.services.video_job_runner.settings") as mock_settings:
            mock_settings.redis_url = "redis://localhost:1"
            video_job_runner._redis_cache = (False, 0.0)
            video_job_runner._redis_reachable()

        assert _gauge_value(metrics.redis_reachable) in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuthMetrics:
    def test_login_rate_limited_increments_metrics(self):
        from app.services import auth_service

        client_key = f"client_{uuid.uuid4().hex}"
        mock_users = MagicMock()
        mock_users.get_by_username.return_value = None

        before_rate_limited = _counter_value(metrics.rate_limit_rejections_total, limiter="login")
        before_events = _counter_value(metrics.auth_events_total, event="login", outcome="rate_limited")

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()):
            for _ in range(5):
                with pytest.raises(Exception):
                    auth_service.login(username="nobody", password="x", client_key=client_key)
            with pytest.raises(Exception) as exc_info:
                auth_service.login(username="nobody", password="x", client_key=client_key)

        assert exc_info.value.status_code == 429
        assert _counter_value(metrics.rate_limit_rejections_total, limiter="login") == before_rate_limited + 1
        assert _counter_value(
            metrics.auth_events_total, event="login", outcome="rate_limited"
        ) == before_events + 1

    def test_login_unknown_user_increments_failure(self):
        from app.services import auth_service

        client_key = f"client_{uuid.uuid4().hex}"
        mock_users = MagicMock()
        mock_users.get_by_username.return_value = None

        before = _counter_value(metrics.auth_events_total, event="login", outcome="failure")

        with patch("app.services.auth_service.get_user_repository", return_value=mock_users), \
             patch("app.services.auth_service.get_auth_audit_repository", return_value=MagicMock()):
            with pytest.raises(Exception) as exc_info:
                auth_service.login(username="nobody", password="x", client_key=client_key)

        assert exc_info.value.status_code == 401
        assert _counter_value(metrics.auth_events_total, event="login", outcome="failure") == before + 1

    def test_resend_otp_rate_limited_increments_metrics(self):
        from app.services import auth_service

        email = f"{uuid.uuid4().hex}@example.com"

        before_rate_limited = _counter_value(metrics.rate_limit_rejections_total, limiter="otp_resend")

        with patch("app.services.auth_service.get_user_repository", return_value=MagicMock(get_by_email=lambda e: None)), \
             patch("app.services.auth_service.get_otp_repository", return_value=MagicMock()), \
             patch("app.services.auth_service.send_otp_email"), \
             patch.object(auth_service.settings, "otp_ttl_minutes", 10):
            auth_service.resend_otp(email=email, purpose="signup")  # consumes the 1 token
            with pytest.raises(Exception) as exc_info:
                auth_service.resend_otp(email=email, purpose="signup")

        assert exc_info.value.status_code == 429
        assert _counter_value(
            metrics.rate_limit_rejections_total, limiter="otp_resend"
        ) == before_rate_limited + 1
