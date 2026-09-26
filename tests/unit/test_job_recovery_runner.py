"""How the pipeline reacts to lease stop signals (docs/design/job-recovery-plan.md §5)."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.pipeline import base
from app.services import job_lease


@pytest.fixture(autouse=True)
def _clean_leases(monkeypatch):
    monkeypatch.setattr(job_lease.settings, "job_heartbeat_interval_seconds", 60, raising=False)
    job_lease._leases.clear()
    yield
    for job_id in list(job_lease._leases):
        job_lease.release(job_id)


def _store_patches():
    return (
        patch("app.services.job_store.load_job", side_effect=lambda _id: {"job_id": _id, "status": "running"}),
        patch("app.services.job_store.save_job", lambda *a, **k: None),
        patch("app.services.job_store.append_event", lambda *a, **k: None),
    )


@pytest.mark.parametrize("reason, exc", [
    (job_lease.STOP_SUPERSEDED, base.JobSupersededException),
    (job_lease.STOP_DEADLINE, base.JobDeadlineExceeded),
    (job_lease.STOP_CANCELLED, base.JobCancelledException),
])
def test_run_script_kills_the_script_on_a_stop_signal(tmp_path, reason, exc):
    job_lease.start("stop_job", "me")
    job_lease._leases["stop_job"].stop_reason = reason
    p1, p2, p3 = _store_patches()
    with patch("app.pipeline.base.PROJECT_ROOT", tmp_path), p1, p2, p3:
        start = time.monotonic()
        with pytest.raises(exc):
            base.run_script("stop_job", "09_export", ["sleep", "60"])
    assert time.monotonic() - start < 15


def _run_pipeline_stopping_with(reason, mock_job_store):
    """Run run_video_job with one stage that stops with `reason`: a stop
    reason string, or an exception class raised straight from the stage."""
    from app.services import video_job_runner as vjr

    mock_job_store["jobs"]["vjob_s"] = {
        "job_id": "vjob_s", "status": "running", "dispatch_id": "d1", "lease_owner": "me",
        "user_id": "u1", "paths": {}, "request": {},
    }
    job_lease.start("vjob_s", "me")

    class _Stage:
        __name__ = "stage_fake"

        @staticmethod
        def run(job_id, request, ctx):
            if isinstance(reason, type):
                raise reason() if reason.__name__ == "SoftTimeLimitExceeded" else reason(job_id, "stage_fake")
            job_lease._leases[job_id].stop_reason = reason
            job_lease.raise_if_stopped(job_id, "stage_fake")

    with patch.object(vjr.model_gate, "ensure_ready", return_value={"embedding": True, "llm": True}), \
         patch("app.pipeline.stage_guardrail", _Stage, create=True), \
         patch.object(vjr, "update_stage", lambda *a, **k: None), \
         patch.object(vjr, "_cleanup_job_tmp_dirs", lambda *a, **k: None), \
         patch.object(vjr.processing_records, "finish", lambda *a, **k: None):
        vjr.run_video_job("vjob_s", {"prompt": "p", "target_duration": 60, "force_refresh": True})
    return mock_job_store["jobs"]["vjob_s"]


def test_superseded_run_records_nothing(mock_job_store):
    job = _run_pipeline_stopping_with(job_lease.STOP_SUPERSEDED, mock_job_store)
    assert job["status"] == "running"  # left for the sweeper or the new owner
    assert not any(e["event_type"] == "failed" for e in mock_job_store["events"].get("vjob_s", []))


def test_runtime_cap_fails_the_job_with_a_clear_message(mock_job_store):
    job = _run_pipeline_stopping_with(job_lease.STOP_DEADLINE, mock_job_store)
    assert job["status"] == "failed"
    assert job["error"]["type"] == "deadline_exceeded"
    assert "too long" in job["error"]["user_message"]


def test_completion_is_skipped_when_the_job_is_no_longer_this_runs(mock_job_store):
    from app.services import video_job_runner as vjr

    mock_job_store["jobs"]["vjob_c"] = {"job_id": "vjob_c", "status": "failed", "dispatch_id": "d1",
                                        "lease_owner": None, "created_at": None}
    job_lease.start("vjob_c", "me")
    with patch.object(vjr.processing_records, "finish") as finish:
        assert vjr._complete_job("vjob_c") is False
    finish.assert_not_called()
    assert job_lease.stop_reason("vjob_c") is None  # heartbeat stopped before the write


def test_failure_after_losing_the_lease_records_nothing(mock_job_store):
    from app.services import video_job_runner as vjr

    mock_job_store["jobs"]["vjob_f"] = {"job_id": "vjob_f", "status": "running", "dispatch_id": "d1",
                                        "lease_owner": "someone-else"}
    job_lease.start("vjob_f", "me")
    with patch.object(vjr.processing_records, "finish") as finish:
        vjr._fail_job("vjob_f", "09_export", "pipeline_error", "boom")
    finish.assert_not_called()
    assert not mock_job_store["events"].get("vjob_f")


@pytest.mark.parametrize("exc_name", ["JobDeadlineExceeded", "SoftTimeLimitExceeded"])
def test_time_limit_raised_inside_a_stage_fails_as_deadline(mock_job_store, exc_name):
    from celery.exceptions import SoftTimeLimitExceeded

    exc = SoftTimeLimitExceeded if exc_name == "SoftTimeLimitExceeded" else base.JobDeadlineExceeded
    job = _run_pipeline_stopping_with(exc, mock_job_store)
    assert job["status"] == "failed" and job["error"]["type"] == "deadline_exceeded"


def test_soft_time_limit_during_a_script_kills_it_and_propagates(tmp_path):
    import threading
    from celery.exceptions import SoftTimeLimitExceeded

    p1, p2, p3 = _store_patches()
    killed = []
    real_kill = base._kill_process_group

    def _record_kill(pid):
        killed.append(pid)
        real_kill(pid)

    calls = {"n": 0}
    real_raise = job_lease.raise_if_stopped

    def _raise_soft_limit_once(job_id, stage):
        calls["n"] += 1
        if calls["n"] == 2:
            raise SoftTimeLimitExceeded()
        return real_raise(job_id, stage)

    with patch("app.pipeline.base.PROJECT_ROOT", tmp_path), p1, p2, p3, \
         patch.object(base, "_kill_process_group", _record_kill), \
         patch.object(job_lease, "raise_if_stopped", _raise_soft_limit_once):
        with pytest.raises(SoftTimeLimitExceeded):
            base.run_script("soft_job", "09_export", ["sleep", "60"])
    assert killed


def test_leftover_group_is_killed_whole_when_its_leader_is_the_jobs_script():
    import subprocess
    import sys

    # Leader names the job; its child (a stand-in for ffmpeg) does not.
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time; subprocess.Popen(['sleep','60']); time.sleep(60)", "vjob_group_1"],
        start_new_session=True,
    )
    try:
        time.sleep(0.5)
        job_lease.kill_job_script_group("vjob_group_1", leader.pid)
        assert leader.wait(timeout=5) != 0
        time.sleep(0.2)
        import os
        survivors = [p for p in os.listdir("/proc") if p.isdigit() and _pgid(p) == leader.pid]
        assert survivors == []
    finally:
        if leader.poll() is None:
            leader.kill()


def _pgid(pid: str) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return int(fh.read().rsplit(b")", 1)[1].split()[2])
    except (OSError, IndexError, ValueError):
        return None
