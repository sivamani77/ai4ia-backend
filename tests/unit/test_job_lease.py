"""Job leases: claim outcomes, heartbeat stop signals, leftover-script kill
(docs/job-recovery-plan.md, section 5)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from app.services import job_lease


@pytest.fixture(autouse=True)
def _clean_leases():
    job_lease._leases.clear()
    yield
    for job_id in list(job_lease._leases):
        job_lease.release(job_id)


def _seed(mock_job_store, job_id="vjob_1", **fields):
    job = {"job_id": job_id, "status": "queued", "dispatch_id": "d1", "attempts": 0, "user_id": "u1"}
    job.update(fields)
    mock_job_store["jobs"][job_id] = job
    return job


# --- claim --------------------------------------------------------------------

def test_first_matching_delivery_runs(mock_job_store):
    _seed(mock_job_store)
    assert job_lease.claim("vjob_1", "d1", "me") == "run"
    assert mock_job_store["jobs"]["vjob_1"]["status"] == "running"


def test_second_delivery_while_holder_is_alive_is_dropped(mock_job_store):
    _seed(mock_job_store)
    job_lease.claim("vjob_1", "d1", "otherhost:1:t1")
    assert job_lease.claim("vjob_1", "d1", "me") == "leased"
    assert mock_job_store["jobs"]["vjob_1"]["status"] == "running"


@pytest.mark.parametrize("fields, outcome", [
    ({"status": "completed"}, "terminal"),
    ({"status": "cancelled"}, "terminal"),
    ({"dispatch_id": "d2"}, "superseded"),
])
def test_stale_or_finished_deliveries_are_dropped(mock_job_store, fields, outcome):
    _seed(mock_job_store, **fields)
    assert job_lease.claim("vjob_1", "d1", "me") == outcome


def test_delivery_after_a_dead_holder_fails_the_job_without_rerunning(mock_job_store, monkeypatch):
    """No retries: a redelivered job whose worker is gone is failed, not rerun."""
    from app.services import processing_records

    monkeypatch.setattr(processing_records, "finish", lambda *a, **k: None)
    _seed(mock_job_store, status="running", attempts=1, lease_owner="gonehost:1:t1", heartbeat_age=9999)

    assert job_lease.claim("vjob_1", "d1", "me") == "worker_lost"
    job = mock_job_store["jobs"]["vjob_1"]
    assert job["status"] == "failed" and job["error"]["type"] == "worker_lost"
    assert any(e["event_type"] == "failed" for e in mock_job_store["events"]["vjob_1"])


def test_holder_pid_gone_on_this_host_counts_as_dead(monkeypatch):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert job_lease._holder_is_dead(f"{job_lease.HOST}:{dead.pid}:t", heartbeat_age=1.0)
    assert not job_lease._holder_is_dead(f"{job_lease.HOST}:{os.getppid()}:t", heartbeat_age=1.0)
    assert not job_lease._holder_is_dead("otherhost:123:t", heartbeat_age=1.0)
    assert job_lease._holder_is_dead("otherhost:123:t", heartbeat_age=10_000)
    assert job_lease._holder_is_dead(None, heartbeat_age=None)


def test_leftover_scripts_killed_only_when_they_name_the_job():
    ours = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "vjob_leftover_1"],
                            start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "vjob_someone_else"],
                             start_new_session=True)
    try:
        job_lease._kill_leftover_scripts("vjob_leftover_1", {"active_host": job_lease.HOST, "active_pid": ours.pid})
        job_lease._kill_leftover_scripts("vjob_leftover_1", {"active_host": job_lease.HOST, "active_pid": other.pid})
        assert ours.wait(timeout=5) != 0
        assert other.poll() is None
    finally:
        for p in (ours, other):
            if p.poll() is None:
                p.kill()
                p.wait()


# --- heartbeat and stop signal ------------------------------------------------

def _fast_heartbeat(monkeypatch, interval=0.05, stale=1):
    monkeypatch.setattr(job_lease.settings, "job_heartbeat_interval_seconds", interval, raising=False)
    monkeypatch.setattr(job_lease.settings, "job_stale_after_seconds", stale, raising=False)


def _wait_for_reason(job_id, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        reason = job_lease.stop_reason(job_id)
        if reason:
            return reason
        time.sleep(0.02)
    return None


def test_heartbeat_detects_a_lost_lease(mock_job_store, monkeypatch):
    _fast_heartbeat(monkeypatch)
    _seed(mock_job_store)
    job_lease.claim("vjob_1", "d1", "me")
    job_lease.start("vjob_1", "me")
    mock_job_store["jobs"]["vjob_1"]["lease_owner"] = "someone-else"

    assert _wait_for_reason("vjob_1") == job_lease.STOP_SUPERSEDED


def test_heartbeat_reports_cancellation_as_such(mock_job_store, monkeypatch):
    _fast_heartbeat(monkeypatch)
    _seed(mock_job_store)
    job_lease.claim("vjob_1", "d1", "me")
    job_lease.start("vjob_1", "me")
    mock_job_store["jobs"]["vjob_1"]["status"] = "cancelled"

    assert _wait_for_reason("vjob_1") == job_lease.STOP_CANCELLED


def test_worker_stops_itself_when_heartbeats_keep_failing(mock_job_store, monkeypatch):
    """Cut off from the database: stop before the sweeper could start a second run."""
    import app.services.job_store as js

    _fast_heartbeat(monkeypatch, interval=0.05, stale=0.5)

    def _db_down(*a, **k):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(js, "heartbeat_job", _db_down)
    job_lease.start("vjob_1", "me")

    assert _wait_for_reason("vjob_1") == job_lease.STOP_HEARTBEAT_LOST


def test_runtime_cap_is_a_stop_signal(monkeypatch):
    monkeypatch.setattr(job_lease.settings, "job_max_runtime_seconds", 0, raising=False)
    monkeypatch.setattr(job_lease.settings, "job_heartbeat_interval_seconds", 60, raising=False)
    job_lease.start("vjob_1", "me")
    assert job_lease.stop_reason("vjob_1") == job_lease.STOP_DEADLINE


def test_release_stops_the_signal_and_returns_the_owner(monkeypatch):
    monkeypatch.setattr(job_lease.settings, "job_heartbeat_interval_seconds", 60, raising=False)
    job_lease.start("vjob_1", "me")
    assert job_lease.release("vjob_1") == "me"
    assert job_lease.stop_reason("vjob_1") is None
    assert job_lease.release("vjob_1") is None


@pytest.mark.parametrize("reason, exc_name", [
    (job_lease.STOP_CANCELLED, "JobCancelledException"),
    (job_lease.STOP_DEADLINE, "JobDeadlineExceeded"),
    (job_lease.STOP_SUPERSEDED, "JobSupersededException"),
    (job_lease.STOP_HEARTBEAT_LOST, "JobSupersededException"),
])
def test_stop_reasons_map_to_exceptions(monkeypatch, reason, exc_name):
    import app.pipeline.base as base

    monkeypatch.setattr(job_lease.settings, "job_heartbeat_interval_seconds", 60, raising=False)
    job_lease.start("vjob_1", "me")
    job_lease._leases["vjob_1"].stop_reason = reason
    with pytest.raises(getattr(base, exc_name)):
        job_lease.raise_if_stopped("vjob_1", "09_export")
