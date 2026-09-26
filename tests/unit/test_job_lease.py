"""Job leases: claim outcomes, heartbeat stop signals, leftover-script kill
(docs/design/job-recovery-plan.md, section 5)."""
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


@pytest.fixture
def _no_records(monkeypatch):
    from app.services import processing_records

    monkeypatch.setattr(processing_records, "finish", lambda *a, **k: None)


def _seed_lost(mock_job_store, **fields):
    lost = {"status": "running", "attempts": 1, "lease_owner": "gonehost:1:t1",
            "heartbeat_age": 9999, "current_stage": "09_export"}
    return _seed(mock_job_store, **{**lost, **fields})


def test_delivery_after_a_dead_holder_reruns_the_job(mock_job_store, _no_records):
    _seed_lost(mock_job_store)

    assert job_lease.claim("vjob_1", "d1", "me") == "run"
    job = mock_job_store["jobs"]["vjob_1"]
    assert job["status"] == "running" and job["lease_owner"] == "me" and job["attempts"] == 2
    [event] = [e for e in mock_job_store["events"]["vjob_1"] if e["event_type"] == "interrupted"]
    assert event["payload"]["stage"] == "09_export"
    assert event["payload"]["attempt"] == 2 and event["payload"]["message"] == job_lease.INTERRUPTED_MESSAGE


def test_no_retries_configured_fails_the_job_as_before(mock_job_store, _no_records, monkeypatch):
    monkeypatch.setattr(job_lease.settings, "job_max_worker_loss_retries", 0, raising=False)
    _seed_lost(mock_job_store)

    assert job_lease.claim("vjob_1", "d1", "me") == "worker_lost"
    job = mock_job_store["jobs"]["vjob_1"]
    assert job["status"] == "failed" and job["error"]["type"] == "worker_lost"
    assert any(e["event_type"] == "failed" for e in mock_job_store["events"]["vjob_1"])


def test_fourth_lost_run_fails_the_job(mock_job_store, _no_records):
    _seed_lost(mock_job_store, attempts=4)

    assert job_lease.claim("vjob_1", "d1", "me") == "worker_lost_repeatedly"
    assert mock_job_store["jobs"]["vjob_1"]["error"]["type"] == "worker_lost_repeatedly"


def test_second_loss_in_the_same_stage_fails_the_job(mock_job_store, _no_records):
    _seed_lost(mock_job_store, attempts=2)
    mock_job_store["events"]["vjob_1"] = [
        {"id": 1, "job_id": "vjob_1", "event_type": "interrupted", "payload": {"stage": "09_export"}}]

    assert job_lease.claim("vjob_1", "d1", "me") == "worker_lost_repeatedly"


def test_loss_in_a_new_stage_is_retried(mock_job_store, _no_records):
    _seed_lost(mock_job_store, attempts=2)
    mock_job_store["events"]["vjob_1"] = [
        {"id": 1, "job_id": "vjob_1", "event_type": "interrupted", "payload": {"stage": "08_assets"}}]

    assert job_lease.claim("vjob_1", "d1", "me") == "run"


@pytest.mark.parametrize("attempts", [1, 3])
def test_requeued_job_is_claimed_while_it_has_retries_left(mock_job_store, attempts):
    _seed(mock_job_store, attempts=attempts)
    assert job_lease.claim("vjob_1", "d1", "me") == "run"


@pytest.mark.parametrize("retries, reason", [(3, "worker_lost_repeatedly"), (0, "worker_lost")])
def test_queued_job_past_a_lowered_retry_setting_is_failed(mock_job_store, _no_records, monkeypatch,
                                                           retries, reason):
    from unittest.mock import MagicMock

    monkeypatch.setattr(job_lease.settings, "job_max_worker_loss_retries", retries, raising=False)
    repo = MagicMock()
    repo.fail_where.return_value = [{"id": "vjob_1"}]
    monkeypatch.setattr(job_lease.job_store, "get_job_repo", lambda: repo)
    _seed(mock_job_store, attempts=4)

    assert job_lease.claim("vjob_1", "d1", "me") == reason
    assert repo.fail_where.call_args.kwargs["error"]["type"] == reason


def test_a_hung_heartbeat_stops_the_run(mock_job_store, monkeypatch):
    """A heartbeat write that blocks never raises; the stop still comes."""
    _seed(mock_job_store)
    job_lease.claim("vjob_1", "d1", "me")
    job_lease.start("vjob_1", "me")
    job_lease._leases["vjob_1"].last_ok -= 0.8 * job_lease.settings.job_stale_after_seconds + 1

    assert job_lease.stop_reason("vjob_1") == job_lease.STOP_HEARTBEAT_LOST
    job_lease.release("vjob_1")
    assert job_lease.ended_by("vjob_1") == job_lease.STOP_HEARTBEAT_LOST
    assert job_lease.ended_by("vjob_1") is None


def test_retried_claim_kills_the_lost_runs_leftover_script(mock_job_store):
    leftover = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "vjob_1"],
                                start_new_session=True)
    try:
        _seed(mock_job_store, attempts=1, active_host=job_lease.HOST, active_pid=leftover.pid)
        assert job_lease.claim("vjob_1", "d1", "me") == "run"
        assert leftover.wait(timeout=5) != 0
    finally:
        if leftover.poll() is None:
            leftover.kill()
            leftover.wait()


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
