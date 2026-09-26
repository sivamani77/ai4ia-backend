"""Recovery sweeper decisions (docs/design/job-recovery-plan.md, section 6)."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from app.services import job_sweeper


class _Cursor:
    def __init__(self, leader: bool):
        self.leader = leader
        self.sql: list[str] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchone(self):
        return {"locked": self.leader}


@pytest.fixture
def sweeper(monkeypatch):
    """A sweeper wired to a fake lock connection and a mock repository."""
    repo = MagicMock()
    repo.fail_where.return_value = []
    repo.lost_queue_candidates.return_value = []
    cursor = _Cursor(leader=True)

    @contextmanager
    def _session():
        yield cursor

    monkeypatch.setattr(job_sweeper, "pg_session", _session)
    monkeypatch.setattr(job_sweeper.job_store, "get_job_repo", lambda: repo)
    monkeypatch.setattr(job_sweeper.settings, "worker_queue", "video_jobs_uat", raising=False)
    monkeypatch.setattr(job_sweeper.settings, "job_stale_after_seconds", 300, raising=False)
    monkeypatch.setattr(job_sweeper, "_healthy_since", None)
    recorded = []
    monkeypatch.setattr(job_sweeper.job_lease, "record_failure", lambda *a, **k: recorded.append((*a, k)))
    return repo, cursor, recorded


def _rule_reasons(repo):
    """Which rule each fail_where call belongs to, read from its WHERE clause."""
    out = []
    for call in repo.fail_where.call_args_list:
        where = call.args[0]
        if "heartbeat_at" in where:
            out.append("worker_lost")
        elif "started_at" in where:
            out.append("deadline_exceeded")
        elif "worker_queue is null" in where:
            out.append("expired")
        else:
            out.append("queue_expired")
    return out


def test_only_the_leader_acts(sweeper):
    repo, cursor, _ = sweeper
    cursor.leader = False

    assert job_sweeper.sweep_once() == "not_leader"
    repo.fail_where.assert_not_called()


def test_stale_heartbeats_ignored_until_sweeps_have_been_healthy_a_full_window(sweeper, monkeypatch):
    """After an API restart or database outage, heartbeats couldn't be written;
    wait a stale window before trusting their age."""
    repo, _, _ = sweeper
    clock = [1000.0]
    monkeypatch.setattr(job_sweeper.time, "monotonic", lambda: clock[0])

    job_sweeper.sweep_once()
    assert "worker_lost" not in _rule_reasons(repo)

    repo.fail_where.reset_mock()
    clock[0] += 301
    job_sweeper.sweep_once()
    assert "worker_lost" in _rule_reasons(repo)


def test_a_failed_pass_resets_the_trust_window(sweeper, monkeypatch):
    repo, _, _ = sweeper
    clock = [1000.0]
    monkeypatch.setattr(job_sweeper.time, "monotonic", lambda: clock[0])
    job_sweeper.sweep_once()
    clock[0] += 301

    repo.fail_where.side_effect = RuntimeError("database down")
    assert job_sweeper.sweep_once() == "error"
    repo.fail_where.side_effect = None
    repo.fail_where.reset_mock()
    job_sweeper.sweep_once()
    assert "worker_lost" not in _rule_reasons(repo)


def test_every_rule_is_scoped_to_this_environments_queue(sweeper):
    repo, _, _ = sweeper
    job_sweeper._healthy_since = -1e9  # trusted
    job_sweeper.sweep_once()

    for call in repo.fail_where.call_args_list:
        where, params = call.args[0], call.args[1]
        assert "worker_queue = %(queue)s" in where or "worker_queue is null" in where
        if "worker_queue = %(queue)s" in where:
            assert params["queue"] == "video_jobs_uat"


def test_failed_rows_get_events_and_records(sweeper):
    repo, _, recorded = sweeper
    job_sweeper._healthy_since = -1e9
    repo.fail_where.side_effect = lambda where, params, error: (
        [{"id": "vjob_dead", "user_id": "u1", "stage": "09_export"}] if "heartbeat_at" in where else []
    )
    job_sweeper.sweep_once()
    assert recorded == [("vjob_dead", "worker_lost", job_sweeper._ERRORS["worker_lost"], {"by_sweeper": True})]


@pytest.mark.parametrize("size, resent", [(0, True), (3, False), (None, False)])
def test_lost_messages_are_resent_only_when_the_queue_is_empty(sweeper, monkeypatch, size, resent):
    repo, _, _ = sweeper
    repo.lost_queue_candidates.return_value = [{"id": "vjob_q", "dispatch_id": "d1", "request": {"prompt": "p"}}]
    repo.redispatch.return_value = True
    monkeypatch.setattr(job_sweeper, "_queue_size", lambda queue: size)
    sent = []
    import app.services.video_job_runner as vjr
    monkeypatch.setattr(vjr, "start_video_job_background", lambda job_id, req, dispatch_id: sent.append(dispatch_id))

    job_sweeper.sweep_once()

    assert bool(sent) is resent
    if resent:
        new_id = repo.redispatch.call_args.kwargs["new_dispatch_id"]
        assert sent == [new_id] and new_id != "d1"


def test_resend_skipped_when_another_pass_already_changed_the_row(sweeper, monkeypatch):
    repo, _, _ = sweeper
    repo.lost_queue_candidates.return_value = [{"id": "vjob_q", "dispatch_id": "d1", "request": {}}]
    repo.redispatch.return_value = False
    monkeypatch.setattr(job_sweeper, "_queue_size", lambda queue: 0)
    import app.services.video_job_runner as vjr
    send = MagicMock()
    monkeypatch.setattr(vjr, "start_video_job_background", send)

    job_sweeper.sweep_once()
    send.assert_not_called()


def test_lock_is_released_even_when_a_rule_fails(sweeper):
    repo, cursor, _ = sweeper
    repo.fail_where.side_effect = RuntimeError("boom")
    job_sweeper.sweep_once()
    assert any("pg_advisory_unlock" in sql for sql in cursor.sql)


def test_queue_size_is_unknown_without_redis_and_never_waits(monkeypatch):
    import app.services.video_job_runner as vjr

    monkeypatch.setattr(vjr, "_redis_reachable", lambda: False)
    assert job_sweeper._queue_size("video_jobs_uat") is None
