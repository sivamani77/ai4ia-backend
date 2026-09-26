"""Regression tests for VID-001 — a cancellation silently reverted by a
concurrent full-row job save.

Every pipeline writer persists through job_store.save_job(), which builds a
full row from a dict read earlier by load_job() and hands it to
JobRepository.upsert_job(). Cancellation does NOT take that path: the API's
DELETE handler writes through the atomic update_job_partial() function. So a
save whose snapshot predates a cancellation used to write the stale status
back over it:

    worker: job = load_job(id)     -> snapshot says "running"
    api   : cancel_job(id)         -> row becomes "cancelled"
    worker: save_job(id, job)      -> row returns to "running"

run_script's cancel poll then read "running" again and the render ran to
completion, burning GPU time on a cancelled job. update_stage() fires on every
progress tick, so the window is hit easily during a long stage-09 export.

The fix makes a terminal status sticky inside the upsert statement itself, so
the decision is made against the row's committed value with no check-then-write
gap. These tests assert the emitted SQL keeps that guard; the behavioural
round-trip is verified against the local Docker Postgres (docker-compose.infra.yml).
"""
from __future__ import annotations

from unittest.mock import patch

from app.db.repositories.job_repository import JobRepository

TERMINAL = ("cancelled", "completed", "failed")


def _norm(sql: str) -> str:
    return " ".join(sql.split()).lower()


def _captured_upsert_sql() -> str:
    captured = {}

    def fake(query, params=None):
        captured["q"] = query

    with patch("app.db.repositories.job_repository.pg_execute", side_effect=fake):
        JobRepository().upsert_job({
            "id": "vjob_1",
            "status": "running",
            "user_id": "u1",
            "metadata": {"status": "running"},
        })
    return _norm(captured["q"])


class TestTerminalStatusIsSticky:
    def test_status_is_not_a_bare_overwrite(self):
        """A bare `status = excluded.status` is exactly the defect."""
        sql = _captured_upsert_sql()
        assert "status = excluded.status," not in sql

    def test_status_guard_names_every_terminal_state(self):
        sql = _captured_upsert_sql()
        assert "status = case" in sql
        for state in TERMINAL:
            assert f"'{state}'" in sql, f"terminal state {state!r} missing from the guard"

    def test_guard_preserves_the_committed_row_value(self):
        """The preserved value must come from the row (jobs.status), not the
        incoming snapshot — otherwise the guard is decorative."""
        sql = _captured_upsert_sql()
        assert "then jobs.status" in sql

    def test_metadata_status_cannot_contradict_the_column(self):
        """load_job() reports the column, so a stale metadata.status is only a
        debugging trap — but it is one, so the blob is re-pinned when the
        column is preserved."""
        sql = _captured_upsert_sql()
        assert "jsonb_build_object('status', jobs.status)" in sql

    def test_metadata_is_still_replaced_wholesale_when_not_terminal(self):
        """Deliberately NOT a jsonb merge: job dicts legitimately drop keys
        (run_script clears active_pid by popping it before saving), and a
        `||` merge cannot express a deletion. This is why the fix guards the
        status column rather than routing save_job through the partial-merge
        path."""
        sql = _captured_upsert_sql()
        assert "else excluded.metadata" in sql


class TestNonTerminalWritesAreUnaffected:
    def test_incoming_status_still_used_for_non_terminal_rows(self):
        sql = _captured_upsert_sql()
        assert "else excluded.status" in sql

    def test_other_columns_remain_plain_overwrites(self):
        """The guard is scoped to status/metadata, plus how a leased job ended;
        everything else in a snapshot save still overwrites."""
        sql = _captured_upsert_sql()
        for column in ("progress", "job_type", "worker_id", "retry_count"):
            assert f"{column} = excluded.{column}" in sql

    def test_leased_jobs_keep_the_start_time_their_claim_set(self):
        """The runtime cap counts from the claim; a late snapshot from an
        earlier run must not move it (issue #19)."""
        sql = " ".join(_captured_upsert_sql().split())
        assert ("started_at = case when jobs.dispatch_id is not null then jobs.started_at "
                "else excluded.started_at end") in sql

    def test_finished_leased_jobs_keep_their_error_and_completion_time(self):
        """A late snapshot from a worker the sweeper already failed must not
        erase the failure (docs/design/job-recovery-plan.md)."""
        sql = " ".join(_captured_upsert_sql().split())
        for column in ("completed_at", "error"):
            assert (f"{column} = case when jobs.dispatch_id is not null and jobs.status in "
                    f"('cancelled', 'completed', 'failed') then jobs.{column} else excluded.{column} end") in sql
