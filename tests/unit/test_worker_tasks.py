"""Unit tests for app.worker.tasks — Celery task dispatch."""
from __future__ import annotations

import contextlib
from unittest.mock import patch, MagicMock

import pytest


@contextlib.contextmanager
def _job_row(status: str = "running", delivery_attempts: int = 0, saves: list | None = None):
    """Patch job_store around a task call with one job row.

    run_video_job_task reads the row twice: once in the delivery guard before
    the pipeline starts, once after it finishes to report the final status.
    The guard also writes the incremented delivery counter back.
    """
    row = {"status": status, "delivery_attempts": delivery_attempts}

    def _save(job_id, data):
        row.update(data)
        if saves is not None:
            saves.append(dict(data))

    with patch("app.services.job_store.load_job", side_effect=lambda jid: dict(row)), \
         patch("app.services.job_store.save_job", side_effect=_save):
        yield row


@contextlib.contextmanager
def _ingest_job_row(status: str = "running", delivery_attempts: int = 0, saves: list | None = None):
    """Same as _job_row but for the ingest store — run_ingest_job_task's
    delivery guard reads app.services.ingest_store, not job_store."""
    row = {"status": status, "delivery_attempts": delivery_attempts}

    def _save(job_id, data):
        row.update(data)
        if saves is not None:
            saves.append(dict(data))

    with patch("app.services.ingest_store.load_ingest_job", side_effect=lambda jid: dict(row)), \
         patch("app.services.ingest_store.save_ingest_job", side_effect=_save):
        yield row


@contextlib.contextmanager
def _lease(outcome: str = "run", calls: list | None = None):
    """Patch job_lease around a video task call: claim returns `outcome`."""
    calls = calls if calls is not None else []

    def _claim(job_id, dispatch_id, owner):
        calls.append(("claim", job_id, dispatch_id, owner))
        return outcome

    with patch("app.services.job_lease.claim", side_effect=_claim), \
         patch("app.services.job_lease.start", side_effect=lambda j, o: calls.append(("start", j, o))), \
         patch("app.services.job_lease.release", side_effect=lambda j: calls.append(("release", j))):
        yield calls


class TestRunVideoJobTask:
    def test_calls_run_video_job(self):
        from app.worker.tasks import run_video_job_task
        with _lease(), _job_row(), patch("app.worker.tasks.run_video_job") as mock_run:
            # Call via .run() to bypass Celery machinery
            run_video_job_task.run("vjob_abc123", {"key": "val"})
        mock_run.assert_called_once_with("vjob_abc123", {"key": "val"})

    def test_returns_status_from_job_row(self):
        from app.worker.tasks import run_video_job_task

        def _complete(jid, req):
            row["status"] = "completed"

        with _lease(), _job_row() as row, patch("app.worker.tasks.run_video_job", side_effect=_complete):
            result = run_video_job_task.run("vjob_ok", {})
        assert result == {"job_id": "vjob_ok", "status": "completed"}

    def test_exception_propagates_and_the_lease_is_released(self):
        from app.worker.tasks import run_video_job_task
        with _lease() as calls, _job_row(), \
             patch("app.worker.tasks.run_video_job", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="fail"):
                run_video_job_task.run("vjob_bad", {})
        assert calls[-1] == ("release", "vjob_bad")


class TestLeaseClaim:
    """A delivery runs only when it wins the job's lease (job_lease.claim)."""

    def test_winning_delivery_heartbeats_while_it_runs(self):
        from app.worker.tasks import run_video_job_task
        with _lease() as calls, _job_row(), patch("app.worker.tasks.run_video_job"):
            run_video_job_task.run("vjob_1", {}, dispatch_id="d1")
        kinds = [c[0] for c in calls]
        assert kinds == ["claim", "start", "release"]
        assert calls[0][2] == "d1"
        assert calls[0][3] == calls[1][2]  # the heartbeat uses the claiming owner

    @pytest.mark.parametrize("outcome", ["terminal", "superseded", "leased", "worker_lost", "missing"])
    def test_losing_delivery_does_not_run(self, outcome):
        from app.worker.tasks import run_video_job_task
        with _lease(outcome) as calls, patch("app.worker.tasks.run_video_job") as mock_run:
            result = run_video_job_task.run("vjob_2", {}, dispatch_id="d1")
        mock_run.assert_not_called()
        assert result == {"job_id": "vjob_2", "status": outcome}
        assert [c[0] for c in calls] == ["claim"]

    def test_messages_from_before_leases_claim_without_an_id(self):
        from app.worker.tasks import run_video_job_task
        with _lease() as calls, _job_row(), patch("app.worker.tasks.run_video_job"):
            run_video_job_task.run("vjob_3", {})
        assert calls[0][2] is None

    def test_task_has_a_hard_time_limit_below_the_visibility_timeout(self):
        from app.core.settings import settings
        from app.worker.celery_app import celery_app
        from app.worker.tasks import run_video_job_task

        assert run_video_job_task.time_limit == settings.job_max_runtime_seconds + 300
        visibility = celery_app.conf.broker_transport_options["visibility_timeout"]
        assert visibility > run_video_job_task.time_limit


class TestRunIngestJobTask:
    def test_calls_run_ingest_job(self):
        from app.worker.tasks import run_ingest_job_task
        with _ingest_job_row(), patch("app.worker.tasks.run_ingest_job") as mock_run:
            run_ingest_job_task.run("ingest_abc123")
        mock_run.assert_called_once_with("ingest_abc123")

    def test_passes_job_id_correctly(self):
        from app.worker.tasks import run_ingest_job_task
        received = []
        with _ingest_job_row(), \
             patch("app.worker.tasks.run_ingest_job", side_effect=lambda jid: received.append(jid)):
            run_ingest_job_task.run("ingest_test_789")
        assert received == ["ingest_test_789"]

    def test_returns_completed_status(self):
        from app.worker.tasks import run_ingest_job_task

        def _complete(jid):
            row["status"] = "completed"

        with _ingest_job_row() as row, patch("app.worker.tasks.run_ingest_job", side_effect=_complete):
            result = run_ingest_job_task.run("ingest_ok")
        assert result["status"] == "completed"

    def test_returns_failed_status_when_the_job_ended_failed(self):
        """run_ingest_job catches its own pipeline failures internally and
        calls _fail_job rather than raising -- a real 0/43-downloads run
        showed the task logging 'completed successfully' and returning
        status: completed while the job row itself said 'failed'."""
        from app.worker.tasks import run_ingest_job_task

        def _fail(jid):
            row["status"] = "failed"

        with _ingest_job_row() as row, patch("app.worker.tasks.run_ingest_job", side_effect=_fail):
            result = run_ingest_job_task.run("ingest_bad_run")
        assert result["status"] == "failed"

    def test_exception_propagates(self):
        from app.worker.tasks import run_ingest_job_task
        with _ingest_job_row(), \
             patch("app.worker.tasks.run_ingest_job", side_effect=RuntimeError("ingest fail")):
            with pytest.raises(RuntimeError, match="ingest fail"):
                run_ingest_job_task.run("ingest_bad")


class TestIngestDeliveryGuard:
    """Same acks_late + task_reject_on_worker_lost hazard as the video task:
    an ingest job downloading tens of GB can OOM-kill its worker too, and
    until now nothing bounded how many times the broker would redeliver it."""

    def test_first_delivery_runs_and_records_attempt(self):
        from app.worker.tasks import run_ingest_job_task
        saves = []
        with _ingest_job_row(saves=saves), patch("app.worker.tasks.run_ingest_job") as mock_run:
            run_ingest_job_task.run("ijob_1")
        mock_run.assert_called_once()
        assert saves[0]["delivery_attempts"] == 1

    def test_second_delivery_still_runs(self):
        from app.worker.tasks import run_ingest_job_task
        saves = []
        with _ingest_job_row(delivery_attempts=1, saves=saves), \
             patch("app.worker.tasks.run_ingest_job") as mock_run:
            run_ingest_job_task.run("ijob_2")
        mock_run.assert_called_once()
        assert saves[0]["delivery_attempts"] == 2

    def test_third_delivery_is_abandoned_without_running_the_pipeline(self):
        from app.worker.tasks import run_ingest_job_task
        with _ingest_job_row(delivery_attempts=2), \
             patch("app.worker.tasks.run_ingest_job") as mock_run, \
             patch("app.services.ingest_job_runner._fail_job") as mock_fail:
            result = run_ingest_job_task.run("ijob_poison")

        mock_run.assert_not_called()
        assert result == {"job_id": "ijob_poison", "status": "abandoned"}
        mock_fail.assert_called_once()
        assert mock_fail.call_args.args[0] == "ijob_poison"
        assert mock_fail.call_args.args[1] == "dispatch"

    def test_terminal_job_is_not_rerun(self):
        from app.worker.tasks import run_ingest_job_task
        for status in ("cancelled", "completed", "failed"):
            with _ingest_job_row(status=status), patch("app.worker.tasks.run_ingest_job") as mock_run:
                result = run_ingest_job_task.run("ijob_term")
            mock_run.assert_not_called()
            assert result["status"] == "abandoned"

    def test_unreadable_job_row_does_not_block_the_pipeline(self):
        from app.worker.tasks import run_ingest_job_task

        # Fail only the guard's read; the post-run status read succeeds.
        reads = [RuntimeError("db down"), {"status": "completed"}]

        def _load(jid):
            result = reads.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch("app.services.ingest_store.load_ingest_job", side_effect=_load), \
             patch("app.services.ingest_store.save_ingest_job"), \
             patch("app.worker.tasks.run_ingest_job") as mock_run:
            run_ingest_job_task.run("ijob_dbdown")
        mock_run.assert_called_once()
