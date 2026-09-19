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


class TestRunVideoJobTask:
    def test_calls_run_video_job(self):
        from app.worker.tasks import run_video_job_task
        with _job_row(), patch("app.worker.tasks.run_video_job") as mock_run:
            # Call via .run() to bypass Celery machinery
            run_video_job_task.run("vjob_abc123", {"key": "val"})
        mock_run.assert_called_once_with("vjob_abc123", {"key": "val"})

    def test_passes_job_id_and_request(self):
        from app.worker.tasks import run_video_job_task
        received = []
        with _job_row(), \
             patch("app.worker.tasks.run_video_job",
                   side_effect=lambda jid, req: received.append((jid, req))):
            run_video_job_task.run("vjob_test_456", {"prompt": "test"})
        assert received == [("vjob_test_456", {"prompt": "test"})]

    def test_returns_status_from_job_row(self):
        from app.worker.tasks import run_video_job_task

        def _complete(jid, req):
            row["status"] = "completed"

        with _job_row() as row, patch("app.worker.tasks.run_video_job", side_effect=_complete):
            result = run_video_job_task.run("vjob_ok", {})
        assert result["status"] == "completed"
        assert result["job_id"] == "vjob_ok"

    def test_exception_propagates(self):
        from app.worker.tasks import run_video_job_task
        with _job_row(), patch("app.worker.tasks.run_video_job", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="fail"):
                run_video_job_task.run("vjob_bad", {})


class TestDeliveryGuard:
    """acks_late + task_reject_on_worker_lost redeliver a task whose worker
    died. Without a bound, an OOM-killing job loops forever."""

    def test_first_delivery_runs_and_records_attempt(self):
        from app.worker.tasks import run_video_job_task
        saves = []
        with _job_row(saves=saves), patch("app.worker.tasks.run_video_job") as mock_run:
            run_video_job_task.run("vjob_1", {})
        mock_run.assert_called_once()
        assert saves[0]["delivery_attempts"] == 1

    def test_second_delivery_still_runs(self):
        from app.worker.tasks import run_video_job_task
        saves = []
        with _job_row(delivery_attempts=1, saves=saves), \
             patch("app.worker.tasks.run_video_job") as mock_run:
            run_video_job_task.run("vjob_2", {})
        mock_run.assert_called_once()
        assert saves[0]["delivery_attempts"] == 2

    def test_third_delivery_is_abandoned_without_running_pipeline(self):
        from app.worker.tasks import run_video_job_task
        with _job_row(delivery_attempts=2), \
             patch("app.worker.tasks.run_video_job") as mock_run, \
             patch("app.services.video_job_runner._fail_job") as mock_fail:
            result = run_video_job_task.run("vjob_poison", {})

        mock_run.assert_not_called()
        assert result == {"job_id": "vjob_poison", "status": "abandoned"}
        mock_fail.assert_called_once()
        assert mock_fail.call_args.args[0] == "vjob_poison"
        assert mock_fail.call_args.args[2] == "poison_pill"

    def test_terminal_job_is_not_rerun(self):
        from app.worker.tasks import run_video_job_task
        for status in ("cancelled", "completed", "failed"):
            with _job_row(status=status), patch("app.worker.tasks.run_video_job") as mock_run:
                result = run_video_job_task.run("vjob_term", {})
            mock_run.assert_not_called()
            assert result["status"] == "abandoned"

    def test_unreadable_job_row_does_not_block_the_pipeline(self):
        """The counter is a safety net, not a gate — a store blip on the
        guard's read must not strand a legitimate job."""
        from app.worker.tasks import run_video_job_task

        # Fail only the guard's read; the post-run status read succeeds.
        reads = [RuntimeError("db down"), {"status": "completed"}]

        def _load(jid):
            result = reads.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch("app.services.job_store.load_job", side_effect=_load), \
             patch("app.services.job_store.save_job"), \
             patch("app.worker.tasks.run_video_job") as mock_run:
            run_video_job_task.run("vjob_dbdown", {})
        mock_run.assert_called_once()


class TestRunIngestJobTask:
    def test_calls_run_ingest_job(self):
        from app.worker.tasks import run_ingest_job_task
        with patch("app.worker.tasks.run_ingest_job") as mock_run:
            run_ingest_job_task.run("ingest_abc123")
        mock_run.assert_called_once_with("ingest_abc123")

    def test_passes_job_id_correctly(self):
        from app.worker.tasks import run_ingest_job_task
        received = []
        with patch("app.worker.tasks.run_ingest_job", side_effect=lambda jid: received.append(jid)):
            run_ingest_job_task.run("ingest_test_789")
        assert received == ["ingest_test_789"]

    def test_returns_completed_status(self):
        from app.worker.tasks import run_ingest_job_task
        with patch("app.worker.tasks.run_ingest_job"):
            result = run_ingest_job_task.run("ingest_ok")
        assert result["status"] == "completed"

    def test_exception_propagates(self):
        from app.worker.tasks import run_ingest_job_task
        with patch("app.worker.tasks.run_ingest_job", side_effect=RuntimeError("ingest fail")):
            with pytest.raises(RuntimeError, match="ingest fail"):
                run_ingest_job_task.run("ingest_bad")
