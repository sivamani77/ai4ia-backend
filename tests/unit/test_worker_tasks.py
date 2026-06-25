"""Unit tests for app.worker.tasks — Celery task dispatch."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


class TestRunVideoJobTask:
    def test_calls_run_video_job(self):
        from app.worker.tasks import run_video_job_task
        with patch("app.worker.tasks.run_video_job") as mock_run:
            # Call via .run() to bypass Celery machinery
            run_video_job_task.run("vjob_abc123", {"key": "val"})
        mock_run.assert_called_once_with("vjob_abc123", {"key": "val"})

    def test_passes_job_id_and_request(self):
        from app.worker.tasks import run_video_job_task
        received = []
        with patch("app.worker.tasks.run_video_job", side_effect=lambda jid, req: received.append((jid, req))):
            run_video_job_task.run("vjob_test_456", {"prompt": "test"})
        assert received == [("vjob_test_456", {"prompt": "test"})]

    def test_returns_completed_status(self):
        from app.worker.tasks import run_video_job_task
        with patch("app.worker.tasks.run_video_job"):
            result = run_video_job_task.run("vjob_ok", {})
        assert result["status"] == "completed"
        assert result["job_id"] == "vjob_ok"

    def test_exception_propagates(self):
        from app.worker.tasks import run_video_job_task
        with patch("app.worker.tasks.run_video_job", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="fail"):
                run_video_job_task.run("vjob_bad", {})


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
