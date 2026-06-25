"""Unit tests for app.services.script_runner."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def job_store(tmp_path):
    """In-memory job store fixture for script_runner tests."""
    jobs = {}

    def _save(job_id, data):
        jobs[job_id] = dict(data)

    def _load(job_id):
        if job_id not in jobs:
            raise FileNotFoundError(f"not found: {job_id}")
        return dict(jobs[job_id])

    import app.services.script_runner as sr
    with patch.object(sr, "save_job", side_effect=_save), \
         patch.object(sr, "load_job", side_effect=_load):
        yield jobs


def _seed_job(jobs, job_id, log_path):
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "log_path": str(log_path),
        "progress": 0,
    }


class TestUpdateJob:
    def test_updates_and_returns_job(self, job_store, tmp_path):
        from app.services.script_runner import _update_job
        log_path = tmp_path / "job.log"
        _seed_job(job_store, "j1", log_path)

        with patch("app.services.script_runner.load_job",
                   side_effect=lambda jid: dict(job_store[jid])), \
             patch("app.services.script_runner.save_job",
                   side_effect=lambda jid, d: job_store.update({jid: dict(d)})):
            result = _update_job("j1", status="running")

        assert result["status"] == "running"


class TestRunScriptBackground:
    def test_starts_thread(self, tmp_path):
        from app.services.script_runner import run_script_background
        jobs = {}
        log_path = tmp_path / "j.log"
        jobs["j1"] = {"job_id": "j1", "status": "queued", "log_path": str(log_path), "progress": 0}
        done = threading.Event()

        def _run_script_mock(job_id, command, result_resolver):
            done.set()

        import app.services.script_runner as sr
        with patch.object(sr, "_run_script", side_effect=_run_script_mock):
            run_script_background("j1", ["echo", "hello"])
            done.wait(timeout=2)

        assert done.is_set()


class TestRunScript:
    def _make_jobs(self, tmp_path, job_id="j1"):
        log_path = tmp_path / f"{job_id}.log"
        jobs = {
            job_id: {
                "job_id": job_id,
                "status": "queued",
                "log_path": str(log_path),
                "progress": 0,
            }
        }
        return jobs, log_path

    def test_successful_script_marks_completed(self, tmp_path):
        from app.services.script_runner import _run_script
        jobs, log_path = self._make_jobs(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        import app.services.script_runner as sr
        with patch.object(sr, "load_job", side_effect=lambda jid: dict(jobs[jid])), \
             patch.object(sr, "save_job", side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
             patch.object(sr, "PROJECT_ROOT", tmp_path):
            _run_script("j1", ["true"])

        assert jobs["j1"]["status"] == "completed"

    def test_failing_script_marks_failed(self, tmp_path):
        from app.services.script_runner import _run_script
        jobs, log_path = self._make_jobs(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        import app.services.script_runner as sr
        with patch.object(sr, "load_job", side_effect=lambda jid: dict(jobs[jid])), \
             patch.object(sr, "save_job", side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
             patch.object(sr, "PROJECT_ROOT", tmp_path):
            _run_script("j1", ["false"])

        assert jobs["j1"]["status"] == "failed"
        assert "exit code" in jobs["j1"]["error"]

    def test_result_resolver_called_on_success(self, tmp_path):
        from app.services.script_runner import _run_script
        jobs, log_path = self._make_jobs(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_path = tmp_path / "result.json"
        resolver_called = []

        def resolver():
            resolver_called.append(True)
            return resolved_path

        import app.services.script_runner as sr
        with patch.object(sr, "load_job", side_effect=lambda jid: dict(jobs[jid])), \
             patch.object(sr, "save_job", side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
             patch.object(sr, "PROJECT_ROOT", tmp_path):
            _run_script("j1", ["true"], result_resolver=resolver)

        assert resolver_called
        assert jobs["j1"]["result_path"] == str(resolved_path)

    def test_result_resolver_not_called_on_failure(self, tmp_path):
        from app.services.script_runner import _run_script
        jobs, log_path = self._make_jobs(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        resolver_called = []

        import app.services.script_runner as sr
        with patch.object(sr, "load_job", side_effect=lambda jid: dict(jobs[jid])), \
             patch.object(sr, "save_job", side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
             patch.object(sr, "PROJECT_ROOT", tmp_path):
            _run_script("j1", ["false"], result_resolver=lambda: resolver_called.append(True))

        assert not resolver_called

    def test_exception_marks_failed(self, tmp_path):
        from app.services.script_runner import _run_script
        jobs, log_path = self._make_jobs(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        import app.services.script_runner as sr
        import subprocess
        with patch.object(sr, "load_job", side_effect=lambda jid: dict(jobs[jid])), \
             patch.object(sr, "save_job", side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
             patch.object(sr, "PROJECT_ROOT", tmp_path), \
             patch("app.services.script_runner.subprocess.Popen",
                   side_effect=OSError("popen failed")):
            _run_script("j1", ["whatever"])

        assert jobs["j1"]["status"] == "failed"
        assert "popen failed" in jobs["j1"]["error"]
