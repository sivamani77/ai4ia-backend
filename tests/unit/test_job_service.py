"""Unit tests for app.services.job_service (deprecated shim)."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture(autouse=True)
def mock_store(monkeypatch, tmp_path):
    """Patch job_store functions and ensure_runtime_dirs."""
    jobs = {}

    def _save(job_id, data):
        jobs[job_id] = dict(data)

    def _load(job_id):
        if job_id not in jobs:
            raise FileNotFoundError(f"not found: {job_id}")
        return dict(jobs[job_id])

    def _list():
        return list(jobs.values())

    import app.services.job_service as js
    monkeypatch.setattr(js, "save_job_db", _save)
    monkeypatch.setattr(js, "load_job_db", _load)
    monkeypatch.setattr(js, "list_jobs_db", _list)
    monkeypatch.setattr("app.services.job_service.ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr("app.services.job_service.LOGS_DIR", tmp_path / "logs")
    yield jobs


class TestCreateJob:
    def test_returns_dict(self):
        from app.services.job_service import create_job
        result = create_job("video_processing")
        assert isinstance(result, dict)

    def test_has_required_fields(self):
        from app.services.job_service import create_job
        job = create_job("video_processing")
        for field in ("job_id", "type", "status", "progress_percent", "created_at"):
            assert field in job

    def test_status_is_queued(self):
        from app.services.job_service import create_job
        job = create_job("video_processing")
        assert job["status"] == "queued"

    def test_job_id_starts_with_job(self):
        from app.services.job_service import create_job
        job = create_job("video_processing")
        assert job["job_id"].startswith("job_")

    def test_type_stored(self):
        from app.services.job_service import create_job
        job = create_job("my_type")
        assert job["type"] == "my_type"

    def test_payload_stored(self):
        from app.services.job_service import create_job
        job = create_job("video_processing", payload={"key": "val"})
        assert job["payload"] == {"key": "val"}

    def test_empty_payload_defaults(self):
        from app.services.job_service import create_job
        job = create_job("video_processing")
        assert job["payload"] == {}


class TestGetJob:
    def test_existing_job(self):
        from app.services.job_service import create_job, get_job
        job = create_job("video_processing")
        result = get_job(job["job_id"])
        assert result is not None
        assert result["job_id"] == job["job_id"]

    def test_missing_returns_none(self):
        from app.services.job_service import get_job
        assert get_job("nonexistent") is None


class TestSaveJob:
    def test_saves_updated_job(self, mock_store):
        from app.services.job_service import create_job, save_job, get_job
        job = create_job("video_processing")
        job["status"] = "completed"
        save_job(job)
        loaded = get_job(job["job_id"])
        assert loaded["status"] == "completed"


class TestUpdateJob:
    def test_updates_fields(self):
        from app.services.job_service import create_job, update_job, get_job
        job = create_job("video_processing")
        update_job(job["job_id"], status="running", progress_percent=50)
        loaded = get_job(job["job_id"])
        assert loaded["status"] == "running"
        assert loaded["progress_percent"] == 50

    def test_returns_updated_job(self):
        from app.services.job_service import create_job, update_job
        job = create_job("video_processing")
        result = update_job(job["job_id"], status="completed")
        assert result["status"] == "completed"


class TestListJobs:
    def test_empty_list(self):
        from app.services.job_service import list_jobs
        assert list_jobs() == []

    def test_lists_created_jobs(self):
        from app.services.job_service import create_job, list_jobs
        create_job("video_processing")
        create_job("video_processing")
        assert len(list_jobs()) == 2

    def test_limit_respected(self):
        from app.services.job_service import create_job, list_jobs
        for _ in range(5):
            create_job("video_processing")
        result = list_jobs(limit=3)
        assert len(result) <= 3
