"""Unit tests for app.services.ingest_store — ingest job/status/event helpers."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_ingest_job_repo():
    repo = MagicMock()
    repo._store: dict[str, dict] = {}

    def upsert_job(row):
        repo._store[row["id"]] = dict(row)

    def get_job(job_id):
        return repo._store.get(job_id)

    def list_jobs(limit=50):
        return list(repo._store.values())[:limit]

    def list_running_jobs():
        return [r for r in repo._store.values() if r.get("status") == "running"]

    repo.upsert_job.side_effect = upsert_job
    repo.get_job.side_effect = get_job
    repo.list_jobs.side_effect = list_jobs
    repo.list_running_jobs.side_effect = list_running_jobs
    return repo


def _make_status_repo():
    repo = MagicMock()
    repo._store: list[dict] = []

    def upsert(row):
        key = (row["video_id"], row["stage"])
        existing = next((r for r in repo._store if (r["video_id"], r["stage"]) == key), None)
        if existing:
            existing.update(row)
        else:
            repo._store.append(dict(row))
        return dict(row)

    def get_for_video(video_id):
        return [r for r in repo._store if r["video_id"] == video_id]

    def get_for_videos(video_ids):
        return [r for r in repo._store if r["video_id"] in video_ids]

    def get_completed_video_ids():
        return {r["video_id"] for r in repo._store
                if r.get("status") == "completed" and r.get("stage") == "embed"}

    def list_by_stage_status(stage, status):
        return [r for r in repo._store if r.get("stage") == stage and r.get("status") == status]

    repo.upsert.side_effect = upsert
    repo.get_for_video.side_effect = get_for_video
    repo.get_for_videos.side_effect = get_for_videos
    repo.get_completed_video_ids.side_effect = get_completed_video_ids
    repo.list_by_stage_status.side_effect = list_by_stage_status
    return repo


def _make_event_repo():
    repo = MagicMock()
    repo._store: list[dict] = []

    def add_event(job_id, event_type, data):
        repo._store.append({
            "id": len(repo._store) + 1,
            "job_id": job_id,
            "event_type": event_type,
            "data": data,
        })

    def get_events(job_id):
        return [e for e in repo._store if e["job_id"] == job_id]

    def get_events_after(job_id, after_id):
        return [e for e in repo._store if e["job_id"] == job_id and e["id"] > after_id]

    repo.add_event.side_effect = add_event
    repo.get_events.side_effect = get_events
    repo.get_events_after.side_effect = get_events_after
    return repo


@pytest.fixture
def patched_ingest_store():
    import app.services.ingest_store as ist

    job_repo = _make_ingest_job_repo()
    status_repo = _make_status_repo()
    event_repo = _make_event_repo()

    ist._job_repo.cache_clear()
    ist._status_repo.cache_clear()
    ist._event_repo.cache_clear()

    with patch.object(ist, "_job_repo", return_value=job_repo), \
         patch.object(ist, "_status_repo", return_value=status_repo), \
         patch.object(ist, "_event_repo", return_value=event_repo):
        yield {
            "job_repo": job_repo,
            "status_repo": status_repo,
            "event_repo": event_repo,
        }


# ---------------------------------------------------------------------------
# save_ingest_job / load_ingest_job
# ---------------------------------------------------------------------------

class TestSaveLoadIngestJob:
    def test_save_and_load(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, load_ingest_job
        data = {"status": "queued", "channel_url": "https://yt.be/@ai4ia", "progress": 0}
        save_ingest_job("ij1", data)
        loaded = load_ingest_job("ij1")
        assert loaded["job_id"] == "ij1"
        assert loaded["status"] == "queued"

    def test_load_missing_raises(self, patched_ingest_store):
        from app.services.ingest_store import load_ingest_job
        with pytest.raises(FileNotFoundError):
            load_ingest_job("nonexistent")

    def test_progress_returned(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, load_ingest_job
        data = {"status": "running", "progress": 55}
        save_ingest_job("ij2", data)
        loaded = load_ingest_job("ij2")
        assert loaded["progress"] == 55

    def test_metadata_preserved(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, load_ingest_job
        data = {"status": "queued", "channel_url": "https://yt.be/@ai4ia",
                "video_limit": 10, "custom_key": "custom_value"}
        save_ingest_job("ij3", data)
        loaded = load_ingest_job("ij3")
        assert loaded.get("custom_key") == "custom_value"


# ---------------------------------------------------------------------------
# list_ingest_jobs
# ---------------------------------------------------------------------------

class TestListIngestJobs:
    def test_empty_returns_empty(self, patched_ingest_store):
        from app.services.ingest_store import list_ingest_jobs
        assert list_ingest_jobs() == []

    def test_lists_all_jobs(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, list_ingest_jobs
        save_ingest_job("ij1", {"status": "queued"})
        save_ingest_job("ij2", {"status": "running"})
        result = list_ingest_jobs()
        assert len(result) == 2

    def test_each_has_job_id(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, list_ingest_jobs
        save_ingest_job("ij1", {"status": "queued"})
        result = list_ingest_jobs()
        assert all("job_id" in j for j in result)


# ---------------------------------------------------------------------------
# list_running_ingest_jobs
# ---------------------------------------------------------------------------

class TestListRunningIngestJobs:
    def test_returns_only_running(self, patched_ingest_store):
        from app.services.ingest_store import save_ingest_job, list_running_ingest_jobs
        save_ingest_job("ij1", {"status": "running"})
        save_ingest_job("ij2", {"status": "queued"})
        save_ingest_job("ij3", {"status": "completed"})
        result = list_running_ingest_jobs()
        # The mock returns rows with status=="running"
        assert all(r.get("status") == "running" for r in result)


# ---------------------------------------------------------------------------
# upsert_video_stage / get_video_stages
# ---------------------------------------------------------------------------

class TestVideoStages:
    def test_upsert_and_get(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_video_stages
        upsert_video_stage("vid1", "download", "completed", job_id="ij1")
        stages = get_video_stages("vid1")
        assert len(stages) == 1
        assert stages[0]["stage"] == "download"
        assert stages[0]["status"] == "completed"

    def test_upsert_overwrites(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_video_stages
        upsert_video_stage("vid1", "download", "running", job_id="ij1")
        upsert_video_stage("vid1", "download", "completed", job_id="ij1")
        stages = get_video_stages("vid1")
        assert len(stages) == 1
        assert stages[0]["status"] == "completed"

    def test_multiple_stages_for_one_video(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_video_stages
        upsert_video_stage("vid1", "download", "completed", job_id="ij1")
        upsert_video_stage("vid1", "transcribe", "running", job_id="ij1")
        stages = get_video_stages("vid1")
        assert len(stages) == 2

    def test_stages_isolated_by_video(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_video_stages
        upsert_video_stage("vid1", "download", "completed", job_id="ij1")
        upsert_video_stage("vid2", "download", "running", job_id="ij1")
        assert len(get_video_stages("vid1")) == 1
        assert len(get_video_stages("vid2")) == 1


# ---------------------------------------------------------------------------
# get_completed_video_ids
# ---------------------------------------------------------------------------

class TestCompletedVideoIds:
    def test_completed_video_included(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_completed_video_ids
        upsert_video_stage("vid1", "embed", "completed", job_id="ij1")
        completed = get_completed_video_ids()
        assert "vid1" in completed

    def test_incomplete_video_excluded(self, patched_ingest_store):
        from app.services.ingest_store import upsert_video_stage, get_completed_video_ids
        upsert_video_stage("vid1", "embed", "running", job_id="ij1")
        completed = get_completed_video_ids()
        assert "vid1" not in completed

    def test_returns_set(self, patched_ingest_store):
        from app.services.ingest_store import get_completed_video_ids
        assert isinstance(get_completed_video_ids(), set)


# ---------------------------------------------------------------------------
# append_ingest_event / load_ingest_events / load_ingest_events_after
# ---------------------------------------------------------------------------

class TestIngestEvents:
    def test_append_and_load(self, patched_ingest_store):
        from app.services.ingest_store import append_ingest_event, load_ingest_events
        append_ingest_event("ij1", "progress", {"stage": "download"})
        events = load_ingest_events("ij1")
        assert len(events) == 1

    def test_load_after_cursor(self, patched_ingest_store):
        from app.services.ingest_store import append_ingest_event, load_ingest_events_after
        append_ingest_event("ij1", "progress", {"stage": "download"})
        append_ingest_event("ij1", "progress", {"stage": "transcribe"})
        append_ingest_event("ij1", "progress", {"stage": "embed"})
        events = load_ingest_events_after("ij1", after_id=1)
        assert len(events) == 2

    def test_events_isolated_by_job(self, patched_ingest_store):
        from app.services.ingest_store import append_ingest_event, load_ingest_events
        append_ingest_event("ij1", "started", {})
        append_ingest_event("ij2", "started", {})
        assert len(load_ingest_events("ij1")) == 1
        assert len(load_ingest_events("ij2")) == 1

    def test_event_append_failure_does_not_raise(self, patched_ingest_store):
        from app.services.ingest_store import append_ingest_event
        # Even if the repo raises, append_ingest_event swallows it
        patched_ingest_store["event_repo"].add_event.side_effect = Exception("DB down")
        # Should not raise
        append_ingest_event("ij1", "progress", {})
