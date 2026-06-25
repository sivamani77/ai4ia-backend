"""Unit tests for app.services.job_store — job and event persistence helpers."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_job_repo():
    repo = MagicMock()
    repo._store: dict[str, dict] = {}

    def upsert_job(row):
        repo._store[row["id"]] = dict(row)

    def get_job(job_id):
        return repo._store.get(job_id)

    def list_jobs():
        return list(repo._store.values())

    def update_job(job_id, data, merge_existing=True):
        if job_id in repo._store:
            if merge_existing:
                repo._store[job_id].update(data)
            else:
                repo._store[job_id] = {**repo._store[job_id], **data}

    repo.upsert_job.side_effect = upsert_job
    repo.get_job.side_effect = get_job
    repo.list_jobs.side_effect = list_jobs
    repo.update_job.side_effect = update_job
    return repo


def _make_mock_event_repo():
    repo = MagicMock()
    repo._store: list[dict] = []

    def add_event(event):
        repo._store.append({**event, "id": len(repo._store) + 1})

    def get_events(job_id):
        return [e for e in repo._store if e["job_id"] == job_id]

    def get_events_after(job_id, after_id):
        return [e for e in repo._store if e["job_id"] == job_id and e["id"] > after_id]

    repo.add_event.side_effect = add_event
    repo.get_events.side_effect = get_events
    repo.get_events_after.side_effect = get_events_after
    return repo


@pytest.fixture
def patched_store():
    """Patch both repo singletons and reset lru_cache between tests."""
    import app.services.job_store as js

    job_repo = _make_mock_job_repo()
    event_repo = _make_mock_event_repo()

    js.get_job_repo.cache_clear()
    js.get_event_repo.cache_clear()

    with patch.object(js, "get_job_repo", return_value=job_repo), \
         patch.object(js, "get_event_repo", return_value=event_repo):
        yield {"job_repo": job_repo, "event_repo": event_repo}


# ---------------------------------------------------------------------------
# _to_supabase_job
# ---------------------------------------------------------------------------

class TestToSupabaseJob:
    def test_basic_mapping(self):
        from app.services.job_store import _to_supabase_job
        data = {"status": "queued", "request": {"prompt": "test"}}
        row = _to_supabase_job("j1", data)
        assert row["id"] == "j1"
        assert row["status"] == "queued"
        assert row["source_value"] == "test"

    def test_prompt_from_payload_fallback(self):
        from app.services.job_store import _to_supabase_job
        data = {"status": "queued", "payload": {"prompt": "from payload"}}
        row = _to_supabase_job("j1", data)
        assert row["source_value"] == "from payload"

    def test_empty_prompt_when_missing(self):
        from app.services.job_store import _to_supabase_job
        data = {"status": "queued"}
        row = _to_supabase_job("j1", data)
        assert row["source_value"] == ""

    def test_source_type_always_prompt(self):
        from app.services.job_store import _to_supabase_job
        row = _to_supabase_job("j1", {})
        assert row["source_type"] == "prompt"

    def test_metadata_contains_full_data(self):
        from app.services.job_store import _to_supabase_job
        data = {"status": "running", "extra_field": "extra"}
        row = _to_supabase_job("j1", data)
        assert row["metadata"] == data

    def test_progress_defaults_to_zero(self):
        from app.services.job_store import _to_supabase_job
        row = _to_supabase_job("j1", {})
        assert row["progress"] == 0

    def test_request_prompt_takes_priority_over_payload(self):
        from app.services.job_store import _to_supabase_job
        data = {
            "request": {"prompt": "from request"},
            "payload": {"prompt": "from payload"},
        }
        row = _to_supabase_job("j1", data)
        assert row["source_value"] == "from request"


# ---------------------------------------------------------------------------
# save_job / load_job
# ---------------------------------------------------------------------------

class TestSaveLoadJob:
    def test_save_and_load(self, patched_store):
        from app.services.job_store import save_job, load_job

        data = {
            "status": "queued",
            "request": {"prompt": "test"},
            "progress_percent": 0,
        }
        save_job("j1", data)
        loaded = load_job("j1")
        assert loaded["status"] == "queued"
        assert loaded["job_id"] == "j1"

    def test_load_missing_job_raises(self, patched_store):
        from app.services.job_store import load_job

        with pytest.raises(FileNotFoundError):
            load_job("nonexistent")

    def test_load_merges_status_and_progress(self, patched_store):
        from app.services.job_store import save_job, load_job

        data = {"status": "running", "progress_percent": 40, "request": {}}
        save_job("j2", data)
        loaded = load_job("j2")
        assert loaded["status"] == "running"
        # progress_percent comes from row["metadata"]["progress_percent"] (the saved data)
        # but load_job overrides it with row.get("progress", 0) which is set from data["progress_percent"]
        assert "progress_percent" in loaded

    def test_overwrite_existing_job(self, patched_store):
        from app.services.job_store import save_job, load_job

        save_job("j3", {"status": "queued", "request": {}})
        save_job("j3", {"status": "completed", "request": {}})
        loaded = load_job("j3")
        assert loaded["status"] == "completed"


# ---------------------------------------------------------------------------
# append_event / load_events / load_events_after
# ---------------------------------------------------------------------------

class TestEvents:
    def test_append_and_load(self, patched_store):
        from app.services.job_store import append_event, load_events

        append_event("j1", "progress", "07_storyline", {"progress": 50})
        events = load_events("j1")
        assert len(events) == 1
        assert events[0]["event_type"] == "progress"

    def test_load_events_after_filters_correctly(self, patched_store):
        from app.services.job_store import append_event, load_events_after

        append_event("j1", "progress", "07_storyline", {"progress": 10})
        append_event("j1", "progress", "07_storyline", {"progress": 50})
        append_event("j1", "progress", "07_storyline", {"progress": 90})

        events = load_events_after("j1", after_id=1)
        assert len(events) == 2

    def test_load_events_after_zero_returns_all(self, patched_store):
        from app.services.job_store import append_event, load_events_after

        append_event("j1", "created", "queued", {})
        append_event("j1", "progress", "07_storyline", {"progress": 50})

        events = load_events_after("j1", after_id=0)
        assert len(events) == 2

    def test_events_isolated_by_job_id(self, patched_store):
        from app.services.job_store import append_event, load_events

        append_event("j1", "progress", "07_storyline", {})
        append_event("j2", "progress", "08_assets", {})

        j1_events = load_events("j1")
        j2_events = load_events("j2")
        assert len(j1_events) == 1
        assert len(j2_events) == 1
        assert j1_events[0]["stage"] == "07_storyline"
        assert j2_events[0]["stage"] == "08_assets"


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------

class TestCancelJob:
    def test_cancel_queued_job(self, patched_store):
        from app.services.job_store import save_job, cancel_job

        save_job("j1", {"status": "queued", "request": {}})
        result = cancel_job("j1")
        assert result is True

    def test_cancel_running_job(self, patched_store):
        from app.services.job_store import save_job, cancel_job

        save_job("j1", {"status": "running", "request": {}})
        result = cancel_job("j1")
        assert result is True

    def test_cancel_completed_job_returns_false(self, patched_store):
        from app.services.job_store import save_job, cancel_job

        save_job("j1", {"status": "completed", "request": {}})
        result = cancel_job("j1")
        assert result is False

    def test_cancel_missing_job_returns_false(self, patched_store):
        from app.services.job_store import cancel_job

        result = cancel_job("nonexistent")
        assert result is False

    def test_cancel_already_cancelled_returns_false(self, patched_store):
        from app.services.job_store import save_job, cancel_job

        save_job("j1", {"status": "cancelled", "request": {}})
        result = cancel_job("j1")
        assert result is False


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_empty_returns_empty_list(self, patched_store):
        from app.services.job_store import list_jobs
        assert list_jobs() == []

    def test_lists_all_jobs(self, patched_store):
        from app.services.job_store import save_job, list_jobs

        save_job("j1", {"status": "queued", "request": {}})
        save_job("j2", {"status": "running", "request": {}})
        jobs = list_jobs()
        assert len(jobs) == 2

    def test_each_job_has_job_id(self, patched_store):
        from app.services.job_store import save_job, list_jobs

        save_job("j1", {"status": "queued", "request": {}})
        jobs = list_jobs()
        assert all("job_id" in j for j in jobs)
