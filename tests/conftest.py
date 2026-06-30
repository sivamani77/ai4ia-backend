"""Shared pytest fixtures for the ai4ia-demo-pipeline test suite.

All tests live under tests/ at the repo root.  The backend package is made
importable by adding backend/ to sys.path via pytest's conftest mechanism
(see pyproject.toml [tool.pytest.ini_options] pythonpath setting).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path so `import app.*` works without installing.
# ---------------------------------------------------------------------------
_BACKEND = Path(__file__).parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ---------------------------------------------------------------------------
# mock_settings — patches app.core.settings.settings with safe in-test values
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path, monkeypatch):
    """Patch app.core.settings.settings with safe in-test values."""
    from app.core import settings as settings_module

    fake = MagicMock()
    fake.app_name = "AI4IA Demo Video Studio"
    fake.supabase_url = "https://test.supabase.co"
    fake.supabase_service_key = "test-key"
    fake.redis_url = "redis://localhost:6379/0"
    fake.project_root = tmp_path
    fake.ai4ia_local_exports_dir = tmp_path / "exports"
    fake.ai4ia_local_assets_dir = tmp_path / "assets"
    fake.ai4ia_local_jobs_dir = tmp_path / "jobs"
    fake.ai4ia_object_storage_bucket = "test-bucket"
    fake.ai4ia_object_storage_endpoint = ""
    fake.ai4ia_object_storage_region = ""
    fake.ai4ia_object_storage_access_key = ""
    fake.ai4ia_object_storage_secret_key = ""
    fake.ollama_base_url = "http://localhost:11434"
    fake.speaker_extraction_model = "llama3.1:8b"
    fake.insight_model = "llama3.1:8b"
    fake.chroma_mode = "local"
    fake.chroma_host = "localhost"
    fake.chroma_port = 8000
    fake.chroma_ssl = False
    fake.chroma_auth_token = ""
    fake.chroma_collection = "fragments"
    fake.api_key = ""
    fake.ai4ia_storage_backend = "local"
    fake.allowed_origins = ["*"]
    fake.validate_deployment_env.return_value = []

    monkeypatch.setattr(settings_module, "settings", fake)
    return fake


# ---------------------------------------------------------------------------
# mock_job_store — in-memory replacement for Supabase-backed job/event store
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_job_store(monkeypatch):
    """Replace job_store I/O with in-memory dicts."""
    jobs: dict[str, dict[str, Any]] = {}
    events: dict[str, list[dict[str, Any]]] = {}

    def _save_job(job_id: str, data: dict[str, Any]) -> None:
        jobs[job_id] = copy.deepcopy(data)

    def _load_job(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise FileNotFoundError(f"Job not found: {job_id}")
        return copy.deepcopy(jobs[job_id])

    def _append_event(job_id: str, event_type: str, stage: str, payload: dict[str, Any]) -> None:
        events.setdefault(job_id, []).append({
            "id": len(events.get(job_id, [])) + 1,
            "job_id": job_id,
            "event_type": event_type,
            "stage": stage,
            "payload": payload,
        })

    def _load_events(job_id: str) -> list[dict[str, Any]]:
        return list(events.get(job_id, []))

    def _load_events_after(job_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return [e for e in events.get(job_id, []) if e["id"] > after_id]

    def _list_jobs() -> list[dict[str, Any]]:
        return [copy.deepcopy(j) for j in jobs.values()]

    def _cancel_job(job_id: str) -> bool:
        if job_id not in jobs:
            return False
        job = jobs[job_id]
        if job.get("status") not in ("queued", "running"):
            return False
        job["status"] = "cancelled"
        return True

    import app.services.job_store as js
    import app.services.video_job_runner as vjr

    monkeypatch.setattr(js, "save_job", _save_job)
    monkeypatch.setattr(js, "load_job", _load_job)
    monkeypatch.setattr(js, "append_event", _append_event)
    monkeypatch.setattr(js, "load_events", _load_events)
    monkeypatch.setattr(js, "load_events_after", _load_events_after)
    monkeypatch.setattr(js, "list_jobs", _list_jobs)
    monkeypatch.setattr(js, "cancel_job", _cancel_job)

    monkeypatch.setattr(vjr, "save_job", _save_job)
    monkeypatch.setattr(vjr, "load_job", _load_job)
    monkeypatch.setattr(vjr, "append_event", _append_event)
    monkeypatch.setattr(vjr, "list_jobs", _list_jobs)

    yield {"jobs": jobs, "events": events}


# ---------------------------------------------------------------------------
# sample_job / sample_request
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_job() -> dict[str, Any]:
    return {
        "job_id": "vjob_test_001",
        "status": "queued",
        "progress_percent": 0,
        "current_stage": "queued",
        "stage_status": {},
        "paths": {
            "storyline_path": None,
            "validated_storyline_path": None,
            "asset_manifest_path": None,
            "export_manifest_path": None,
            "video_path": None,
            "srt_path": None,
            "vtt_path": None,
        },
        "request": {"prompt": "test prompt"},
        "warnings": [],
        "error": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "started_at": None,
        "completed_at": None,
    }


@pytest.fixture
def sample_request() -> dict[str, Any]:
    return {
        "prompt": "AI in healthcare documentary",
        "target_duration": 150,
        "style": "documentary",
        "validation_mode": "iterative_llm",
        "visual_mode": "generated_assets",
        "subtitle_mode": "external",
        "max_validation_iterations": 3,
        "burn_subtitles": False,
    }


# ---------------------------------------------------------------------------
# Minimal fragment factory
# ---------------------------------------------------------------------------

def make_fragment(
    fragment_id: str = "frag_001",
    speaker: str = "Alice",
    text: str = "AI is transforming the world.",
    theme: str = "T1",
    chapter: str = "Ch 1",
    start: float = 0.0,
    end: float = 30.0,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "fragment_id": fragment_id,
        "speaker": speaker,
        "speaker_attribution": f"{speaker} — AI Researcher",
        "text": text,
        "summary": text[:50],
        "primary_theme": theme,
        "primary_chapter": chapter,
        "start": start,
        "end": end,
        "duration": end - start,
        "quote_strength": 0.8,
        "clarity_score": 0.9,
        "source_video": "0001_abc123_speaker",
        "edition": "AI4IA 2025",
        "year": 2025,
        "keywords": ["AI", "ethics"],
        "stream_url": f"/api/archive/videos/abc123/stream",
        **extra,
    }
