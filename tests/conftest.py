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
# Keep model-host health records out of a real Redis during tests.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_dependency_state_redis(monkeypatch):
    """Force app.core.dependency_state onto its in-memory path.

    llm_service._post_with_retries records model-host health on every call,
    so without this a test run publishes its fake hosts ("http://x", ...) to
    whatever Redis the developer has running, and they then show up in
    /api/health for the next 15 minutes.
    """
    from app.core import dependency_state

    monkeypatch.setattr(dependency_state, "_get_redis", lambda: None)
    dependency_state.reset_for_tests()
    yield
    dependency_state.reset_for_tests()


@pytest.fixture(autouse=True)
def _no_intent_router_llm(monkeypatch):
    """The intent router never reaches a model host in tests: it falls back to the
    rules unless a test stubs intent_router._ask itself."""
    from app.services import intent_router

    def unavailable(*args, **kwargs):
        raise RuntimeError("intent router LLM is not available in tests")

    monkeypatch.setattr(intent_router, "_ask", unavailable)


@pytest.fixture(autouse=True)
def _no_feedback_auto_reply_llm(monkeypatch):
    """The feedback auto-reply never reaches a model host in tests: it falls
    back to the canned acknowledgment unless a test stubs the internal _ask."""
    from app.services import feedback_auto_reply

    def unavailable(*args, **kwargs):
        raise RuntimeError("feedback auto-reply LLM is not available in tests")

    monkeypatch.setattr(feedback_auto_reply, "_ask", unavailable)


@pytest.fixture(autouse=True)
def _no_rate_limit_redis(monkeypatch):
    """Force app.core.rate_limit onto its in-process fallback path.

    TokenBucket is Redis-backed so the limit is shared across replicas; left
    unpatched, every rate-limited endpoint hit in a test run would consume
    real counters against whatever Redis the developer has running, and
    those counters persist for the window (up to 300s) — polluting the next
    test run's results. Tests that want to exercise the Redis path itself
    patch app.core.rate_limit._get_redis locally instead.
    """
    from app.core import rate_limit
    from app.services import auth_service

    monkeypatch.setattr(rate_limit, "_get_redis", lambda: None)
    # The in-process buckets are module globals; start every test with them
    # empty so limits reached in one test never leak into the next.
    for value in vars(auth_service).values():
        if isinstance(value, rate_limit.TokenBucket):
            value._tokens.clear()
            value._last.clear()


@pytest.fixture(autouse=True)
def user_states(monkeypatch):
    """Stand-in for the account row get_current_user checks tokens against.

    Unknown users are an active "general" account at token version 0, which
    matches a token minted with the defaults. Tests that need an admin, a
    bumped version or a deactivated account set an entry here.
    """
    from app.core import auth

    states: dict = {}
    monkeypatch.setattr(
        auth, "_load_user_state",
        lambda user_id: states.get(user_id, auth.UserState(token_version=0, persona="general", active=True)),
    )
    auth._user_state_cache.clear()
    auth._user_state_generation.clear()
    yield states
    auth._user_state_cache.clear()
    auth._user_state_generation.clear()


class _FakeRecords:
    """In-memory stand-in for the prompt_resolutions / llm_call_payloads repositories."""

    def __init__(self):
        self.resolutions: dict[str, dict] = {}
        self.payloads: list[dict] = []

    # PromptResolutionRepository
    def create(self, data):
        if "request_text" in data:
            self.payloads.append(dict(data))
        else:
            self.resolutions[data["job_id"]] = dict(data)

    def update_by_job(self, job_id, *, fields=None, merge=None, only_if_outcome=None):
        row = self.resolutions.get(job_id)
        if row is None or (only_if_outcome and row.get("layer_outcome") != only_if_outcome):
            return
        row.update(fields or {})
        for key, value in (merge or {}).items():
            row[key] = {**row.get(key, {}), **value}

    def finish_completed(self, job_id):
        verdict = ((self.resolutions.get(job_id) or {}).get("coverage") or {}).get("verdict")
        self.update_by_job(job_id, fields={"layer_outcome": verdict if verdict in ("covered", "partial")
                                           else "completed"}, only_if_outcome="in_progress")


@pytest.fixture(autouse=True)
def processing_records_store(monkeypatch):
    """Keep request and LLM-exchange records out of any real Postgres."""
    from app.services import processing_records

    fake = _FakeRecords()
    monkeypatch.setattr(processing_records, "_resolutions", lambda: fake)
    monkeypatch.setattr(processing_records, "_payloads", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# mock_settings — patches app.core.settings.settings with safe in-test values
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path, monkeypatch):
    """Patch app.core.settings.settings with safe in-test values."""
    from app.core import settings as settings_module

    fake = MagicMock()
    fake.app_name = "AI4IA Demo Video Studio"
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
    # Read at import time by client_ip/auth_service; must be real values, not mocks.
    from app.core.settings import Settings

    _defaults = Settings.model_fields
    fake.trusted_proxy_cidrs = list(_defaults["trusted_proxy_cidrs"].default)
    fake.jwt_min_key_bytes = _defaults["jwt_min_key_bytes"].default
    fake.job_sweeper_enabled = False
    fake.worker_queue = "video_jobs_test"
    for _name in ("job_heartbeat_interval_seconds", "job_stale_after_seconds", "job_max_runtime_seconds",
                  "job_queued_max_age_seconds", "job_sweep_interval_seconds",
                  "celery_visibility_timeout_seconds", "job_max_worker_loss_retries"):
        setattr(fake, _name, _defaults[_name].default)
    fake.video_job_max_running_per_user = 1
    fake.video_job_max_queued_per_user = 3
    fake.video_job_max_active = 50
    for _name in ("otp_requests_per_email_ip_per_hour", "otp_requests_per_email_per_hour",
                  "otp_requests_per_ip_per_hour", "otp_verifies_per_email_ip_per_hour",
                  "otp_verifies_per_email_per_hour", "otp_verifies_per_ip_per_hour",
                  "otp_codes_per_email_ip_per_day", "otp_codes_per_email_per_day",
                  "otp_verifies_per_email_ip_per_day", "otp_verifies_per_email_per_day",
                  "otp_max_attempts", "otp_ttl_minutes",
                  "refresh_reuse_grace_seconds"):
        setattr(fake, _name, _defaults[_name].default)
    # Interactive-endpoint model gate (api/deps.py's require_models) — off by
    # default so a MagicMock doesn't leak into a real int/bool comparison
    # (e.g. `wait_seconds > 0` against model_gate_inline_wait_seconds) for any
    # test that hits a gated route without explicitly exercising the gate.
    fake.model_gate_enabled = False
    fake.model_gate_inline_wait_seconds = 0
    fake.prompt_max_chars = 2000
    fake.feedback_auto_reply_enabled = True
    fake.feedback_auto_reply_model = ""
    fake.feedback_auto_reply_timeout_seconds = 20
    fake.feedback_submissions_per_user_per_hour = 10

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

    def _create_job_within_quota(job_id, data, *, user_id, max_running, max_queued, max_active,
                                 dispatch_id=None, worker_queue=None):
        active = [j for j in jobs.values() if j.get("status") in ("queued", "running")]
        mine = [j for j in active if j.get("user_id") == user_id]
        if sum(j["status"] == "running" for j in mine) >= max_running:
            return "user_running"
        if sum(j["status"] == "queued" for j in mine) >= max_queued:
            return "user_queued"
        if len(active) >= max_active:
            return "active"
        jobs[job_id] = copy.deepcopy(data)
        jobs[job_id].update(dispatch_id=dispatch_id, worker_queue=worker_queue, attempts=0)
        return None

    def _finish_job(job_id, status, owner):
        job = jobs.get(job_id)
        if job is None:
            return False
        if job.get("status") not in ("queued", "running", status) or job.get("lease_owner") != owner:
            return job.get("dispatch_id") is None
        job["status"] = status
        job["lease_owner"] = None
        return True

    def _claim_job(job_id, *, dispatch_id, owner, worker_queue, max_retries=0):
        job = jobs.get(job_id)
        if (job is None or job.get("status") != "queued" or job.get("attempts", 0) > max_retries
                or job.get("dispatch_id") != dispatch_id):
            return False
        job.update(status="running", lease_owner=owner, attempts=job.get("attempts", 0) + 1, heartbeat_age=0.0,
                   worker_queue=job.get("worker_queue") or worker_queue)
        return True

    def _take_over_job(job_id, *, dispatch_id, dead_owner, owner, max_retries):
        job = jobs.get(job_id)
        if (job is None or job.get("status") != "running" or job.get("lease_owner") != dead_owner
                or job.get("dispatch_id") != dispatch_id or job.get("attempts", 0) > max_retries):
            return False
        job.update(lease_owner=owner, attempts=job.get("attempts", 0) + 1, heartbeat_age=0.0,
                   current_stage="queued", stage_status={})
        return True

    def _interrupted_stages(job_id):
        # Like the real table, which keeps no stage column.
        return [e["payload"].get("stage") for e in events.get(job_id, []) if e["event_type"] == "interrupted"]

    def _job_lease_state(job_id):
        job = jobs.get(job_id)
        if job is None:
            return None
        return {"status": job.get("status"), "dispatch_id": job.get("dispatch_id"),
                "lease_owner": job.get("lease_owner"), "attempts": job.get("attempts", 0),
                "heartbeat_age": job.get("heartbeat_age"), "active_pid": job.get("active_pid"),
                "current_stage": job.get("current_stage") or "unknown",
                "active_host": job.get("active_host")}

    def _heartbeat_job(job_id, owner):
        job = jobs.get(job_id)
        return bool(job and job.get("lease_owner") == owner and job.get("status") == "running")

    def _fail_job_if_lease(job_id, *, expected_owner, error):
        job = jobs.get(job_id)
        if not job or job.get("status") != "running" or job.get("lease_owner") != expected_owner:
            return False
        job.update(status="failed", lease_owner=None, error=error)
        return True

    def _list_user_jobs(user_id, limit, offset):
        mine = sorted((j for j in jobs.values() if j.get("user_id") == user_id),
                      key=lambda j: j.get("created_at") or "", reverse=True)
        return [copy.deepcopy(j) for j in mine[offset:offset + limit]], len(mine)

    def _queue_position(job_id):
        created = jobs[job_id].get("created_at") or ""
        return sum(1 for j in jobs.values() if j.get("status") == "queued" and (j.get("created_at") or "") <= created)

    import app.services.job_store as js
    import app.services.video_job_runner as vjr
    import app.services.media_registry as media_registry

    monkeypatch.setattr(js, "create_job_within_quota", _create_job_within_quota)
    for _name, _fn in (("finish_job", _finish_job), ("claim_job", _claim_job),
                       ("job_lease_state", _job_lease_state), ("heartbeat_job", _heartbeat_job),
                       ("fail_job_if_lease", _fail_job_if_lease), ("take_over_job", _take_over_job),
                       ("interrupted_stages", _interrupted_stages)):
        monkeypatch.setattr(js, _name, _fn)
    monkeypatch.setattr(vjr, "finish_job", _finish_job)
    monkeypatch.setattr(js, "list_user_jobs", _list_user_jobs)
    monkeypatch.setattr(js, "queue_position", _queue_position)

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

    # app.services.media_registry does `from app.services.job_store import
    # load_job` at module top level (a plain name bound once, at whichever
    # test first imports it in this process) — the patch on js.load_job above
    # doesn't reach that already-bound copy. Rebind it directly too.
    monkeypatch.setattr(media_registry, "load_job", _load_job)

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
        **extra,
    }
