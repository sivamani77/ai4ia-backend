"""Auth + rate-limit coverage for the AI-cost endpoints that were previously
reachable with only the optional shared API key (or, for insights.py, with
no auth check at all): ingest.py, compare.py, concept_mindmap.py, insights.py.

See docs/AI4IA_Production_Readiness_Audit.xlsx, Critical Findings #3/#7 and
Checklist 5.1/5.3.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user
from app.core.rate_limit import TokenBucket
from app.core import metrics as _metrics


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    """`api_app` returns the process-wide `main.app` singleton (module-level,
    cached across tests), so a dependency_overrides entry set by one test
    would otherwise leak into every test that runs after it in the same
    session."""
    yield
    api_app.dependency_overrides.clear()


def _counter_value(counter, **labels) -> float:
    metric = counter.labels(**labels) if labels else counter
    return metric._value.get()  # type: ignore[attr-defined]


def _override_user(api_app, *, user_id: str, persona: str) -> None:
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id=user_id, persona=persona)


# ---------------------------------------------------------------------------
# ingest.py — admin-only, rate-limited triggers
# ---------------------------------------------------------------------------

def test_ingest_sync_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.post("/api/ingest/sync", json={})
    assert response.status_code == 401


def test_ingest_sync_rejects_non_admin(api_app, mock_settings):
    _override_user(api_app, user_id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/ingest/sync", json={})
    assert response.status_code == 403


def test_ingest_sync_admin_allowed_then_rate_limited(api_app, mock_settings, monkeypatch):
    import app.api.ingest as ingest_module

    _override_user(api_app, user_id="admin-1", persona="admin")
    monkeypatch.setattr(ingest_module, "_is_job_running", lambda: None)
    monkeypatch.setattr(ingest_module, "create_ingest_job", lambda **kw: "ijob_test")
    monkeypatch.setattr(ingest_module, "start_ingest_job_background", lambda job_id: None)
    monkeypatch.setattr(ingest_module, "_trigger_limiter", TokenBucket(rate=1, per=300.0))

    before = _counter_value(_metrics.rate_limit_rejections_total, limiter="ingest_trigger")

    with TestClient(api_app) as client:
        first = client.post("/api/ingest/sync", json={})
        second = client.post("/api/ingest/sync", json={})

    assert first.status_code == 202
    assert second.status_code == 429
    assert _counter_value(_metrics.rate_limit_rejections_total, limiter="ingest_trigger") == before + 1


# ---------------------------------------------------------------------------
# compare.py — authenticated user, rate-limited
# ---------------------------------------------------------------------------

def test_compare_speakers_requires_auth(api_app, mock_settings, monkeypatch):
    from app.services import model_gate

    monkeypatch.setattr(model_gate, "ensure_ready", lambda *a, **k: {key: True for key in a})

    with TestClient(api_app) as client:
        response = client.post(
            "/api/compare/speakers",
            json={"speakers": ["Alice", "Bob"]},
        )
    assert response.status_code == 401


def test_compare_speakers_allowed_then_rate_limited(api_app, mock_settings, monkeypatch):
    import app.api.compare as compare_module
    from app.services import model_gate

    monkeypatch.setattr(model_gate, "ensure_ready", lambda *a, **k: {key: True for key in a})
    _override_user(api_app, user_id="user-1", persona="general")
    monkeypatch.setattr(compare_module.comparison_service, "compare_speakers", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(compare_module, "_compare_limiter", TokenBucket(rate=1, per=60.0))

    before = _counter_value(_metrics.rate_limit_rejections_total, limiter="compare")

    with TestClient(api_app) as client:
        first = client.post("/api/compare/speakers", json={"speakers": ["Alice", "Bob"]})
        second = client.post("/api/compare/speakers", json={"speakers": ["Alice", "Bob"]})

    assert first.status_code == 200
    assert second.status_code == 429
    assert _counter_value(_metrics.rate_limit_rejections_total, limiter="compare") == before + 1


# ---------------------------------------------------------------------------
# concept_mindmap.py — authenticated user, rate-limited
# ---------------------------------------------------------------------------

def test_concept_mindmap_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.post("/api/v1/concept-mindmap", json={"query": "What is AI access?"})
    assert response.status_code == 401


def test_concept_mindmap_allowed_then_rate_limited(api_app, mock_settings, monkeypatch):
    import app.api.concept_mindmap as mindmap_module

    _override_user(api_app, user_id="user-1", persona="general")
    monkeypatch.setattr(mindmap_module, "generate_concept_mindmap", lambda **kw: {"ok": True})
    monkeypatch.setattr(mindmap_module, "_mindmap_limiter", TokenBucket(rate=1, per=60.0))

    before = _counter_value(_metrics.rate_limit_rejections_total, limiter="concept_mindmap")

    with TestClient(api_app) as client:
        first = client.post("/api/v1/concept-mindmap", json={"query": "What is AI access?"})
        second = client.post("/api/v1/concept-mindmap", json={"query": "What is AI access?"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert _counter_value(_metrics.rate_limit_rejections_total, limiter="concept_mindmap") == before + 1


# ---------------------------------------------------------------------------
# insights.py — auth consistency fix + rate-limited /generate
# ---------------------------------------------------------------------------

def test_insights_list_requires_auth(api_app, mock_settings):
    """Previously api_list_insights had no auth dependency at all."""
    with TestClient(api_app) as client:
        response = client.get("/api/insights")
    assert response.status_code == 401


def test_insight_sources_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.get("/api/insights/some-id/sources")
    assert response.status_code == 401


def test_insights_generate_allowed_then_rate_limited(api_app, mock_settings, monkeypatch):
    import app.api.insights as insights_module
    from app.services import model_gate

    monkeypatch.setattr(model_gate, "ensure_ready", lambda *a, **k: {key: True for key in a})
    _override_user(api_app, user_id="user-1", persona="general")
    monkeypatch.setattr(insights_module.insight_service, "generate_insight", lambda *a, **k: {"id": "insight-1"})
    monkeypatch.setattr(insights_module, "_insight_limiter", TokenBucket(rate=1, per=60.0))

    before = _counter_value(_metrics.rate_limit_rejections_total, limiter="insights_generate")

    with TestClient(api_app) as client:
        first = client.post("/api/insights/generate", json={"query": "What is AI access?"})
        second = client.post("/api/insights/generate", json={"query": "What is AI access?"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert (
        _counter_value(_metrics.rate_limit_rejections_total, limiter="insights_generate") == before + 1
    )
