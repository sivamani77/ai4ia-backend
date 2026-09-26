"""Auth coverage for /api/provenance/feed — an admin-only, cross-user
aggregate view (see the module docstring on provenance.py). Previously
gated only by the optional shared API key, not admin role, letting any
caller holding that key see every user's job_id/user_id/decision records.

See local/reports/AI4IA_Production_Readiness_Audit.xlsx, Critical Finding #6.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    """api_app returns the process-wide main.app singleton — clear overrides
    so they don't leak into other test modules in the same session."""
    yield
    api_app.dependency_overrides.clear()


def test_provenance_feed_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.get("/api/provenance/feed")
    assert response.status_code == 401


def test_provenance_feed_rejects_non_admin(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.get("/api/provenance/feed")
    assert response.status_code == 403


def test_provenance_feed_allows_admin_and_returns_cross_user_data(api_app, mock_settings, monkeypatch):
    import app.db.repositories.decision_record_repository as decision_record_repository
    import app.db.repositories.llm_trace_repository as llm_trace_repository

    records = [
        {"id": "dr1", "job_id": "vjob_1", "insight_id": None, "user_id": "user-a", "subject": "documentary_generation", "outcome": "completed", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "dr2", "job_id": "vjob_2", "insight_id": None, "user_id": "user-b", "subject": "documentary_generation", "outcome": "completed", "created_at": "2026-01-02T00:00:00Z"},
    ]
    monkeypatch.setattr(
        decision_record_repository.DecisionRecordRepository, "list_all", lambda self, limit=100, offset=0: records
    )
    monkeypatch.setattr(
        llm_trace_repository.LLMTraceRepository, "list_all", lambda self, limit=200, offset=0: []
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/provenance/feed")

    assert response.status_code == 200
    body = response.json()
    returned_user_ids = {item["user_id"] for item in body["items"]}
    assert returned_user_ids == {"user-a", "user-b"}
