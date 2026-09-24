"""Auth gating and shape checks for the read-only admin data explorer
(app/api/admin_data.py). Repository calls are stubbed — this tests the
router's own logic (allow-list, pagination plumbing, unsupported-table
handling), not the repositories themselves, which have their own tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    yield
    api_app.dependency_overrides.clear()


def test_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables")
    assert response.status_code == 401


def test_rejects_non_admin(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables")
    assert response.status_code == 403


def test_lists_expected_tables(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables")
    assert response.status_code == 200
    assert set(response.json()["tables"]) == {
        "fragments", "videos", "storylines", "insights",
        "jobs", "ingest_jobs", "decision_records", "llm_call_traces",
    }


def test_list_rows_uses_repository_and_paginates(api_app, mock_settings, monkeypatch):
    import app.db.repositories.decision_record_repository as decision_record_repository

    fake_rows = [{"id": f"rec-{i}"} for i in range(5)]
    monkeypatch.setattr(
        decision_record_repository.DecisionRecordRepository, "list_all",
        lambda self, limit, offset: fake_rows[offset:offset + limit],
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/decision_records", params={"limit": 2, "offset": 2})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "table": "decision_records", "limit": 2, "offset": 2, "count": 2,
        "rows": [{"id": "rec-2"}, {"id": "rec-3"}],
    }


def test_get_row_not_supported_for_llm_call_traces(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/llm_call_traces/whatever")
    assert response.status_code == 400


def test_get_row_404_when_missing(api_app, mock_settings, monkeypatch):
    import app.db.repositories.job_repository as job_repository

    monkeypatch.setattr(job_repository.JobRepository, "get_job", lambda self, record_id: None)

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/jobs/does-not-exist")
    assert response.status_code == 404


def test_unknown_table_404(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/not_a_real_table")
    assert response.status_code == 404
