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
        lambda self, limit, offset: fake_rows[:limit],
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/decision_records", params={"limit": 2, "offset": 2})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "table": "decision_records", "limit": 2, "offset": 2, "matched": 4, "count": 2,
        "rows": [{"id": "rec-2"}, {"id": "rec-3"}],
    }


def test_search_filters_case_insensitive_substring(api_app, mock_settings, monkeypatch):
    import app.db.repositories.job_repository as job_repository

    fake_rows = [
        {"id": "vjob_abc", "status": "completed"},
        {"id": "vjob_xyz", "status": "failed"},
        {"id": "other_abc", "status": "completed"},
    ]
    monkeypatch.setattr(job_repository.JobRepository, "list_jobs", lambda self: fake_rows)

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get(
            "/api/admin/data/tables/jobs",
            params={"search_field": "id", "search_value": "VJOB"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] == 2
    assert {r["id"] for r in body["rows"]} == {"vjob_abc", "vjob_xyz"}


def test_search_requires_both_params_together(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/tables/jobs", params={"search_field": "id"})
    assert response.status_code == 400


def test_job_full_joins_events_decision_record_and_traces(api_app, mock_settings, monkeypatch):
    import app.db.repositories.decision_record_repository as decision_record_repository
    import app.db.repositories.event_repository as event_repository
    import app.db.repositories.job_repository as job_repository
    import app.db.repositories.llm_trace_repository as llm_trace_repository

    monkeypatch.setattr(job_repository.JobRepository, "get_job", lambda self, record_id: {"id": record_id})
    monkeypatch.setattr(event_repository.EventRepository, "get_events", lambda self, job_id: [{"event": "started"}])
    monkeypatch.setattr(
        decision_record_repository.DecisionRecordRepository, "get_by_job",
        lambda self, job_id: {"id": "rec-1"},
    )
    monkeypatch.setattr(llm_trace_repository.LLMTraceRepository, "list_by_job", lambda self, job_id: [{"id": "trace-1"}])

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/jobs/vjob_123/full")

    assert response.status_code == 200
    assert response.json() == {
        "job": {"id": "vjob_123"},
        "events": [{"event": "started"}],
        "decision_record": {"id": "rec-1"},
        "llm_traces": [{"id": "trace-1"}],
    }


def test_job_full_404_when_job_missing(api_app, mock_settings, monkeypatch):
    import app.db.repositories.job_repository as job_repository

    monkeypatch.setattr(job_repository.JobRepository, "get_job", lambda self, record_id: None)

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.get("/api/admin/data/jobs/does-not-exist/full")
    assert response.status_code == 404


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
