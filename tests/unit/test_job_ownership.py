"""Job ownership checks deny jobs that have no owner (C-14)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.auth import CurrentUser

OWNER = CurrentUser(id="user-1", persona="general")
ADMIN = CurrentUser(id="admin-1", persona="admin")


@pytest.fixture(params=["app.api.video_jobs", "app.api.video_media"])
def assert_owns(request):
    import importlib

    return importlib.import_module(request.param)._assert_owns


def test_owner_is_allowed(assert_owns):
    assert_owns({"job_id": "j1", "user_id": "user-1"}, OWNER)


@pytest.mark.parametrize("user_id", ["user-2", None, ""])
def test_other_or_missing_owner_is_not_found(assert_owns, user_id):
    with pytest.raises(HTTPException) as exc:
        assert_owns({"job_id": "j1", "user_id": user_id}, OWNER)
    assert exc.value.status_code == 404


def test_admin_can_open_an_ownerless_job(assert_owns):
    assert_owns({"job_id": "j1", "user_id": None}, ADMIN)


def test_hard_delete_user_removes_their_insights(monkeypatch):
    """Deleting an account also deletes the insights it owns (C-01)."""
    from unittest.mock import MagicMock

    import app.services.admin_user_service as svc
    import app.db.repositories.insight_repository as insight_repo
    import app.db.repositories.llm_trace_repository as trace_repo
    import app.db.repositories.decision_record_repository as record_repo
    import app.services.processing_records as processing_records

    users = MagicMock()
    users.get_by_id.return_value = {"id": "user-1"}
    jobs = MagicMock()
    jobs.list_by_user.return_value = []
    sessions = MagicMock()
    sessions.delete_all_for_user.return_value = 0
    insights = MagicMock()

    monkeypatch.setattr(svc, "get_user_repository", lambda: users)
    monkeypatch.setattr(svc, "get_job_repository", lambda: jobs)
    monkeypatch.setattr(svc, "get_event_repository", MagicMock)
    monkeypatch.setattr(svc, "get_session_repository", lambda: sessions)
    monkeypatch.setattr(insight_repo, "InsightRepository", lambda: insights)
    monkeypatch.setattr(trace_repo, "LLMTraceRepository", MagicMock)
    monkeypatch.setattr(record_repo, "DecisionRecordRepository", MagicMock)
    monkeypatch.setattr(processing_records, "delete_for_user", lambda user_id: None)

    svc.hard_delete_user("user-1")

    insights.delete_by_user.assert_called_once_with("user-1")
    users.delete_user.assert_called_once_with("user-1")
