"""Feedback and notifications: auth gating, and that submitting feedback
notifies admins while resolving it notifies the submitter — the behavior
the original (removed) direct-Supabase implementation had, now rebuilt
server-side (see docs/security/frontend-changes-for-security-fixes.md F-5)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    yield
    api_app.dependency_overrides.clear()


def test_submit_feedback_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.post("/api/feedback", json={"feedback": "hello"})
    assert response.status_code == 401


def test_list_feedback_requires_admin(api_app, mock_settings):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.get("/api/feedback")
    assert response.status_code == 403


def test_submit_feedback_notifies_admins(api_app, mock_settings, monkeypatch):
    import app.db.repositories.user_repository as user_repository
    import app.db.repositories.feedback_repository as feedback_repository
    import app.services.notification_service as notification_service

    monkeypatch.setattr(
        user_repository.UserRepository, "get_by_id",
        lambda self, user_id: {"id": user_id, "username": "alice", "email_id": "alice@example.com"},
    )
    monkeypatch.setattr(
        feedback_repository.FeedbackRepository, "create",
        lambda self, data: {**data, "resolution_message": None, "resolved_at": None,
                             "resolved_by": None, "created_at": "2026-01-01T00:00:00Z",
                             "auto_reply": None, "auto_reply_category": None, "auto_reply_sent_at": None},
    )
    monkeypatch.setattr(feedback_repository.FeedbackRepository, "count_recent_by_user", lambda self, user_id, **kw: 0)
    notified = {}
    monkeypatch.setattr(
        notification_service, "notify_admins",
        lambda message, **kw: notified.update(message=message, kw=kw) or 1,
    )
    monkeypatch.setattr(notification_service, "notify_user", lambda user_id, message, **kw: {})

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/feedback", json={"feedback": "the export button is confusing"})

    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert body["email"] == "alice@example.com"
    assert body["user_id"] == "user-1"
    assert "alice" in notified["message"]


def test_resolve_feedback_notifies_submitter(api_app, mock_settings, monkeypatch):
    import app.db.repositories.feedback_repository as feedback_repository
    import app.services.notification_service as notification_service

    existing = {
        "id": "fb-1", "user_id": "submitter-1", "username": "alice", "email": "alice@example.com",
        "feedback": "x", "status": "open", "resolution_message": None,
        "resolved_at": None, "resolved_by": None, "created_at": "2026-01-01T00:00:00Z",
    }
    monkeypatch.setattr(feedback_repository.FeedbackRepository, "get", lambda self, feedback_id: existing)
    monkeypatch.setattr(
        feedback_repository.FeedbackRepository, "resolve",
        lambda self, feedback_id, *, resolution_message, resolved_by: {
            **existing, "status": "resolved", "resolution_message": resolution_message, "resolved_by": resolved_by,
        },
    )
    notified = {}
    monkeypatch.setattr(
        notification_service, "notify_user",
        lambda user_id, message, **kw: notified.update(user_id=user_id, message=message) or {},
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as client:
        response = client.post("/api/feedback/fb-1/resolve", json={"resolution_message": "Fixed it"})

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    assert notified["user_id"] == "submitter-1"


def test_submit_feedback_over_the_rate_limit_is_rejected(api_app, mock_settings, monkeypatch):
    import app.db.repositories.user_repository as user_repository
    import app.db.repositories.feedback_repository as feedback_repository

    monkeypatch.setattr(
        user_repository.UserRepository, "get_by_id",
        lambda self, user_id: {"id": user_id, "username": "alice", "email_id": "alice@example.com"},
    )
    monkeypatch.setattr(feedback_repository.FeedbackRepository, "count_recent_by_user", lambda self, user_id, **kw: 999)

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/feedback", json={"feedback": "spam"})
    assert response.status_code == 429


def test_submit_feedback_auto_reply_sets_columns_and_notifies_user(api_app, mock_settings, monkeypatch):
    import app.db.repositories.user_repository as user_repository
    import app.db.repositories.feedback_repository as feedback_repository
    import app.services.notification_service as notification_service
    import app.services.feedback_auto_reply as feedback_auto_reply
    from app.services.feedback_auto_reply import AutoReply

    monkeypatch.setattr(
        user_repository.UserRepository, "get_by_id",
        lambda self, user_id: {"id": user_id, "username": "alice", "email_id": "alice@example.com"},
    )
    monkeypatch.setattr(feedback_repository.FeedbackRepository, "count_recent_by_user", lambda self, user_id, **kw: 0)
    monkeypatch.setattr(
        feedback_repository.FeedbackRepository, "create",
        lambda self, data: {**data, "resolution_message": None, "resolved_at": None,
                             "resolved_by": None, "created_at": "2026-01-01T00:00:00Z",
                             "auto_reply": None, "auto_reply_category": None, "auto_reply_sent_at": None},
    )
    set_calls = {}
    monkeypatch.setattr(
        feedback_repository.FeedbackRepository, "set_auto_reply",
        lambda self, feedback_id, *, category, reply: set_calls.update(feedback_id=feedback_id, category=category, reply=reply) or None,
    )
    monkeypatch.setattr(feedback_auto_reply, "generate", lambda user_id, feedback: AutoReply(category="bug", reply="It's on our list."))
    monkeypatch.setattr(notification_service, "notify_admins", lambda message, **kw: 1)
    notified_user = {}
    monkeypatch.setattr(
        notification_service, "notify_user",
        lambda user_id, message, **kw: notified_user.update(user_id=user_id, message=message) or {},
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/feedback", json={"feedback": "the export button is broken"})

    assert response.status_code == 201
    assert set_calls == {"feedback_id": response.json()["id"], "category": "bug", "reply": "It's on our list."}
    assert notified_user == {"user_id": "user-1", "message": "It's on our list."}


def test_submit_feedback_falls_back_to_canned_reply_on_llm_failure(api_app, mock_settings, monkeypatch):
    import app.db.repositories.user_repository as user_repository
    import app.db.repositories.feedback_repository as feedback_repository
    import app.services.notification_service as notification_service
    import app.services.feedback_auto_reply as feedback_auto_reply

    monkeypatch.setattr(
        user_repository.UserRepository, "get_by_id",
        lambda self, user_id: {"id": user_id, "username": "alice", "email_id": "alice@example.com"},
    )
    monkeypatch.setattr(feedback_repository.FeedbackRepository, "count_recent_by_user", lambda self, user_id, **kw: 0)
    monkeypatch.setattr(
        feedback_repository.FeedbackRepository, "create",
        lambda self, data: {**data, "resolution_message": None, "resolved_at": None,
                             "resolved_by": None, "created_at": "2026-01-01T00:00:00Z",
                             "auto_reply": None, "auto_reply_category": None, "auto_reply_sent_at": None},
    )

    def boom(user_id, feedback):
        raise RuntimeError("host down")
    monkeypatch.setattr(feedback_auto_reply, "generate", boom)
    monkeypatch.setattr(notification_service, "notify_admins", lambda message, **kw: 1)
    notified_user = {}
    monkeypatch.setattr(
        notification_service, "notify_user",
        lambda user_id, message, **kw: notified_user.update(user_id=user_id, message=message) or {},
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/feedback", json={"feedback": "the export button is broken"})

    assert response.status_code == 201
    assert notified_user == {"user_id": "user-1", "message": feedback_auto_reply.CANNED_FALLBACK}


def test_notifications_are_self_scoped(api_app, mock_settings, monkeypatch):
    import app.db.repositories.notification_repository as notification_repository

    written = {}
    monkeypatch.setattr(
        notification_repository.NotificationRepository, "create",
        lambda self, data: written.update(data) or dict(data),
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/notifications", json={"message": "hi", "user_id": "someone-else"})
    assert response.status_code == 201
    assert response.json()["user_id"] == "user-1"
    assert written["user_id"] == "user-1"


def test_mark_notification_read_404_for_missing(api_app, mock_settings, monkeypatch):
    import app.db.repositories.notification_repository as notification_repository

    monkeypatch.setattr(notification_repository.NotificationRepository, "mark_read", lambda self, nid, uid: False)

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.post("/api/notifications/does-not-exist/read")
    assert response.status_code == 404
