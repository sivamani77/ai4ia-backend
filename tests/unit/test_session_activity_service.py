"""Unit tests for app.services.session_activity_service — the read-path
activity feed. Unlike session_service.py's best-effort write-path
functions, these must let a repository failure propagate (they're plain
GET endpoints, not login-critical bookkeeping), so no exception-swallowing
is expected or tested here.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services import session_activity_service


def _session(session_id, user_id, **extra):
    base = {
        "id": session_id,
        "user_id": user_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_seen_at": "2026-01-01T00:00:00+00:00",
        "ended_at": None,
        "end_reason": None,
        "user_agent": "curl/8",
        "ip": "1.2.3.4",
        "warmup_state": "ready",
        "warmup_started_at": None,
        "warmup_ready_at": None,
        "warmup_error": None,
    }
    base.update(extra)
    return base


class TestListAllSessions:
    def test_enriches_with_usernames_via_batch_lookup(self):
        session_repo = MagicMock()
        session_repo.list_all.return_value = (
            [_session("s1", "u1"), _session("s2", "u2"), _session("s3", "u1")],
            3,
        )
        user_repo = MagicMock()
        user_repo.get_by_ids.return_value = [
            {"id": "u1", "username": "alice"},
            {"id": "u2", "username": "bob"},
        ]

        with patch("app.services.session_activity_service.get_session_repository", return_value=session_repo), \
             patch("app.services.session_activity_service.get_user_repository", return_value=user_repo):
            result = session_activity_service.list_all_sessions(limit=50, offset=0)

        # One batch call, not one per session.
        user_repo.get_by_ids.assert_called_once_with(["u1", "u2"])
        usernames = {s["id"]: s["username"] for s in result["sessions"]}
        assert usernames == {"s1": "alice", "s2": "bob", "s3": "alice"}
        assert result["total"] == 3

    def test_username_is_none_when_user_not_found(self):
        session_repo = MagicMock()
        session_repo.list_all.return_value = ([_session("s1", "deleted-user")], 1)
        user_repo = MagicMock()
        user_repo.get_by_ids.return_value = []

        with patch("app.services.session_activity_service.get_session_repository", return_value=session_repo), \
             patch("app.services.session_activity_service.get_user_repository", return_value=user_repo):
            result = session_activity_service.list_all_sessions(limit=50, offset=0)

        assert result["sessions"][0]["username"] is None

    def test_no_lookup_when_no_sessions(self):
        session_repo = MagicMock()
        session_repo.list_all.return_value = ([], 0)
        user_repo = MagicMock()

        with patch("app.services.session_activity_service.get_session_repository", return_value=session_repo), \
             patch("app.services.session_activity_service.get_user_repository", return_value=user_repo):
            result = session_activity_service.list_all_sessions(limit=50, offset=0)

        user_repo.get_by_ids.assert_not_called()
        assert result == {"sessions": [], "total": 0, "limit": 50, "offset": 0}

    def test_passes_filters_through(self):
        session_repo = MagicMock()
        session_repo.list_all.return_value = ([], 0)
        with patch("app.services.session_activity_service.get_session_repository", return_value=session_repo), \
             patch("app.services.session_activity_service.get_user_repository", return_value=MagicMock()):
            session_activity_service.list_all_sessions(
                limit=10, offset=20, user_id="u1", active_only=True, since="2026-01-01"
            )
        session_repo.list_all.assert_called_once_with(
            limit=10, offset=20, user_id="u1", active_only=True, since="2026-01-01"
        )

    def test_repository_exception_propagates(self):
        session_repo = MagicMock()
        session_repo.list_all.side_effect = RuntimeError("db down")
        with patch("app.services.session_activity_service.get_session_repository", return_value=session_repo):
            with pytest.raises(RuntimeError):
                session_activity_service.list_all_sessions(limit=50, offset=0)


class TestListSessionsForUser:
    def test_404_when_user_does_not_exist(self):
        user_repo = MagicMock()
        user_repo.get_by_id.return_value = None
        with patch("app.services.session_activity_service.get_user_repository", return_value=user_repo):
            with pytest.raises(HTTPException) as exc_info:
                session_activity_service.list_sessions_for_user("ghost", limit=50, offset=0)
        assert exc_info.value.status_code == 404

    def test_returns_full_history_without_username_enrichment(self):
        user_repo = MagicMock()
        user_repo.get_by_id.return_value = {"id": "u1", "username": "alice"}
        session_repo = MagicMock()
        session_repo.list_for_user.return_value = ([_session("s1", "u1"), _session("s2", "u1")], 2)

        with patch("app.services.session_activity_service.get_user_repository", return_value=user_repo), \
             patch("app.services.session_activity_service.get_session_repository", return_value=session_repo):
            result = session_activity_service.list_sessions_for_user("u1", limit=50, offset=0)

        session_repo.list_for_user.assert_called_once_with("u1", limit=50, offset=0)
        assert result["total"] == 2
        assert "username" not in result["sessions"][0]
