"""Unit tests for the 30-day auth retention sweep (privacy policy 2.4).

user_sessions / refresh_tokens / auth_audit all store the client IP and
user-agent, and every end-of-life path in the auth flow is an UPDATE rather
than a DELETE. These tests pin the two properties that make the sweep safe:
it deletes only rows that are already finished, and it never touches a live
session or a usable refresh token.

Postgres is mocked via pg_fetch_all, so these assert the query shape and the
task's orchestration, not real DB behavior.
"""
from __future__ import annotations

from unittest.mock import patch

from app.db.repositories.auth_audit_repository import AuthAuditRepository
from app.db.repositories.refresh_token_repository import RefreshTokenRepository
from app.db.repositories.session_repository import SessionRepository


def _normalize(sql: str) -> str:
    return " ".join(sql.split()).lower()


class TestSessionPurge:
    def test_only_ended_sessions_are_eligible(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]) as mock_all:
            repo.purge_ended_before(retention_days=30)

        sql = _normalize(mock_all.call_args[0][0])
        assert sql.startswith("delete from user_sessions")
        # An active session must survive regardless of age.
        assert "ended_at is not null" in sql
        # The window runs from logout, not from login.
        assert "ended_at <" in sql
        assert "created_at" not in sql

    def test_retention_days_is_parameterized(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]) as mock_all:
            repo.purge_ended_before(retention_days=30)

        assert mock_all.call_args[0][1] == {"days": 30}

    def test_returns_deleted_row_count(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_all",
                   return_value=[{"id": "s1"}, {"id": "s2"}]):
            assert repo.purge_ended_before(retention_days=30) == 2

    def test_delete_all_for_user_is_scoped_and_parameterized(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_all",
                   return_value=[{"id": "s1"}]) as mock_all:
            assert repo.delete_all_for_user("u1") == 1

        sql = _normalize(mock_all.call_args[0][0])
        assert sql.startswith("delete from user_sessions where user_id = %s")
        assert mock_all.call_args[0][1] == ("u1",)


class TestRefreshTokenPurge:
    def test_only_revoked_or_expired_tokens_are_eligible(self):
        repo = RefreshTokenRepository()
        with patch("app.db.repositories.refresh_token_repository.pg_fetch_all",
                   return_value=[]) as mock_all:
            repo.purge_spent_before(retention_days=30)

        sql = _normalize(mock_all.call_args[0][0])
        assert sql.startswith("delete from refresh_tokens")
        # A live token (not revoked, not expired) matches neither arm.
        assert "revoked_at is not null" in sql
        assert "expires_at <" in sql

    def test_retention_days_is_parameterized(self):
        repo = RefreshTokenRepository()
        with patch("app.db.repositories.refresh_token_repository.pg_fetch_all",
                   return_value=[]) as mock_all:
            repo.purge_spent_before(retention_days=30)

        assert mock_all.call_args[0][1] == {"days": 30}


class TestAuditPurge:
    def test_deletes_by_created_at(self):
        repo = AuthAuditRepository()
        with patch("app.db.repositories.auth_audit_repository.pg_fetch_all",
                   return_value=[{"id": "a1"}]) as mock_all:
            assert repo.purge_before(retention_days=30) == 1

        sql = _normalize(mock_all.call_args[0][0])
        assert sql.startswith("delete from auth_audit")
        assert "created_at <" in sql
        assert mock_all.call_args[0][1] == {"days": 30}


class TestPurgeTask:
    def _repos(self, sessions=0, tokens=0, audit=0):
        from unittest.mock import MagicMock

        s, t, a = MagicMock(), MagicMock(), MagicMock()
        s.purge_ended_before.return_value = sessions
        t.purge_spent_before.return_value = tokens
        a.purge_before.return_value = audit
        return s, t, a

    def test_sweeps_all_three_tables_with_configured_window(self):
        from app.worker.celery_app import purge_expired_auth_data_task

        s, t, a = self._repos(sessions=3, tokens=5, audit=7)
        with patch("app.db.repositories.get_session_repository", return_value=s), \
             patch("app.db.repositories.get_refresh_token_repository", return_value=t), \
             patch("app.db.repositories.get_auth_audit_repository", return_value=a), \
             patch("app.core.settings.settings.auth_data_retention_days", 30):
            result = purge_expired_auth_data_task()

        assert result == {
            "retention_days": 30,
            "sessions": 3,
            "refresh_tokens": 5,
            "auth_audit": 7,
        }
        s.purge_ended_before.assert_called_once_with(retention_days=30)
        t.purge_spent_before.assert_called_once_with(retention_days=30)
        a.purge_before.assert_called_once_with(retention_days=30)

    def test_one_failing_sweep_does_not_abandon_the_others(self):
        from app.worker.celery_app import purge_expired_auth_data_task

        s, t, a = self._repos(tokens=5, audit=7)
        s.purge_ended_before.side_effect = RuntimeError("connection lost")
        with patch("app.db.repositories.get_session_repository", return_value=s), \
             patch("app.db.repositories.get_refresh_token_repository", return_value=t), \
             patch("app.db.repositories.get_auth_audit_repository", return_value=a), \
             patch("app.core.settings.settings.auth_data_retention_days", 30):
            result = purge_expired_auth_data_task()

        assert result["sessions"] == "error"
        assert result["refresh_tokens"] == 5
        assert result["auth_audit"] == 7
