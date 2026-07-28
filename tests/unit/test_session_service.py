"""Unit tests for app.services.session_service.

Every method must be best-effort against Postgres: a repository failure is
logged, never raised, so a Postgres hiccup can't fail a login/refresh/logout.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services import session_service


class TestCreateSession:
    def test_calls_repository_with_expected_fields(self):
        repo = MagicMock()
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.create_session(
                session_id="sess-1", user_id="user-1", user_agent="curl/8", ip="1.2.3.4"
            )
        repo.create.assert_called_once_with(
            session_id="sess-1", user_id="user-1", user_agent="curl/8", ip="1.2.3.4"
        )

    def test_repository_exception_is_swallowed(self):
        repo = MagicMock()
        repo.create.side_effect = RuntimeError("db down")
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.create_session(session_id="sess-1", user_id="user-1")  # must not raise


class TestTouchAndEnd:
    def test_touch_session_swallows_errors(self):
        repo = MagicMock()
        repo.touch.side_effect = RuntimeError("db down")
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.touch_session("sess-1")  # must not raise

    def test_end_session_passes_reason(self):
        repo = MagicMock()
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.end_session("sess-1", reason="logout")
        repo.end.assert_called_once_with("sess-1", reason="logout")

    def test_end_all_sessions_for_user_swallows_errors(self):
        repo = MagicMock()
        repo.end_all_for_user.side_effect = RuntimeError("db down")
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.end_all_sessions_for_user("user-1", reason="password_reset")


class TestWarmupState:
    def test_mark_warmup_started(self):
        repo = MagicMock()
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.mark_warmup_started("sess-1")
        repo.set_warmup_state.assert_called_once_with("sess-1", state="warming", started_at=True)

    def test_mark_warmup_ready(self):
        repo = MagicMock()
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.mark_warmup_ready("sess-1")
        repo.set_warmup_state.assert_called_once_with("sess-1", state="ready", ready_at=True)

    def test_mark_warmup_error_truncates_long_message(self):
        repo = MagicMock()
        long_error = "x" * 1000
        with patch("app.services.session_service.get_session_repository", return_value=repo):
            session_service.mark_warmup_error("sess-1", long_error)
        _, kwargs = repo.set_warmup_state.call_args
        assert len(kwargs["error"]) == 500
