"""Unit tests for SessionRepository.list_all's filter/pagination SQL
building — the piece session_activity_service and the three activity-feed
endpoints depend on. Postgres itself is mocked out via pg_fetch_one/
pg_fetch_all so these only assert the query shape, not real DB behavior.
"""
from __future__ import annotations

from unittest.mock import patch

from app.db.repositories.session_repository import SessionRepository


def _row(session_id="s1", user_id="u1"):
    return {
        "id": session_id,
        "user_id": user_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_seen_at": "2026-01-01T00:00:00+00:00",
        "ended_at": None,
        "end_reason": None,
        "user_agent": None,
        "ip": None,
        "warmup_state": "ready",
        "warmup_started_at": None,
        "warmup_ready_at": None,
        "warmup_error": None,
        "metadata": {},
    }


class TestListAll:
    def test_no_filters_produces_unconditioned_query(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 0}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]) as mock_all:
            repo.list_all(limit=50, offset=0)

        count_sql = mock_one.call_args[0][0]
        page_sql = mock_all.call_args[0][0]
        assert "where" not in count_sql
        assert "where" not in page_sql

    def test_user_id_filter_is_parameterized(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 1}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[_row()]) as mock_all:
            repo.list_all(limit=50, offset=0, user_id="u1")

        assert "user_id = %(user_id)s" in mock_one.call_args[0][0]
        assert mock_one.call_args[0][1]["user_id"] == "u1"
        assert mock_all.call_args[0][1]["user_id"] == "u1"

    def test_active_only_true_filters_unended(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 0}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]):
            repo.list_all(limit=50, offset=0, active_only=True)
        assert "ended_at is null" in mock_one.call_args[0][0]

    def test_active_only_false_filters_ended(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 0}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]):
            repo.list_all(limit=50, offset=0, active_only=False)
        assert "ended_at is not null" in mock_one.call_args[0][0]

    def test_since_filter_is_parameterized(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 0}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]):
            repo.list_all(limit=50, offset=0, since="2026-01-01")
        assert "created_at >= %(since)s" in mock_one.call_args[0][0]
        assert mock_one.call_args[0][1]["since"] == "2026-01-01"

    def test_combined_filters_are_anded(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 0}) as mock_one, \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]):
            repo.list_all(limit=50, offset=0, user_id="u1", active_only=True, since="2026-01-01")
        sql = mock_one.call_args[0][0]
        assert "user_id = %(user_id)s and ended_at is null and created_at >= %(since)s" in sql

    def test_returns_page_and_total(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value={"n": 5}), \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[_row(), _row("s2")]):
            rows, total = repo.list_all(limit=2, offset=0)
        assert total == 5
        assert [r["id"] for r in rows] == ["s1", "s2"]

    def test_zero_count_row_handled(self):
        repo = SessionRepository()
        with patch("app.db.repositories.session_repository.pg_fetch_one", return_value=None), \
             patch("app.db.repositories.session_repository.pg_fetch_all", return_value=[]):
            rows, total = repo.list_all(limit=50, offset=0)
        assert rows == []
        assert total == 0


class TestListForUser:
    def test_delegates_to_list_all_with_user_id(self):
        repo = SessionRepository()
        with patch.object(repo, "list_all", return_value=([], 0)) as mock_list_all:
            repo.list_for_user("u1", limit=10, offset=5)
        mock_list_all.assert_called_once_with(user_id="u1", limit=10, offset=5)
