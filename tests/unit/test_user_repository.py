"""Unit tests for UserRepository — specifically the malformed-id handling
found while testing GET /api/admin/users/{user_id}/sessions: a non-UUID
path param must 404, not 500. app_users.id is a uuid column, so Postgres
rejects a non-UUID string before the query runs at all; get_by_id/get_by_ids
must treat that as "not found" rather than let the raw error surface.

These were written against the Supabase/PostgREST client and now run against
psycopg2 — the behaviour under test is unchanged, but the error type is a
psycopg2 error carrying a pgcode instead of a PostgREST APIError.
"""
from __future__ import annotations

from unittest.mock import patch

import psycopg2
import pytest
from psycopg2 import errorcodes

from app.db.repositories.user_repository import UserRepository


class _FakePgError(psycopg2.Error):
    """psycopg2.Error.pgcode is a read-only descriptor set by the C driver, so
    a hand-built error needs a subclass that shadows it with a property."""

    def __init__(self, pgcode: str, message: str) -> None:
        super().__init__(message)
        self._pgcode = pgcode

    @property
    def pgcode(self) -> str:  # type: ignore[override]
        return self._pgcode


def _pg_error(pgcode: str, message: str) -> psycopg2.Error:
    return _FakePgError(pgcode, message)


_INVALID_UUID_ERROR = _pg_error(
    errorcodes.INVALID_TEXT_REPRESENTATION,
    'invalid input syntax for type uuid: "does-not-exist"',
)


class TestGetById:
    def test_malformed_uuid_returns_none_not_raise(self):
        with patch(
            "app.db.repositories.user_repository.pg_fetch_one",
            side_effect=_INVALID_UUID_ERROR,
        ):
            assert UserRepository().get_by_id("does-not-exist") is None

    def test_other_db_errors_still_raise(self):
        other = _pg_error(errorcodes.CONNECTION_FAILURE, "connection reset")
        with patch(
            "app.db.repositories.user_repository.pg_fetch_one",
            side_effect=other,
        ):
            with pytest.raises(psycopg2.Error):
                UserRepository().get_by_id("11111111-1111-1111-1111-111111111111")


class TestGetByIds:
    def test_empty_list_short_circuits_without_a_query(self):
        with patch("app.db.repositories.user_repository.pg_fetch_all") as fetch_all:
            assert UserRepository().get_by_ids([]) == []
            fetch_all.assert_not_called()

    def test_malformed_uuid_in_batch_returns_empty_not_raise(self):
        with patch(
            "app.db.repositories.user_repository.pg_fetch_all",
            side_effect=_INVALID_UUID_ERROR,
        ):
            assert UserRepository().get_by_ids(["does-not-exist"]) == []


class TestNormalization:
    """psycopg2 hands back UUID and datetime objects where PostgREST returned
    strings. Callers — the JWT `sub` claim above all — expect strings."""

    def test_uuid_and_datetime_are_stringified(self):
        import uuid
        from datetime import datetime, timezone

        row = {
            "id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
            "username": "cordel",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
        with patch("app.db.repositories.user_repository.pg_fetch_one", return_value=row):
            rec = UserRepository().get_by_id("11111111-1111-1111-1111-111111111111")

        assert rec is not None
        assert rec["id"] == "11111111-1111-1111-1111-111111111111"
        assert isinstance(rec["id"], str)
        assert isinstance(rec["created_at"], str)


class TestWritableColumnFiltering:
    """Repository SQL is built from dict keys, so unknown keys must be dropped
    rather than interpolated into the statement."""

    def test_unknown_columns_are_not_written(self):
        captured = {}

        def fake_fetch_one(query, params):
            captured["query"] = query
            captured["params"] = params
            return {"id": "x", "username": "u"}

        with patch(
            "app.db.repositories.user_repository.pg_fetch_one",
            side_effect=fake_fetch_one,
        ):
            UserRepository().create_user(
                {"username": "u", "email_id": "e@x.com", "password": "",
                 "evil); drop table app_users; --": "boom"}
            )

        assert "drop table" not in captured["query"].lower()
        assert "evil" not in captured["query"]
        assert set(captured["params"]) == {"username", "email_id", "password"}
