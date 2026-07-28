"""Unit tests for UserRepository — specifically the malformed-id handling
found while testing GET /api/admin/users/{user_id}/sessions: a non-UUID
path param must 404, not 500. app_users.id is a uuid column, so Postgres
rejects a non-UUID string before the query runs at all; get_by_id/get_by_ids
must treat that as "not found" rather than let the raw error surface.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from postgrest.exceptions import APIError

from app.db.repositories.user_repository import UserRepository

_INVALID_UUID_ERROR = APIError(
    {"message": 'invalid input syntax for type uuid: "does-not-exist"', "code": "22P02"}
)


def _repo_with_table(table: MagicMock) -> UserRepository:
    client = MagicMock()
    client.table.return_value = table
    with patch("app.db.repositories.user_repository.get_auth_supabase_client", return_value=client):
        return UserRepository()


class TestGetById:
    def test_malformed_uuid_returns_none_not_raise(self):
        table = MagicMock()
        table.select.return_value.eq.return_value.limit.return_value.execute.side_effect = _INVALID_UUID_ERROR
        repo = _repo_with_table(table)

        assert repo.get_by_id("does-not-exist") is None

    def test_other_api_errors_still_raise(self):
        table = MagicMock()
        other_error = APIError({"message": "connection reset", "code": "08006"})
        table.select.return_value.eq.return_value.limit.return_value.execute.side_effect = other_error
        repo = _repo_with_table(table)

        with pytest.raises(APIError):
            repo.get_by_id("11111111-1111-1111-1111-111111111111")


class TestGetByIds:
    def test_empty_list_short_circuits_without_a_query(self):
        table = MagicMock()
        repo = _repo_with_table(table)
        assert repo.get_by_ids([]) == []
        table.select.assert_not_called()

    def test_malformed_uuid_in_batch_returns_empty_not_raise(self):
        table = MagicMock()
        table.select.return_value.in_.return_value.execute.side_effect = _INVALID_UUID_ERROR
        repo = _repo_with_table(table)

        assert repo.get_by_ids(["does-not-exist"]) == []
