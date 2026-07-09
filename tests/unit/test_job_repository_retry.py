"""Tests for JobRepository's transient-disconnect retry.

Regression: after a long (~17min) render, Supabase closed the idle HTTP/2
keepalive connection; the next load_job raised
'RemoteProtocolError: Server disconnected', which flipped an ALREADY-completed
job to 'failed' and skipped the S3 upload. Reads/writes must now ride through a
dropped connection by resetting the cached client and retrying.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.db.repositories.job_repository as jr  # noqa: E402


class RemoteProtocolError(Exception):
    """Mimics httpx/httpcore RemoteProtocolError by class name."""


class TestIsTransient:
    def test_server_disconnected_message_is_transient(self):
        assert jr._is_transient_connection_error(Exception("Server disconnected")) is True

    def test_remoteprotocol_by_class_name_is_transient(self):
        assert jr._is_transient_connection_error(RemoteProtocolError("boom")) is True

    def test_connection_reset_is_transient(self):
        assert jr._is_transient_connection_error(Exception("Connection reset by peer")) is True

    def test_plain_value_error_is_not_transient(self):
        assert jr._is_transient_connection_error(ValueError("bad column")) is False

    def test_http_4xx_message_is_not_transient(self):
        assert jr._is_transient_connection_error(Exception("400 Bad Request: invalid filter")) is False


@pytest.fixture
def repo():
    """A JobRepository with the Supabase client fully mocked out."""
    with patch.object(jr, "get_supabase_client", return_value=MagicMock()):
        r = jr.JobRepository()
    return r


class TestWithRetry:
    def test_recovers_from_one_transient_then_succeeds(self, repo):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RemoteProtocolError("Server disconnected")
            return "ok"

        with patch.object(jr, "reset_supabase_client") as mock_reset, \
             patch.object(jr, "get_supabase_client", return_value=MagicMock()), \
             patch.object(jr.time, "sleep"):
            result = repo._with_retry("select_one", "job1", flaky)

        assert result == "ok"
        assert calls["n"] == 2
        mock_reset.assert_called_once()

    def test_non_transient_error_raises_immediately_without_reset(self, repo):
        def bad():
            raise ValueError("bad column")

        with patch.object(jr, "reset_supabase_client") as mock_reset, \
             patch.object(jr.time, "sleep"):
            with pytest.raises(ValueError):
                repo._with_retry("select_one", "job1", bad)
        mock_reset.assert_not_called()

    def test_persistent_transient_raises_after_attempts(self, repo):
        def always_down():
            raise RemoteProtocolError("Server disconnected")

        with patch.object(jr, "reset_supabase_client") as mock_reset, \
             patch.object(jr, "get_supabase_client", return_value=MagicMock()), \
             patch.object(jr.time, "sleep"):
            with pytest.raises(RemoteProtocolError):
                repo._with_retry("select_one", "job1", always_down, attempts=3)
        # reset attempted on each non-final failure (attempts-1 times)
        assert mock_reset.call_count == 2

    def test_get_job_rides_through_disconnect(self, repo):
        """End-to-end through the real get_job path: first execute() drops, retry succeeds."""
        state = {"n": 0}

        class FakeQuery:
            def select(self, *a): return self
            def eq(self, *a): return self
            def limit(self, *a): return self
            def execute(self):
                state["n"] += 1
                if state["n"] == 1:
                    raise RemoteProtocolError("Server disconnected")
                return MagicMock(data=[{"id": "job1", "status": "completed"}])

        fake_query = FakeQuery()
        # A fresh client whose .table() always yields our fake query, so the
        # post-reset rebuild inside _with_retry still hits FakeQuery.
        fresh_client = MagicMock()
        fresh_client.table = MagicMock(return_value=fake_query)
        repo._table = fake_query

        with patch.object(jr, "reset_supabase_client"), \
             patch.object(jr, "get_supabase_client", return_value=fresh_client), \
             patch.object(jr.time, "sleep"):
            result = repo.get_job("job1")

        assert result is not None
        assert result["id"] == "job1"
        assert result["status"] == "completed"
        assert state["n"] == 2  # first drop, retry succeeded
