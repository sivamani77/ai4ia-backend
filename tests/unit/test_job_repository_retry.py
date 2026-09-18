"""Tests for the Postgres layer's transient-disconnect retry.

Regression: after a long (~17min) render, the DB connection's idle TCP
connection got dropped; the next query raised psycopg2.OperationalError,
which flipped an ALREADY-completed job to 'failed' and skipped the S3
upload. Reads/writes must now ride through a dropped connection by
discarding the bad connection and retrying on a fresh one.

The retry logic used to live on JobRepository (backed by a Supabase REST
client); it has since moved to app.db.postgres, which owns a pooled,
direct psycopg2 connection shared by all repositories (see
app/db/postgres.py's module docstring).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import psycopg2.errors
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.db.postgres as pg  # noqa: E402


class TestIsTransient:
    def test_operational_error_is_transient(self):
        assert pg._is_transient_connection_error(
            psycopg2.OperationalError("server closed the connection unexpectedly")
        ) is True

    def test_interface_error_is_transient(self):
        assert pg._is_transient_connection_error(
            psycopg2.InterfaceError("connection already closed")
        ) is True

    def test_operational_error_subclass_is_transient(self):
        # e.g. AdminShutdown is a psycopg2.OperationalError subclass; the
        # check is isinstance-based so subclasses ride through it too.
        assert pg._is_transient_connection_error(
            psycopg2.errors.AdminShutdown("terminating connection due to administrator command")
        ) is True

    def test_value_error_is_not_transient(self):
        assert pg._is_transient_connection_error(ValueError("bad column")) is False

    def test_plain_exception_is_not_transient(self):
        assert pg._is_transient_connection_error(Exception("400 Bad Request: invalid filter")) is False


class TestWithRetry:
    def test_recovers_from_one_transient_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise psycopg2.OperationalError("server closed the connection unexpectedly")
            return "ok"

        with patch.object(pg.time, "sleep"):
            result = pg._with_retry("select_one", flaky)

        assert result == "ok"
        assert calls["n"] == 2

    def test_non_transient_error_raises_immediately_without_further_attempts(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise ValueError("bad column")

        with patch.object(pg.time, "sleep") as mock_sleep:
            with pytest.raises(ValueError):
                pg._with_retry("select_one", bad)

        assert calls["n"] == 1
        mock_sleep.assert_not_called()

    def test_persistent_transient_raises_after_attempts(self):
        calls = {"n": 0}

        def always_down():
            calls["n"] += 1
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

        with patch.object(pg.time, "sleep"):
            with pytest.raises(psycopg2.OperationalError):
                pg._with_retry("select_one", always_down, attempts=3)

        assert calls["n"] == 3

    def test_sleeps_with_increasing_backoff_between_attempts(self):
        def always_down():
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

        with patch.object(pg.time, "sleep") as mock_sleep:
            with pytest.raises(psycopg2.OperationalError):
                pg._with_retry("select_one", always_down, attempts=3)

        # sleeps between attempts 1->2 and 2->3 (not after the final failure)
        assert mock_sleep.call_args_list == [((0.5,),), ((1.0,),)]


class TestConnectionDiscardsOnTransientError:
    def test_pg_fetch_one_rides_through_one_dropped_connection(self):
        """End-to-end through pg_fetch_one: first getconn's cursor drops
        with a transient error, the connection is discarded (closed) and a
        retry against a fresh connection succeeds."""
        state = {"n": 0}

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **kw):
                state["n"] += 1
                if state["n"] == 1:
                    raise psycopg2.OperationalError("server closed the connection unexpectedly")

            def fetchone(self):
                return {"id": "job1", "status": "completed"}

        fake_conn = MagicMock()
        fake_conn.cursor.return_value = FakeCursor()

        fake_pool = MagicMock()
        fake_pool.getconn.return_value = fake_conn

        with patch.object(pg, "get_pool", return_value=fake_pool), \
             patch.object(pg.time, "sleep"):
            result = pg.pg_fetch_one("select * from jobs where id = %s limit 1", ("job1",))

        assert result == {"id": "job1", "status": "completed"}
        assert state["n"] == 2

        # First attempt's connection was returned with close=True (discarded);
        # the retry's connection was returned healthy (close=False).
        close_flags = [call.kwargs.get("close", call.args[1] if len(call.args) > 1 else None)
                       for call in fake_pool.putconn.call_args_list]
        assert close_flags == [True, False]
