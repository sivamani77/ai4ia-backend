"""Direct Postgres connection pool for the video-job pipeline tables
(jobs, job_events, agent_decision_records, llm_call_traces).

Replaces the Supabase REST client for these four tables: the REST/
PostgREST layer was the actual source of the "RemoteProtocolError: Server
disconnected" errors seen during long-running jobs (an idle HTTP/2
keepalive getting dropped), not the database itself. A pooled, persistent
direct connection doesn't have that failure mode — a dropped TCP connection
here is just a normal "reconnect and retry", handled below.

Connection address comes from settings.database_url (DATABASE_URL env var).
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from app.core.settings import settings

logger = logging.getLogger("ai4ia.db.postgres")

_T = TypeVar("_T")

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _is_transient_connection_error(exc: Exception) -> bool:
    """A dropped connection a fresh one would recover from — distinct from a
    real query error (bad SQL, constraint violation), which should not be
    retried."""
    return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))


def get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:  # re-check: another thread may have won the race
                logger.info("Initializing Postgres connection pool")
                _pool = pg_pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=settings.database_url,
                )
    return _pool


def reset_pool() -> None:
    """Drop the pool so the next get_pool() builds a fresh one. Reserved for
    when the pool itself is unusable (e.g. DSN/credentials changed) — normal
    per-connection failures are handled by discarding just that connection
    in _connection(), not by tearing down the pool other threads are using."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.closeall()
            except Exception:  # noqa: BLE001
                pass
            _pool = None


@contextmanager
def _connection() -> Iterator[Any]:
    pool = get_pool()
    conn = pool.getconn()
    discard = False
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        discard = _is_transient_connection_error(exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            discard = True
        raise
    finally:
        try:
            pool.putconn(conn, close=discard)
        except pg_pool.PoolError:
            logger.warning("Pool already closed while returning connection")


def _with_retry(op_name: str, fn: Callable[[], _T], *, attempts: int = 3) -> _T:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_transient_connection_error(exc) or attempt == attempts:
                logger.exception("Postgres operation failed: op=%s", op_name)
                raise
            logger.warning(
                "Postgres transient error on %s (attempt %d/%d): %s — "
                "discarding connection and retrying",
                op_name, attempt, attempts, exc,
            )
            time.sleep(0.5 * attempt)
    raise last_exc  # unreachable, keeps type-checkers happy


def pg_execute(query: str, params: tuple | dict | None = None) -> None:
    """Run a statement with no result rows expected (INSERT/UPDATE/DELETE
    without RETURNING)."""
    def _run() -> None:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
    _with_retry("execute", _run)


def pg_fetch_one(query: str, params: tuple | dict | None = None) -> dict[str, Any] | None:
    def _run() -> dict[str, Any] | None:
        with _connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row is not None else None
    return _with_retry("fetch_one", _run)


@contextmanager
def pg_transaction() -> Iterator[Any]:
    """One connection and one transaction for several statements; commits on
    success, rolls back on error. Yields a dict-row cursor. Not retried: a
    transaction cannot be safely replayed halfway."""
    with _connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur


def pg_fetch_all(query: str, params: tuple | dict | None = None) -> list[dict[str, Any]]:
    def _run() -> list[dict[str, Any]]:
        with _connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                return [dict(row) for row in rows]
    return _with_retry("fetch_all", _run)
