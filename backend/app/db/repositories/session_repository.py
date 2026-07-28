"""user_sessions — lives in the pipeline Postgres database (DATABASE_URL),
alongside jobs/job_events/llm_call_traces — NOT the auth Supabase project.

See db/migrations/0005_user_sessions.sql for the schema and why `id` is the
refresh-token family_id rather than a newly generated identifier.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one

logger = logging.getLogger("ai4ia.db.repositories.sessions")


class SessionRecord(TypedDict, total=False):
    id: str
    user_id: str
    created_at: str | None
    last_seen_at: str | None
    ended_at: str | None
    end_reason: str | None
    user_agent: str | None
    ip: str | None
    warmup_state: str
    warmup_started_at: str | None
    warmup_ready_at: str | None
    warmup_error: str | None
    metadata: dict[str, Any]


class SessionRepository:
    table_name = "user_sessions"

    def create(
        self,
        *,
        session_id: str,
        user_id: str,
        user_agent: str | None = None,
        ip: str | None = None,
    ) -> SessionRecord | None:
        row = pg_fetch_one(
            """
            insert into user_sessions (id, user_id, user_agent, ip)
            values (%(id)s, %(user_id)s, %(user_agent)s, %(ip)s)
            on conflict (id) do update set
                last_seen_at = now(),
                ended_at = null,
                end_reason = null
            returning *
            """,
            {"id": session_id, "user_id": user_id, "user_agent": user_agent, "ip": ip},
        )
        return self._normalize(row) if row is not None else None

    def touch(self, session_id: str) -> None:
        pg_execute(
            "update user_sessions set last_seen_at = now() where id = %s",
            (session_id,),
        )

    def end(self, session_id: str, *, reason: str) -> None:
        pg_execute(
            """
            update user_sessions
            set ended_at = now(), end_reason = %(reason)s
            where id = %(id)s and ended_at is null
            """,
            {"id": session_id, "reason": reason},
        )

    def set_warmup_state(
        self,
        session_id: str,
        *,
        state: str,
        started_at: bool = False,
        ready_at: bool = False,
        error: str | None = None,
    ) -> None:
        """Status only — never gates a request. See app/services/model_gate.py."""
        pg_execute(
            """
            update user_sessions
            set warmup_state = %(state)s,
                warmup_started_at = case when %(started)s then now() else warmup_started_at end,
                warmup_ready_at = case when %(ready)s then now() else warmup_ready_at end,
                warmup_error = %(error)s
            where id = %(id)s
            """,
            {
                "id": session_id,
                "state": state,
                "started": started_at,
                "ready": ready_at,
                "error": error,
            },
        )

    def end_all_for_user(self, user_id: str, *, reason: str) -> None:
        pg_execute(
            """
            update user_sessions
            set ended_at = now(), end_reason = %(reason)s
            where user_id = %(user_id)s and ended_at is null
            """,
            {"user_id": user_id, "reason": reason},
        )

    def get(self, session_id: str) -> SessionRecord | None:
        row = pg_fetch_one("select * from user_sessions where id = %s limit 1", (session_id,))
        return self._normalize(row) if row is not None else None

    def list_active_for_user(self, user_id: str) -> list[SessionRecord]:
        rows = pg_fetch_all(
            """
            select * from user_sessions
            where user_id = %s and ended_at is null
            order by last_seen_at desc
            """,
            (user_id,),
        )
        return [self._normalize(row) for row in rows]

    def _normalize(self, row: dict[str, Any]) -> SessionRecord:
        data = dict(row)
        if not isinstance(data.get("metadata"), dict):
            data["metadata"] = {}
        return SessionRecord(**data)


__all__ = ["SessionRepository", "SessionRecord"]
