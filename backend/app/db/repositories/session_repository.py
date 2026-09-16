"""user_sessions — lives in the pipeline Postgres database (DATABASE_URL),
alongside jobs/job_events/llm_call_traces — NOT the auth Supabase project.

See db/migrations/0005_user_sessions.sql for the schema and why `id` is the
refresh-token family_id rather than a newly generated identifier.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

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

    def purge_ended_before(self, *, retention_days: int) -> int:
        """Delete sessions that ended more than `retention_days` ago.

        Only rows with a non-null ended_at are eligible, so an active session
        is never purged however old it is — the clock starts at logout, not
        at login. Returns the number of rows removed."""
        rows = pg_fetch_all(
            """
            delete from user_sessions
            where ended_at is not null
              and ended_at < now() - make_interval(days => %(days)s)
            returning id
            """,
            {"days": retention_days},
        )
        return len(rows)

    def delete_all_for_user(self, user_id: str) -> int:
        """Hard delete every session row for a user. user_sessions has no FK
        to app_users (see db/schema.py), so account deletion must call this
        explicitly or the row's ip/user_agent outlives the account."""
        rows = pg_fetch_all(
            "delete from user_sessions where user_id = %s returning id",
            (user_id,),
        )
        return len(rows)

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

    def list_all(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        user_id: str | None = None,
        active_only: bool | None = None,
        since: str | None = None,
    ) -> tuple[list[SessionRecord], int]:
        """Activity feed — every session (any user) unless `user_id` narrows
        it to one. Returns (page, total_count), same shape as
        UserRepository.list_all. `active_only=True` means "not yet ended";
        `False` means "already ended"; omitted means both.
        """
        where_clauses: list[str] = []
        params: dict[str, Any] = {}

        if user_id:
            where_clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id
        if active_only is True:
            where_clauses.append("ended_at is null")
        elif active_only is False:
            where_clauses.append("ended_at is not null")
        if since:
            where_clauses.append("created_at >= %(since)s")
            params["since"] = since

        where_sql = f"where {' and '.join(where_clauses)}" if where_clauses else ""

        total_row = pg_fetch_one(f"select count(*) as n from user_sessions {where_sql}", params)
        total = int(total_row["n"]) if total_row else 0

        rows = pg_fetch_all(
            f"""
            select * from user_sessions
            {where_sql}
            order by created_at desc
            limit %(limit)s offset %(offset)s
            """,
            {**params, "limit": limit, "offset": offset},
        )
        return [self._normalize(row) for row in rows], total

    def list_for_user(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> tuple[list[SessionRecord], int]:
        """Full history for one user (active + ended) — for the activity-log
        use case. list_active_for_user above stays for its original,
        narrower purpose (does this user have a live session right now)."""
        return self.list_all(user_id=user_id, limit=limit, offset=offset)

    def _normalize(self, row: dict[str, Any]) -> SessionRecord:
        # psycopg2's RealDictCursor returns live datetime objects for
        # timestamptz columns (unlike the Supabase/PostgREST client used
        # elsewhere, which already hands back ISO strings) — normalize_value
        # converts them so callers (and Pydantic response models) get a
        # consistent str shape regardless of which client produced the row.
        data = normalize_value(dict(row))
        if not isinstance(data.get("metadata"), dict):
            data["metadata"] = {}
        return SessionRecord(**data)


__all__ = ["SessionRepository", "SessionRecord"]
