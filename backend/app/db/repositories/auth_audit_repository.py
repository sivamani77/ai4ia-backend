"""auth_audit — lives in the pipeline Postgres database (DATABASE_URL),
alongside app_users. Migrated off the auth Supabase project.
"""
from __future__ import annotations

import logging

from app.db.postgres import pg_execute, pg_fetch_all

logger = logging.getLogger("ai4ia.db.repositories.auth_audit")


class AuthAuditRepository:
    """auth_audit lives in Postgres alongside app_users."""

    table_name = "auth_audit"

    def log(
        self,
        *,
        event: str,
        success: bool,
        user_id: str | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        try:
            pg_execute(
                """
                insert into auth_audit (event, success, user_id, ip, user_agent)
                values (%(event)s, %(success)s, %(user_id)s, %(ip)s, %(user_agent)s)
                """,
                {
                    "event": event,
                    "success": success,
                    # Cast happens server-side against the uuid column; a
                    # malformed id would raise 22P02, which the except below
                    # swallows exactly like every other write failure here.
                    "user_id": user_id,
                    "ip": ip,
                    "user_agent": user_agent,
                },
            )
        except Exception:
            # Audit logging must never break the auth flow it's observing.
            logger.exception("Failed to write auth_audit row: event=%s user_id=%s", event, user_id)

    def purge_before(self, *, retention_days: int) -> int:
        """Delete audit rows older than `retention_days`, counted from
        created_at — unlike sessions and refresh tokens these rows have no
        end-of-life state, the write itself is the event. Returns the number
        of rows removed."""
        rows = pg_fetch_all(
            """
            delete from auth_audit
            where created_at < now() - make_interval(days => %(days)s)
            returning id
            """,
            {"days": retention_days},
        )
        return len(rows)


__all__ = ["AuthAuditRepository"]
