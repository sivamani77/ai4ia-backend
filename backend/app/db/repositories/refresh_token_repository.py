"""refresh_tokens — lives in the pipeline Postgres database (DATABASE_URL),
alongside app_users. Migrated off the auth Supabase project.

family_id is the rotation lineage id and is also user_sessions.id. Both
tables now share a database, so that link could be a real foreign key — it is
deliberately left unenforced for now because a refresh token is issued before
its session row exists in some flows. See db/schema.py for the full note.
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

_WRITABLE_COLUMNS = frozenset({
    "id", "user_id", "token_hash", "family_id", "expires_at",
    "created_at", "used_at", "revoked_at", "user_agent", "ip",
})


class RefreshTokenRecord(TypedDict, total=False):
    id: str
    user_id: str
    token_hash: str
    family_id: str
    expires_at: str
    created_at: str | None
    used_at: str | None
    revoked_at: str | None
    user_agent: str | None
    ip: str | None


class RefreshTokenRepository:
    """refresh_tokens lives in Postgres alongside app_users."""

    table_name = "refresh_tokens"

    def create(self, data: dict[str, Any]) -> RefreshTokenRecord:
        payload = {k: v for k, v in data.items() if k in _WRITABLE_COLUMNS}
        if not payload:
            raise ValueError("RefreshTokenRepository.create called with no writable columns")

        columns = list(payload)
        column_sql = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f"%({c})s" for c in columns)

        row = pg_fetch_one(
            f"insert into refresh_tokens ({column_sql}) values ({placeholders}) returning *",
            payload,
        )
        if row is None:
            raise RuntimeError("Insert into refresh_tokens returned no row")
        return RefreshTokenRecord(**normalize_value(dict(row)))

    def get_by_hash(self, token_hash: str) -> RefreshTokenRecord | None:
        row = pg_fetch_one(
            "select * from refresh_tokens where token_hash = %s limit 1",
            (token_hash,),
        )
        if row is None:
            return None
        return RefreshTokenRecord(**normalize_value(dict(row)))

    def mark_used(self, token_id: str) -> bool:
        """Claim a token for rotation. False if it was already used or revoked,
        so two parallel refreshes cannot both rotate the same token."""
        row = pg_fetch_one(
            """
            update refresh_tokens set used_at = now()
            where id = %s and used_at is null and revoked_at is null
            returning id
            """,
            (token_id,),
        )
        return row is not None

    def revoke(self, token_id: str) -> None:
        pg_execute("update refresh_tokens set revoked_at = now() where id = %s", (token_id,))

    def revoke_family(self, family_id: str) -> None:
        """Revoke every token in a rotation lineage — used on reuse detection
        (a used-and-replayed token means the family may be compromised) and
        on password reset (force logout everywhere)."""
        pg_execute(
            """
            update refresh_tokens set revoked_at = now()
            where family_id = %s and revoked_at is null
            """,
            (family_id,),
        )

    def revoke_all_for_user(self, user_id: str) -> None:
        pg_execute(
            """
            update refresh_tokens set revoked_at = now()
            where user_id = %s and revoked_at is null
            """,
            (user_id,),
        )

    def purge_spent_before(self, *, retention_days: int) -> int:
        """Delete refresh tokens that can no longer be used and have been
        dead for longer than `retention_days`.

        A token is dead once it is revoked or past expires_at; nothing in the
        auth flow ever deleted those rows, so they accumulated forever along
        with their ip/user_agent. A live token is never touched, whatever its
        age. Returns the number of rows removed."""
        rows = pg_fetch_all(
            """
            delete from refresh_tokens
            where (
                    (revoked_at is not null
                     and revoked_at < now() - make_interval(days => %(days)s))
                 or (expires_at < now() - make_interval(days => %(days)s))
            )
            returning id
            """,
            {"days": retention_days},
        )
        return len(rows)


__all__ = ["RefreshTokenRepository", "RefreshTokenRecord"]
