"""app_users — lives in the pipeline Postgres database (DATABASE_URL).

Migrated off the separate auth Supabase project; see
db/migrations/0006_auth_tables_to_postgres.sql for the schema and
scripts/migrate_auth_data_from_supabase.py for the one-time data copy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, TypedDict

import psycopg2
from psycopg2 import errorcodes

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

# Postgres error code for a value that doesn't parse as the column's type
# (app_users.id is uuid) — raised before the query even runs, so an
# arbitrary caller-supplied id string (e.g. a path param that isn't a UUID)
# must be treated as "not found", not surfaced as a 500.
_INVALID_TEXT_REPRESENTATION = errorcodes.INVALID_TEXT_REPRESENTATION  # 22P02

# Postgres error code for a unique-index violation — raised by the
# case-insensitive unique indexes on lower(username)/lower(email_id). The
# username_exists()/email_exists() pre-checks are a TOCTOU race (two signups
# for "Cordel"/"cordel" within milliseconds could both pass), so this is the
# actual backstop: the DB serializes the insert, one of the two hits this.
_UNIQUE_VIOLATION = errorcodes.UNIQUE_VIOLATION  # 23505

logger = logging.getLogger("ai4ia.db.repositories.users")

# Columns a caller is allowed to write. Anything else in an incoming mapping
# is ignored rather than interpolated — the previous PostgREST client would
# have rejected an unknown column server-side, and building SQL from unchecked
# dict keys is exactly the kind of thing that must not become possible here.
_WRITABLE_COLUMNS = frozenset({
    "id", "username", "password_hash", "password", "first_name", "last_name",
    "email_id", "persona", "status", "email_verified", "failed_login_count",
    "locked_until", "created_at", "updated_at",
})


class DuplicateUserError(Exception):
    """Raised when create_user loses a uniqueness race at the DB level —
    field is 'username' or 'email', inferred from the violated index name."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"Duplicate {field}")


def _duplicate_field(exc: psycopg2.Error) -> str:
    """Infer which unique index was violated. psycopg2 exposes the constraint
    name directly on diag, which is more reliable than the old string-matching
    on the PostgREST error message."""
    name = (getattr(exc.diag, "constraint_name", None) or "").lower()
    detail = (getattr(exc.diag, "message_detail", None) or "").lower()
    haystack = f"{name} {detail}"
    return "username" if "username" in haystack else "email"


def _escape_for_ilike(value: str) -> str:
    """Escape Postgres LIKE/ILIKE wildcards (% and _) so an exact
    case-insensitive match doesn't behave as a pattern — usernames are
    allowed to contain literal underscores."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class UserRecord(TypedDict, total=False):
    id: str
    username: str
    password_hash: str | None
    password: str | None  # legacy plaintext column, pre-migration rows only
    first_name: str
    last_name: str
    email_id: str
    persona: str
    status: str
    email_verified: bool
    failed_login_count: int
    locked_until: str | None
    created_at: str | None
    updated_at: str | None


class UserRepository:
    """Reads/writes app_users in the pipeline Postgres database."""

    table_name = "app_users"

    # -- lookups ------------------------------------------------------------

    def get_by_username(self, username: str) -> UserRecord | None:
        """Case-insensitive: matches regardless of how the account's
        username was originally cased, without requiring stored data to be
        rewritten to a canonical case. Uses lower(username) so the
        case-insensitive unique index is what answers the query."""
        row = pg_fetch_one(
            "select * from app_users where lower(username) = lower(%s) limit 1",
            (username,),
        )
        return self._record(row)

    def get_by_email(self, email: str) -> UserRecord | None:
        row = pg_fetch_one(
            "select * from app_users where lower(email_id) = lower(%s) limit 1",
            (email,),
        )
        return self._record(row)

    def get_by_id(self, user_id: str) -> UserRecord | None:
        try:
            row = pg_fetch_one("select * from app_users where id = %s limit 1", (user_id,))
        except psycopg2.Error as exc:
            if exc.pgcode == _INVALID_TEXT_REPRESENTATION:
                return None
            raise
        return self._record(row)

    def get_by_ids(self, ids: list[str]) -> list[UserRecord]:
        """Batch fetch for enrichment — e.g. resolving usernames onto a list
        of sessions. One query instead of N get_by_id calls.

        Now that app_users and user_sessions share a database a SQL JOIN is
        possible, but this stays a separate query so the calling code is
        unchanged by the migration.
        """
        if not ids:
            return []
        try:
            rows = pg_fetch_all("select * from app_users where id = any(%s::uuid[])", (list(ids),))
        except psycopg2.Error as exc:
            if exc.pgcode == _INVALID_TEXT_REPRESENTATION:
                return []
            raise
        return [self._record(row) for row in rows]  # type: ignore[misc]

    def username_exists(self, username: str) -> bool:
        row = pg_fetch_one(
            "select 1 as hit from app_users where lower(username) = lower(%s) limit 1",
            (username,),
        )
        return row is not None

    def email_exists(self, email: str) -> bool:
        row = pg_fetch_one(
            "select 1 as hit from app_users where lower(email_id) = lower(%s) limit 1",
            (email,),
        )
        return row is not None

    # -- writes -------------------------------------------------------------

    def create_user(self, data: Mapping[str, Any]) -> UserRecord:
        payload = {k: v for k, v in data.items() if k in _WRITABLE_COLUMNS}
        if not payload:
            raise ValueError("create_user called with no writable columns")

        columns = list(payload)
        placeholders = ", ".join(f"%({c})s" for c in columns)
        column_sql = ", ".join(f'"{c}"' for c in columns)

        try:
            row = pg_fetch_one(
                f"insert into app_users ({column_sql}) values ({placeholders}) returning *",
                payload,
            )
        except psycopg2.Error as exc:
            if exc.pgcode == _UNIQUE_VIOLATION:
                raise DuplicateUserError(_duplicate_field(exc)) from exc
            raise

        created = self._record(row)
        if created is None:
            raise RuntimeError("Insert into app_users returned no row")
        return created

    def update_user(self, user_id: str, updates: Mapping[str, Any]) -> None:
        payload = {k: v for k, v in updates.items() if k in _WRITABLE_COLUMNS}
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        assignments = ", ".join(f'"{c}" = %({c})s' for c in payload)
        try:
            pg_execute(
                f"update app_users set {assignments} where id = %(_user_id)s",
                {**payload, "_user_id": user_id},
            )
        except psycopg2.Error as exc:
            if exc.pgcode == _UNIQUE_VIOLATION:
                raise DuplicateUserError(_duplicate_field(exc)) from exc
            if exc.pgcode == _INVALID_TEXT_REPRESENTATION:
                return  # non-UUID id matches nothing, same as before
            raise

    def upgrade_legacy_password(self, user_id: str, password_hash: str) -> None:
        """Called once a legacy plaintext row has been verified — moves the
        account onto a real Argon2id hash. The legacy `password` column is
        NOT NULL in the existing schema, so it's blanked rather than nulled."""
        self.update_user(user_id, {"password_hash": password_hash, "password": ""})

    def record_failed_login(self, user_id: str, *, failed_count: int, locked_until: str | None) -> None:
        self.update_user(
            user_id,
            {"failed_login_count": failed_count, "locked_until": locked_until},
        )

    def reset_login_attempts(self, user_id: str) -> None:
        self.update_user(user_id, {"failed_login_count": 0, "locked_until": None})

    def mark_verified(self, user_id: str) -> None:
        self.update_user(user_id, {"status": "active", "email_verified": True})

    def list_all(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        persona: str | None = None,
        search: str | None = None,
    ) -> tuple[list[UserRecord], int]:
        """Returns (page, total_count). `search` matches username, email, or
        first/last name (case-insensitive substring)."""
        where: list[str] = []
        params: dict[str, Any] = {}

        if status:
            where.append("status = %(status)s")
            params["status"] = status
        if persona:
            where.append("persona = %(persona)s")
            params["persona"] = persona
        if search:
            # ESCAPE '\' pairs with _escape_for_ilike so a user searching for
            # "a_b" gets a literal underscore, not a single-character wildcard.
            where.append(
                "(username ilike %(search)s escape '\\'"
                " or email_id ilike %(search)s escape '\\'"
                " or first_name ilike %(search)s escape '\\'"
                " or last_name ilike %(search)s escape '\\')"
            )
            params["search"] = f"%{_escape_for_ilike(search)}%"

        where_sql = f"where {' and '.join(where)}" if where else ""

        total_row = pg_fetch_one(f"select count(*) as n from app_users {where_sql}", params)
        total = int(total_row["n"]) if total_row else 0

        rows = pg_fetch_all(
            f"""
            select * from app_users
            {where_sql}
            order by created_at desc
            limit %(_limit)s offset %(_offset)s
            """,
            {**params, "_limit": limit, "_offset": offset},
        )
        return [self._record(row) for row in rows], total  # type: ignore[misc]

    def deactivate(self, user_id: str) -> None:
        self.update_user(user_id, {"status": "deactivated"})

    def reactivate(self, user_id: str) -> None:
        self.update_user(user_id, {"status": "active", "failed_login_count": 0, "locked_until": None})

    def unlock(self, user_id: str) -> None:
        self.update_user(user_id, {"failed_login_count": 0, "locked_until": None})

    def set_persona(self, user_id: str, persona: str) -> None:
        self.update_user(user_id, {"persona": persona})

    def delete_user(self, user_id: str) -> None:
        """Hard delete — cascades to refresh_tokens/otp_codes (ON DELETE
        CASCADE) and nulls auth_audit.user_id (ON DELETE SET NULL). Does not
        touch the pipeline job tables (jobs/insights/traces) — callers must
        clean those up separately before calling this."""
        try:
            pg_execute("delete from app_users where id = %s", (user_id,))
        except psycopg2.Error as exc:
            if exc.pgcode == _INVALID_TEXT_REPRESENTATION:
                return
            raise

    # -- helpers ------------------------------------------------------------

    def _record(self, row: dict[str, Any] | None) -> UserRecord | None:
        """psycopg2 returns live UUID/datetime objects where PostgREST handed
        back strings; normalize_value restores the shape callers expect."""
        if row is None:
            return None
        return UserRecord(**normalize_value(dict(row)))


__all__ = ["UserRepository", "UserRecord", "DuplicateUserError"]
