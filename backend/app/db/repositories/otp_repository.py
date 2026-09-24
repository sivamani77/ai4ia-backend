"""otp_codes — lives in the pipeline Postgres database (DATABASE_URL),
alongside app_users. Migrated off the auth Supabase project.
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.db.postgres import pg_execute, pg_fetch_one
from app.db.utils import normalize_value

_WRITABLE_COLUMNS = frozenset({
    "id", "email", "user_id", "purpose", "code_hash",
    "expires_at", "attempts", "used_at", "created_at",
})


class OtpRecord(TypedDict, total=False):
    id: str
    email: str
    user_id: str | None
    purpose: str
    code_hash: str
    expires_at: str
    attempts: int
    used_at: str | None
    created_at: str | None


class OtpRepository:
    """otp_codes lives in Postgres alongside app_users."""

    table_name = "otp_codes"

    def invalidate_active(self, email: str, purpose: str) -> None:
        """Mark any not-yet-used codes for this (email, purpose) as used, so
        only the most recently issued code is ever valid.

        now() is the database clock rather than the application's, which also
        removes the client/server clock-skew the previous implementation had
        when it sent an application-generated timestamp.
        """
        pg_execute(
            """
            update otp_codes set used_at = now()
            where email = %(email)s and purpose = %(purpose)s and used_at is null
            """,
            {"email": email, "purpose": purpose},
        )

    def create(self, data: dict[str, Any]) -> OtpRecord:
        payload = {k: v for k, v in data.items() if k in _WRITABLE_COLUMNS}
        if not payload:
            raise ValueError("OtpRepository.create called with no writable columns")

        columns = list(payload)
        column_sql = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(f"%({c})s" for c in columns)

        row = pg_fetch_one(
            f"insert into otp_codes ({column_sql}) values ({placeholders}) returning *",
            payload,
        )
        if row is None:
            raise RuntimeError("Insert into otp_codes returned no row")
        return OtpRecord(**normalize_value(dict(row)))

    def get_latest_active(self, email: str, purpose: str) -> OtpRecord | None:
        row = pg_fetch_one(
            """
            select * from otp_codes
            where email = %(email)s and purpose = %(purpose)s and used_at is null
            order by created_at desc
            limit 1
            """,
            {"email": email, "purpose": purpose},
        )
        if row is None:
            return None
        return OtpRecord(**normalize_value(dict(row)))

    def claim_attempt(self, otp_id: str, max_attempts: int) -> int | None:
        """Count one guess against an unused code, in a single statement so
        parallel guesses cannot exceed the cap. Returns the new attempt count,
        or None when the code is used up or already at the cap."""
        row = pg_fetch_one(
            """
            update otp_codes set attempts = attempts + 1
            where id = %(id)s and used_at is null and attempts < %(max)s
            returning attempts
            """,
            {"id": otp_id, "max": max_attempts},
        )
        return int(row["attempts"]) if row is not None else None

    def mark_used(self, otp_id: str) -> bool:
        """Consume a code. False if it was already used, so only one request
        can ever succeed with it."""
        row = pg_fetch_one(
            "update otp_codes set used_at = now() where id = %s and used_at is null returning id",
            (otp_id,),
        )
        return row is not None


__all__ = ["OtpRepository", "OtpRecord"]
