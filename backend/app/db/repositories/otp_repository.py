from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from app.db.supabase import get_auth_supabase_client


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
    """otp_codes lives in the auth Supabase project alongside app_users."""

    table_name = "otp_codes"

    def __init__(self) -> None:
        self._client = get_auth_supabase_client()
        self._table = self._client.table(self.table_name)

    def invalidate_active(self, email: str, purpose: str) -> None:
        """Mark any not-yet-used codes for this (email, purpose) as used, so
        only the most recently issued code is ever valid."""
        now = datetime.now(timezone.utc).isoformat()
        (
            self._table.update({"used_at": now})
            .eq("email", email)
            .eq("purpose", purpose)
            .is_("used_at", "null")
            .execute()
        )

    def create(self, data: dict[str, Any]) -> OtpRecord:
        response = self._table.insert(data).execute()
        return OtpRecord(**response.data[0])

    def get_latest_active(self, email: str, purpose: str) -> OtpRecord | None:
        response = (
            self._table.select("*")
            .eq("email", email)
            .eq("purpose", purpose)
            .is_("used_at", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return OtpRecord(**response.data[0])

    def increment_attempts(self, otp_id: str, attempts: int) -> None:
        self._table.update({"attempts": attempts}).eq("id", otp_id).execute()

    def mark_used(self, otp_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._table.update({"used_at": now}).eq("id", otp_id).execute()
