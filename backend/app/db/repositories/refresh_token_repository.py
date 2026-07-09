from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from app.db.supabase import get_auth_supabase_client


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
    """refresh_tokens lives in the auth Supabase project alongside app_users."""

    table_name = "refresh_tokens"

    def __init__(self) -> None:
        self._client = get_auth_supabase_client()
        self._table = self._client.table(self.table_name)

    def create(self, data: dict[str, Any]) -> RefreshTokenRecord:
        response = self._table.insert(data).execute()
        return RefreshTokenRecord(**response.data[0])

    def get_by_hash(self, token_hash: str) -> RefreshTokenRecord | None:
        response = self._table.select("*").eq("token_hash", token_hash).limit(1).execute()
        if not response.data:
            return None
        return RefreshTokenRecord(**response.data[0])

    def mark_used(self, token_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._table.update({"used_at": now}).eq("id", token_id).execute()

    def revoke(self, token_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._table.update({"revoked_at": now}).eq("id", token_id).execute()

    def revoke_family(self, family_id: str) -> None:
        """Revoke every token in a rotation lineage — used on reuse detection
        (a used-and-replayed token means the family may be compromised) and
        on password reset (force logout everywhere)."""
        now = datetime.now(timezone.utc).isoformat()
        (
            self._table.update({"revoked_at": now})
            .eq("family_id", family_id)
            .is_("revoked_at", "null")
            .execute()
        )

    def revoke_all_for_user(self, user_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        (
            self._table.update({"revoked_at": now})
            .eq("user_id", user_id)
            .is_("revoked_at", "null")
            .execute()
        )
