from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_auth_supabase_client

logger = logging.getLogger("ai4ia.db.repositories.users")


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
    """Reads/writes app_users in the SEPARATE auth Supabase project. Never use
    get_supabase_client() (pipeline project) here — always get_auth_supabase_client()."""

    table_name = "app_users"

    def __init__(self) -> None:
        self._client = get_auth_supabase_client()
        self._table = self._client.table(self.table_name)

    def get_by_username(self, username: str) -> UserRecord | None:
        response = self._table.select("*").eq("username", username).limit(1).execute()
        return self._first(response.data)

    def get_by_email(self, email: str) -> UserRecord | None:
        response = self._table.select("*").eq("email_id", email).limit(1).execute()
        return self._first(response.data)

    def get_by_id(self, user_id: str) -> UserRecord | None:
        response = self._table.select("*").eq("id", user_id).limit(1).execute()
        return self._first(response.data)

    def username_exists(self, username: str) -> bool:
        response = self._table.select("id").eq("username", username).limit(1).execute()
        return bool(response.data)

    def email_exists(self, email: str) -> bool:
        response = self._table.select("id").eq("email_id", email).limit(1).execute()
        return bool(response.data)

    def create_user(self, data: Mapping[str, Any]) -> UserRecord:
        payload = dict(data)
        response = self._table.insert(payload).execute()
        created = self._first(response.data)
        if created is None:
            raise RuntimeError("Insert into app_users returned no row")
        return created

    def update_user(self, user_id: str, updates: Mapping[str, Any]) -> None:
        payload = dict(updates)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._table.update(payload).eq("id", user_id).execute()

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
        query = self._table.select("*", count="exact")
        if status:
            query = query.eq("status", status)
        if persona:
            query = query.eq("persona", persona)
        if search:
            like = f"%{search}%"
            query = query.or_(
                f"username.ilike.{like},email_id.ilike.{like},"
                f"first_name.ilike.{like},last_name.ilike.{like}"
            )
        response = (
            query.order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        rows = [UserRecord(**row) for row in (response.data or [])]
        return rows, response.count or 0

    def deactivate(self, user_id: str) -> None:
        self.update_user(user_id, {"status": "deactivated"})

    def reactivate(self, user_id: str) -> None:
        self.update_user(user_id, {"status": "active", "failed_login_count": 0, "locked_until": None})

    def unlock(self, user_id: str) -> None:
        self.update_user(user_id, {"failed_login_count": 0, "locked_until": None})

    def set_persona(self, user_id: str, persona: str) -> None:
        self.update_user(user_id, {"persona": persona})

    def delete_user(self, user_id: str) -> None:
        """Hard delete — cascades to refresh_tokens/otp_codes/auth_audit in
        THIS (auth) project via FK ON DELETE CASCADE/SET NULL. Does not
        touch the pipeline project (jobs/insights/traces) — callers must
        clean those up separately before calling this."""
        self._table.delete().eq("id", user_id).execute()

    def _first(self, rows: list[dict[str, Any]] | None) -> UserRecord | None:
        if not rows:
            return None
        return UserRecord(**rows[0])
