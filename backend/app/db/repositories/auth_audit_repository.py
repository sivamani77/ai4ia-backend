from __future__ import annotations

import logging
from typing import Any

from app.db.supabase import get_auth_supabase_client

logger = logging.getLogger("ai4ia.db.repositories.auth_audit")


class AuthAuditRepository:
    """auth_audit lives in the auth Supabase project alongside app_users."""

    table_name = "auth_audit"

    def __init__(self) -> None:
        self._client = get_auth_supabase_client()
        self._table = self._client.table(self.table_name)

    def log(
        self,
        *,
        event: str,
        success: bool,
        user_id: str | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": event,
            "success": success,
            "user_id": user_id,
            "ip": ip,
            "user_agent": user_agent,
        }
        try:
            self._table.insert(payload).execute()
        except Exception:
            # Audit logging must never break the auth flow it's observing.
            logger.exception("Failed to write auth_audit row: event=%s user_id=%s", event, user_id)
