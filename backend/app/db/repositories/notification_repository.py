from __future__ import annotations

import json
import logging
from typing import Any, Mapping, TypedDict

from app.db.postgres import pg_execute, pg_fetch_one, pg_fetch_all
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.notifications")


class NotificationRecord(TypedDict, total=False):
    id: str
    user_id: str
    message: str
    type: str
    redirect_page: str | None
    meta: dict
    read: bool
    created_at: str | None


class NotificationRepository:
    # Not "notifications" — that name already exists on UAT's Postgres with
    # an unrelated, unknown schema (no git history, no known owner). See the
    # comment in schema.py next to app_notifications for the full story.
    table_name = "app_notifications"

    def create(self, data: Mapping[str, Any]) -> NotificationRecord:
        payload = self._to_row(data)
        row = pg_fetch_one(
            """
            insert into app_notifications (id, user_id, message, type, redirect_page, meta, read)
            values (%(id)s, %(user_id)s, %(message)s, %(type)s, %(redirect_page)s, %(meta)s, %(read)s)
            returning *
            """,
            payload,
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def create_many(self, rows: list[Mapping[str, Any]]) -> int:
        """Notify several users at once (e.g. every admin on a new feedback
        submission). Row count here is always small (a handful of admins),
        so a plain loop over create() stays simple and correct rather than
        reaching for a batched insert."""
        for row in rows:
            self.create(row)
        return len(rows)

    def list_by_user(self, user_id: str, limit: int = 50) -> list[NotificationRecord]:
        rows = pg_fetch_all(
            "select * from app_notifications where user_id = %s order by created_at desc limit %s",
            (user_id, limit),
        )
        return [self._from_row(row) for row in rows]

    def mark_read(self, notification_id: str, user_id: str) -> bool:
        row = pg_fetch_one(
            "update app_notifications set read = true where id = %s and user_id = %s returning id",
            (notification_id, user_id),
        )
        return row is not None

    def mark_all_read(self, user_id: str) -> None:
        pg_execute("update app_notifications set read = true where user_id = %s and read = false", (user_id,))

    def delete_all_for_user(self, user_id: str) -> None:
        pg_execute("delete from app_notifications where user_id = %s", (user_id,))

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = normalize_value(dict(data))
        return {
            "id": d.get("id"),
            "user_id": d.get("user_id"),
            "message": d.get("message", ""),
            "type": d.get("type") or "info",
            "redirect_page": d.get("redirect_page"),
            "meta": json.dumps(d.get("meta") or {}),
            "read": bool(d.get("read", False)),
        }

    def _from_row(self, row: Mapping[str, Any]) -> NotificationRecord:
        d = normalize_value(dict(row))
        return NotificationRecord(
            id=d.get("id", ""),
            user_id=d.get("user_id", ""),
            message=d.get("message", ""),
            type=d.get("type", "info"),
            redirect_page=d.get("redirect_page"),
            meta=d.get("meta") or {},
            read=bool(d.get("read", False)),
            created_at=d.get("created_at"),
        )
