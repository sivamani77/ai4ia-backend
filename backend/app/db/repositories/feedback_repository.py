from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.postgres import pg_execute, pg_fetch_one, pg_fetch_all
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.feedbacks")


class FeedbackRecord(TypedDict, total=False):
    id: str
    user_id: str
    username: str
    email: str
    feedback: str
    status: str
    resolution_message: str | None
    resolved_at: str | None
    resolved_by: str | None
    created_at: str | None


class FeedbackRepository:
    table_name = "feedbacks"

    def create(self, data: Mapping[str, Any]) -> FeedbackRecord:
        payload = self._to_row(data)
        row = pg_fetch_one(
            """
            insert into feedbacks (id, user_id, username, email, feedback, status)
            values (%(id)s, %(user_id)s, %(username)s, %(email)s, %(feedback)s, %(status)s)
            returning *
            """,
            payload,
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        row = pg_fetch_one("select * from feedbacks where id = %s", (feedback_id,))
        return self._from_row(row) if row is not None else None

    def list(self, status: str | None = None, limit: int = 200) -> list[FeedbackRecord]:
        if status:
            rows = pg_fetch_all(
                "select * from feedbacks where status = %s order by created_at desc limit %s",
                (status, limit),
            )
        else:
            rows = pg_fetch_all(
                "select * from feedbacks order by created_at desc limit %s",
                (limit,),
            )
        return [self._from_row(row) for row in rows]

    def resolve(self, feedback_id: str, *, resolution_message: str | None, resolved_by: str) -> FeedbackRecord | None:
        row = pg_fetch_one(
            """
            update feedbacks
            set status = 'resolved', resolution_message = %s, resolved_by = %s, resolved_at = now()
            where id = %s
            returning *
            """,
            (resolution_message, resolved_by, feedback_id),
        )
        return self._from_row(row) if row is not None else None

    def delete_by_user(self, user_id: str) -> None:
        pg_execute("delete from feedbacks where user_id = %s", (user_id,))

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = normalize_value(dict(data))
        return {
            "id": d.get("id"),
            "user_id": d.get("user_id"),
            "username": d.get("username", ""),
            "email": d.get("email", ""),
            "feedback": d.get("feedback", ""),
            "status": d.get("status", "open"),
        }

    def _from_row(self, row: Mapping[str, Any]) -> FeedbackRecord:
        d = normalize_value(dict(row))
        return FeedbackRecord(
            id=d.get("id", ""),
            user_id=d.get("user_id", ""),
            username=d.get("username", ""),
            email=d.get("email", ""),
            feedback=d.get("feedback", ""),
            status=d.get("status", "open"),
            resolution_message=d.get("resolution_message"),
            resolved_at=d.get("resolved_at"),
            resolved_by=d.get("resolved_by"),
            created_at=d.get("created_at"),
        )
