from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.events")


class EventRecord(TypedDict, total=False):
    job_id: str
    created_at: str
    event_type: str
    stage: str
    payload: dict[str, Any]


class EventRepository:
    table_name = "job_events"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def add_event(self, data: Mapping[str, Any]) -> EventRecord:
        payload = self._normalize_input(data)
        job_id = payload.get("job_id")
        self._log("insert", job_id)

        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception(
                "Supabase event operation failed: table=%s operation=%s job_id=%s",
                self.table_name,
                "insert",
                job_id,
            )
            raise

        if not response.data:
            raise RuntimeError(
                f"Supabase returned no rows for job_events insert operation (job_id={job_id})"
            )
        return self._normalize_output(response.data[0])

    def get_events(self, job_id: str) -> list[dict[str, Any]]:
        self._log("select_many", job_id)

        try:
            response = (
                self._table.select("*")
                .eq("job_id", job_id)
                .order("created_at", desc=False)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase event operation failed: table=%s operation=%s job_id=%s",
                self.table_name,
                "select_many",
                job_id,
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def get_events_after(self, job_id: str, after_id: int | None = 0) -> list[dict[str, Any]]:
        """Return events for job_id with id > after_id, ordered by id ascending.

        If after_id is 0 or None, returns all events (same as get_events()).
        The id column is an auto-increment integer, so .gt() gives a stable cursor.
        """
        if not after_id:
            return self.get_events(job_id)

        self._log("select_after", job_id)

        try:
            response = (
                self._table.select("*")
                .eq("job_id", job_id)
                .gt("id", after_id)
                .order("id", desc=False)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase event operation failed: table=%s operation=%s job_id=%s",
                self.table_name,
                "select_after",
                job_id,
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def _log(self, operation: str, job_id: str | None = None) -> None:
        logger.debug(
            "Supabase repository operation: table=%s operation=%s job_id=%s",
            self.table_name,
            operation,
            job_id,
        )

    def _normalize_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            key: self._normalize_value(value)
            for key, value in dict(data).items()
        }
        # `stage` is not a column in job_events — fold it into payload so
        # the pipeline's stage context is preserved without a schema change.
        if "stage" in row:
            stage_val = row.pop("stage")
            payload = row.get("payload")
            if isinstance(payload, dict):
                payload.setdefault("stage", stage_val)
            else:
                row["payload"] = {"stage": stage_val}
        return row

    def _normalize_output(self, data: Mapping[str, Any]) -> EventRecord:
        return EventRecord(
            **{
                key: self._normalize_value(value)
                for key, value in dict(data).items()
            }
        )

    def _normalize_value(self, value: Any) -> Any:
        return normalize_value(value)
