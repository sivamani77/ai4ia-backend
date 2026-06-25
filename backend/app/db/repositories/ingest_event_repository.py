from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.ingest_events")


class IngestEventRecord(TypedDict, total=False):
    id: int
    job_id: str
    type: str
    data: dict[str, Any]
    created_at: str


class IngestEventRepository:
    table_name = "ingest_job_events"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def add_event(self, job_id: str, event_type: str, data: dict[str, Any] | None = None) -> IngestEventRecord:
        payload = {
            "job_id": job_id,
            "type": event_type,
            "data": normalize_value(data or {}),
        }
        self._log("insert", job_id)

        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception(
                "Supabase ingest_job_events operation failed: operation=insert job_id=%s type=%s",
                job_id,
                event_type,
            )
            raise

        if not response.data:
            raise RuntimeError(
                f"Supabase returned no rows for ingest_job_events insert (job_id={job_id})"
            )
        return self._normalize_output(response.data[0])

    def get_events(self, job_id: str) -> list[IngestEventRecord]:
        self._log("select_many", job_id)

        try:
            response = (
                self._table.select("*")
                .eq("job_id", job_id)
                .order("id", desc=False)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_job_events operation failed: operation=select_many job_id=%s",
                job_id,
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def get_events_after(self, job_id: str, after_id: int | None = 0) -> list[IngestEventRecord]:
        """Cursor-based fetch for SSE streaming. Returns events with id > after_id."""
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
                "Supabase ingest_job_events operation failed: operation=select_after job_id=%s after_id=%s",
                job_id,
                after_id,
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def _normalize_output(self, data: Mapping[str, Any]) -> IngestEventRecord:
        row = {k: normalize_value(v) for k, v in dict(data).items()}
        event_data = row.get("data")
        if not isinstance(event_data, dict):
            event_data = {}
        return IngestEventRecord(
            id=row.get("id"),
            job_id=str(row.get("job_id", "")),
            type=str(row.get("type", "")),
            data=event_data,
            created_at=row.get("created_at"),
        )

    def _log(self, operation: str, job_id: str | None = None) -> None:
        logger.debug(
            "Supabase repository operation: table=%s operation=%s job_id=%s",
            self.table_name,
            operation,
            job_id,
        )
