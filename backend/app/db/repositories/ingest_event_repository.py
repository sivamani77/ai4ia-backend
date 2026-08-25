from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.db.postgres import pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.ingest_events")


class IngestEventRecord(TypedDict, total=False):
    id: int
    job_id: str
    type: str
    data: dict[str, Any]
    created_at: str


class IngestEventRepository:
    """Ingest job events live entirely in the cluster Postgres database."""

    table_name = "ingest_job_events"

    def add_event(self, job_id: str, event_type: str, data: dict[str, Any] | None = None) -> IngestEventRecord:
        self._log("insert", job_id)
        row = pg_fetch_one(
            """
            insert into ingest_job_events (job_id, type, data)
            values (%(job_id)s, %(type)s, %(data)s)
            returning *
            """,
            {
                "job_id": job_id,
                "type": event_type,
                "data": psycopg2.extras.Json(normalize_value(data or {})),
            },
        )
        if row is None:
            raise RuntimeError(f"Postgres returned no rows for ingest_job_events insert (job_id={job_id})")
        return self._normalize_output(row)

    def get_events(self, job_id: str) -> list[IngestEventRecord]:
        self._log("select_many", job_id)
        rows = pg_fetch_all(
            "select * from ingest_job_events where job_id = %s order by id asc",
            (job_id,),
        )
        return [self._normalize_output(row) for row in rows]

    def get_events_after(self, job_id: str, after_id: int | None = 0) -> list[IngestEventRecord]:
        """Cursor-based fetch for SSE streaming. Returns events with id > after_id."""
        if not after_id:
            return self.get_events(job_id)

        self._log("select_after", job_id)
        rows = pg_fetch_all(
            "select * from ingest_job_events where job_id = %s and id > %s order by id asc",
            (job_id, after_id),
        )
        return [self._normalize_output(row) for row in rows]

    # -- row shaping ----------------------------------------------------------

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
            "Postgres repository operation: table=%s operation=%s job_id=%s",
            self.table_name,
            operation,
            job_id,
        )
