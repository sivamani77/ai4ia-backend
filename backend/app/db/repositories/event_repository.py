from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_one, pg_fetch_all
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

    def delete_by_job(self, job_id: str) -> None:
        pg_execute("delete from job_events where job_id = %s", (job_id,))

    def add_event(self, data: Mapping[str, Any]) -> EventRecord:
        payload = self._normalize_input(data)
        job_id = payload.get("job_id")
        self._log("insert", job_id)

        row = pg_fetch_one(
            """
            insert into job_events (job_id, event_type, payload)
            values (%(job_id)s, %(event_type)s, %(payload)s)
            returning *
            """,
            {
                "job_id": payload.get("job_id"),
                "event_type": payload.get("event_type"),
                "payload": psycopg2.extras.Json(payload.get("payload") or {}),
            },
        )
        if row is None:
            raise RuntimeError(
                f"Postgres returned no row for job_events insert operation (job_id={job_id})"
            )
        return self._normalize_output(row)

    def get_events(self, job_id: str) -> list[dict[str, Any]]:
        self._log("select_many", job_id)
        rows = pg_fetch_all(
            "select * from job_events where job_id = %s order by created_at asc",
            (job_id,),
        )
        return [self._normalize_output(row) for row in rows]

    def payload_stages(self, job_id: str, event_type: str) -> list[str]:
        """The payload "stage" of each event of one type for a job, oldest
        first. The table has no stage column, so the stage must be in the payload."""
        rows = pg_fetch_all(
            "select payload->>'stage' as stage from job_events"
            " where job_id = %s and event_type = %s order by id asc",
            (job_id, event_type),
        )
        return [row["stage"] for row in rows]

    def get_events_after(self, job_id: str, after_id: int | None = 0) -> list[dict[str, Any]]:
        """Return events for job_id with id > after_id, ordered by id ascending.

        If after_id is 0 or None, returns all events (same as get_events()).
        The id column is an auto-increment integer, so > gives a stable cursor.
        """
        if not after_id:
            return self.get_events(job_id)

        self._log("select_after", job_id)
        rows = pg_fetch_all(
            "select * from job_events where job_id = %s and id > %s order by id asc",
            (job_id, after_id),
        )
        return [self._normalize_output(row) for row in rows]

    def _log(self, operation: str, job_id: str | None = None) -> None:
        logger.debug(
            "Postgres repository operation: table=%s operation=%s job_id=%s",
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
