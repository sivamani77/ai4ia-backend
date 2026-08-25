from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.ingest_jobs")


class IngestJobRecord(TypedDict, total=False):
    id: str
    status: str
    channel_url: str | None
    video_limit: int | None
    progress: int
    message: str | None
    error: Any
    metadata: dict[str, Any]
    created_at: str | None
    started_at: str | None
    completed_at: str | None


class IngestJobRepository:
    """Ingest jobs live entirely in the cluster Postgres database."""

    table_name = "ingest_jobs"

    def create_job(self, data: Mapping[str, Any]) -> IngestJobRecord:
        payload = self._normalize_input(data)
        record_id = payload.get("id")
        self._log("insert", record_id)

        row = pg_fetch_one(
            """
            insert into ingest_jobs (id, status, channel_url, video_limit, progress, message,
                                     error, metadata, created_at, started_at, completed_at)
            values (%(id)s, %(status)s, %(channel_url)s, %(video_limit)s, %(progress)s, %(message)s,
                    %(error)s, %(metadata)s, coalesce(%(created_at)s, now()), %(started_at)s, %(completed_at)s)
            returning *
            """,
            self._to_row_params(payload),
        )
        if row is not None:
            return self._normalize_output(row)
        return self.get_job(str(record_id))

    def get_job(self, record_id: str) -> IngestJobRecord | None:
        self._log("select_one", record_id)
        row = pg_fetch_one("select * from ingest_jobs where id = %s limit 1", (record_id,))
        return self._normalize_output(row) if row is not None else None

    def upsert_job(self, data: Mapping[str, Any]) -> None:
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("id"))
        pg_execute(
            """
            insert into ingest_jobs (id, status, channel_url, video_limit, progress, message,
                                     error, metadata, created_at, started_at, completed_at)
            values (%(id)s, %(status)s, %(channel_url)s, %(video_limit)s, %(progress)s, %(message)s,
                    %(error)s, %(metadata)s, coalesce(%(created_at)s, now()), %(started_at)s, %(completed_at)s)
            on conflict (id) do update set
                status = excluded.status,
                channel_url = excluded.channel_url,
                video_limit = excluded.video_limit,
                progress = excluded.progress,
                message = excluded.message,
                error = excluded.error,
                metadata = excluded.metadata,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at
            """,
            self._to_row_params(payload),
        )

    def list_jobs(self, limit: int = 50) -> list[IngestJobRecord]:
        self._log("select_many")
        rows = pg_fetch_all("select * from ingest_jobs order by created_at desc limit %s", (limit,))
        return [self._normalize_output(row) for row in rows]

    def list_running_jobs(self) -> list[IngestJobRecord]:
        """Return jobs in queued or running state (used for orphan reaping)."""
        self._log("select_running")
        rows = pg_fetch_all(
            "select * from ingest_jobs where status = any(%s) order by created_at desc",
            (["queued", "running"],),
        )
        return [self._normalize_output(row) for row in rows]

    # -- row shaping ----------------------------------------------------------

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(payload)
        params["metadata"] = psycopg2.extras.Json(params.get("metadata") or {})
        if params.get("error") is not None:
            params["error"] = psycopg2.extras.Json(params["error"])
        return params

    def _normalize_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        row = {k: normalize_value(v) for k, v in dict(data).items()}
        if "progress" in row:
            try:
                row["progress"] = int(row["progress"])
            except (TypeError, ValueError):
                row["progress"] = 0
        row.setdefault("metadata", {})
        return row

    def _normalize_output(self, data: Mapping[str, Any]) -> IngestJobRecord:
        row = {k: normalize_value(v) for k, v in dict(data).items()}
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return IngestJobRecord(
            id=str(row.get("id", "")),
            status=row.get("status", "queued"),
            channel_url=row.get("channel_url"),
            video_limit=row.get("video_limit"),
            progress=int(row.get("progress") or 0),
            message=row.get("message"),
            error=row.get("error"),
            metadata=metadata,
            created_at=row.get("created_at"),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug(
            "Postgres repository operation: table=%s operation=%s id=%s",
            self.table_name,
            operation,
            record_id,
        )
