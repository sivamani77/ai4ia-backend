from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
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
    table_name = "ingest_jobs"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create_job(self, data: Mapping[str, Any]) -> IngestJobRecord:
        payload = self._normalize_input(data)
        record_id = payload.get("id")
        self._log("insert", record_id)

        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception(
                "Supabase ingest_jobs operation failed: operation=insert id=%s", record_id
            )
            raise

        created = self._extract_optional_single(response.data)
        if created is not None:
            return created
        return self.get_job(str(record_id))

    def get_job(self, record_id: str) -> IngestJobRecord | None:
        self._log("select_one", record_id)

        try:
            response = (
                self._table.select("*").eq("id", record_id).limit(1).execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_jobs operation failed: operation=select_one id=%s", record_id
            )
            raise

        return self._extract_optional_single(response.data)

    def upsert_job(self, data: Mapping[str, Any]) -> None:
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("id"))

        try:
            self._table.upsert(payload, on_conflict="id").execute()
        except Exception:
            logger.exception(
                "Supabase ingest_jobs operation failed: operation=upsert id=%s", payload.get("id")
            )
            raise

    def list_jobs(self, limit: int = 50) -> list[IngestJobRecord]:
        self._log("select_many")

        try:
            response = (
                self._table.select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("Supabase ingest_jobs operation failed: operation=select_many")
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def list_running_jobs(self) -> list[IngestJobRecord]:
        """Return jobs in queued or running state (used for orphan reaping)."""
        self._log("select_running")

        try:
            response = (
                self._table.select("*")
                .in_("status", ["queued", "running"])
                .order("created_at", desc=True)
                .execute()
            )
        except Exception:
            logger.exception("Supabase ingest_jobs operation failed: operation=select_running")
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def _extract_optional_single(
        self, rows: list[dict[str, Any]] | None
    ) -> IngestJobRecord | None:
        if not rows:
            return None
        return self._normalize_output(rows[0])

    def _normalize_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        row = {k: normalize_value(v) for k, v in dict(data).items()}
        # Coerce progress to int
        if "progress" in row:
            try:
                row["progress"] = int(row["progress"])
            except (TypeError, ValueError):
                row["progress"] = 0
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
            "Supabase repository operation: table=%s operation=%s id=%s",
            self.table_name,
            operation,
            record_id,
        )
