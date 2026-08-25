from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.ingest_status")

VALID_STAGES = frozenset(
    ["discover", "download", "extract_audio", "transcribe", "create_fragments", "classify", "embed"]
)
VALID_STATUSES = frozenset(["pending", "running", "completed", "failed", "skipped"])


class IngestStatusRecord(TypedDict, total=False):
    id: int
    video_id: str
    stage: str
    status: str
    job_id: str | None
    error: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class IngestStatusRepository:
    """Ingest status rows live entirely in the cluster Postgres database."""

    table_name = "ingest_status"

    def upsert(self, data: Mapping[str, Any]) -> IngestStatusRecord:
        """Upsert a (video_id, stage) row. The UNIQUE constraint handles deduplication."""
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("video_id"), payload.get("stage"))

        row = pg_fetch_one(
            """
            insert into ingest_status (video_id, stage, status, job_id, error, metadata, updated_at)
            values (%(video_id)s, %(stage)s, %(status)s, %(job_id)s, %(error)s, %(metadata)s, now())
            on conflict (video_id, stage) do update set
                status = excluded.status,
                job_id = excluded.job_id,
                error = excluded.error,
                metadata = excluded.metadata,
                updated_at = now()
            returning *
            """,
            self._to_row_params(payload),
        )
        return self._normalize_output(row) if row is not None else self.get(payload["video_id"], payload["stage"])  # type: ignore[return-value]

    def get(self, video_id: str, stage: str) -> IngestStatusRecord | None:
        self._log("select_one", video_id, stage)
        row = pg_fetch_one(
            "select * from ingest_status where video_id = %s and stage = %s limit 1",
            (video_id, stage),
        )
        return self._normalize_output(row) if row is not None else None

    def get_for_video(self, video_id: str) -> list[IngestStatusRecord]:
        """All stage rows for a single video, ordered by stage definition order."""
        self._log("select_video", video_id)
        rows = pg_fetch_all("select * from ingest_status where video_id = %s", (video_id,))
        records = [self._normalize_output(row) for row in rows]
        stage_order = list(VALID_STAGES)
        return sorted(records, key=lambda r: stage_order.index(r["stage"]) if r["stage"] in stage_order else 99)

    def get_for_videos(self, video_ids: list[str]) -> list[IngestStatusRecord]:
        """Bulk fetch all stage rows for a list of video_ids."""
        if not video_ids:
            return []
        self._log("select_bulk")
        rows = pg_fetch_all("select * from ingest_status where video_id = any(%s)", (list(video_ids),))
        return [self._normalize_output(row) for row in rows]

    def list_by_stage_status(self, stage: str, status: str) -> list[IngestStatusRecord]:
        """Find all videos at a given stage+status (e.g. all failed downloads)."""
        self._log("select_stage_status")
        rows = pg_fetch_all(
            "select * from ingest_status where stage = %s and status = %s",
            (stage, status),
        )
        return [self._normalize_output(row) for row in rows]

    def get_completed_video_ids(self) -> set[str]:
        """Return set of video_ids where ALL 7 stages are completed — fully ingested videos."""
        rows = pg_fetch_all(
            "select video_id from ingest_status where stage = %s and status = %s",
            ("embed", "completed"),
        )
        return {row["video_id"] for row in rows}

    # -- row shaping ----------------------------------------------------------

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(payload)
        params["metadata"] = psycopg2.extras.Json(params.get("metadata") or {})
        return params

    def _normalize_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for k, v in dict(data).items():
            row[k] = normalize_value(v)

        if row.get("stage") not in VALID_STAGES:
            raise ValueError(f"Invalid ingest stage: {row.get('stage')!r}. Must be one of {sorted(VALID_STAGES)}")
        if row.get("status") and row.get("status") not in VALID_STATUSES:
            raise ValueError(f"Invalid ingest status: {row.get('status')!r}. Must be one of {sorted(VALID_STATUSES)}")

        row.setdefault("status", "pending")
        row.setdefault("job_id", None)
        row.setdefault("error", None)
        if not isinstance(row.get("metadata"), dict):
            row["metadata"] = {}

        return row

    def _normalize_output(self, data: Mapping[str, Any]) -> IngestStatusRecord:
        row = {k: normalize_value(v) for k, v in dict(data).items()}
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return IngestStatusRecord(
            id=row.get("id"),
            video_id=str(row.get("video_id", "")),
            stage=str(row.get("stage", "")),
            status=str(row.get("status", "pending")),
            job_id=row.get("job_id"),
            error=row.get("error"),
            metadata=metadata,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def _log(self, operation: str, video_id: str | None = None, stage: str | None = None) -> None:
        logger.debug(
            "Postgres repository operation: table=%s operation=%s video_id=%s stage=%s",
            self.table_name,
            operation,
            video_id,
            stage,
        )
