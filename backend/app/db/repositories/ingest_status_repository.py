from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
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
    table_name = "ingest_status"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def upsert(self, data: Mapping[str, Any]) -> IngestStatusRecord:
        """Upsert a (video_id, stage) row. The UNIQUE constraint handles deduplication."""
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("video_id"), payload.get("stage"))

        try:
            response = (
                self._table
                .upsert(payload, on_conflict="video_id,stage")
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=upsert video_id=%s stage=%s",
                payload.get("video_id"),
                payload.get("stage"),
            )
            raise

        if response.data:
            return self._normalize_output(response.data[0])
        return self.get(payload["video_id"], payload["stage"])

    def get(self, video_id: str, stage: str) -> IngestStatusRecord | None:
        self._log("select_one", video_id, stage)

        try:
            response = (
                self._table.select("*")
                .eq("video_id", video_id)
                .eq("stage", stage)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=select_one video_id=%s stage=%s",
                video_id,
                stage,
            )
            raise

        if not response.data:
            return None
        return self._normalize_output(response.data[0])

    def get_for_video(self, video_id: str) -> list[IngestStatusRecord]:
        """All stage rows for a single video, ordered by stage definition order."""
        self._log("select_video", video_id)

        try:
            response = (
                self._table.select("*")
                .eq("video_id", video_id)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=select_video video_id=%s",
                video_id,
            )
            raise

        rows = [self._normalize_output(row) for row in response.data or []]
        stage_order = list(VALID_STAGES)
        return sorted(rows, key=lambda r: stage_order.index(r["stage"]) if r["stage"] in stage_order else 99)

    def get_for_videos(self, video_ids: list[str]) -> list[IngestStatusRecord]:
        """Bulk fetch all stage rows for a list of video_ids."""
        if not video_ids:
            return []
        self._log("select_bulk")

        try:
            response = (
                self._table.select("*")
                .in_("video_id", video_ids)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=select_bulk count=%d",
                len(video_ids),
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def list_by_stage_status(self, stage: str, status: str) -> list[IngestStatusRecord]:
        """Find all videos at a given stage+status (e.g. all failed downloads)."""
        self._log("select_stage_status")

        try:
            response = (
                self._table.select("*")
                .eq("stage", stage)
                .eq("status", status)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=select_stage_status stage=%s status=%s",
                stage,
                status,
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def get_completed_video_ids(self) -> set[str]:
        """Return set of video_ids where ALL 7 stages are completed — fully ingested videos."""
        try:
            response = (
                self._table.select("video_id")
                .eq("stage", "embed")
                .eq("status", "completed")
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase ingest_status operation failed: operation=get_completed_video_ids"
            )
            raise

        return {row["video_id"] for row in response.data or []}

    def _normalize_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for k, v in dict(data).items():
            row[k] = normalize_value(v)

        if row.get("stage") not in VALID_STAGES:
            raise ValueError(f"Invalid ingest stage: {row.get('stage')!r}. Must be one of {sorted(VALID_STAGES)}")
        if row.get("status") and row.get("status") not in VALID_STATUSES:
            raise ValueError(f"Invalid ingest status: {row.get('status')!r}. Must be one of {sorted(VALID_STATUSES)}")

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
            "Supabase repository operation: table=%s operation=%s video_id=%s stage=%s",
            self.table_name,
            operation,
            video_id,
            stage,
        )
