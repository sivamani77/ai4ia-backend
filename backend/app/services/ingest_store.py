from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.db.repositories.ingest_job_repository import IngestJobRepository, IngestJobRecord
from app.db.repositories.ingest_status_repository import IngestStatusRepository, IngestStatusRecord
from app.db.repositories.ingest_event_repository import IngestEventRepository, IngestEventRecord

logger = logging.getLogger("ai4ia.services.ingest_store")


@lru_cache(maxsize=1)
def _job_repo() -> IngestJobRepository:
    return IngestJobRepository()


@lru_cache(maxsize=1)
def _status_repo() -> IngestStatusRepository:
    return IngestStatusRepository()


@lru_cache(maxsize=1)
def _event_repo() -> IngestEventRepository:
    return IngestEventRepository()


# ---------------------------------------------------------------------------
# Ingest job persistence
# ---------------------------------------------------------------------------

def save_ingest_job(job_id: str, data: dict[str, Any]) -> None:
    repo = _job_repo()
    repo.upsert_job({
        "id": job_id,
        "status": data.get("status", "queued"),
        "channel_url": data.get("channel_url"),
        "video_limit": data.get("video_limit"),
        "progress": data.get("progress", 0),
        "message": data.get("message"),
        "error": data.get("error"),
        "metadata": data,
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
    })
    logger.info("Ingest job persisted id=%s status=%s", job_id, data.get("status"))


def load_ingest_job(job_id: str) -> dict[str, Any]:
    repo = _job_repo()
    row = repo.get_job(job_id)
    if not row:
        raise FileNotFoundError(f"Ingest job not found: {job_id}")
    data: dict[str, Any] = dict(row.get("metadata") or {})
    data["job_id"] = row["id"]
    data["status"] = row.get("status")
    data["progress"] = row.get("progress", 0)
    return data


def list_ingest_jobs(limit: int = 50) -> list[dict[str, Any]]:
    repo = _job_repo()
    rows = repo.list_jobs(limit=limit)
    result = []
    for row in rows:
        data: dict[str, Any] = dict(row.get("metadata") or {})
        data["job_id"] = row["id"]
        data["status"] = row.get("status")
        data["progress"] = row.get("progress", 0)
        result.append(data)
    return result


def list_running_ingest_jobs() -> list[IngestJobRecord]:
    return _job_repo().list_running_jobs()


# ---------------------------------------------------------------------------
# Per-video ingest status
# ---------------------------------------------------------------------------

def upsert_video_stage(
    video_id: str,
    stage: str,
    status: str,
    job_id: str | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestStatusRecord:
    return _status_repo().upsert({
        "video_id": video_id,
        "stage": stage,
        "status": status,
        "job_id": job_id,
        "error": error,
        "metadata": metadata or {},
    })


def get_video_stages(video_id: str) -> list[IngestStatusRecord]:
    return _status_repo().get_for_video(video_id)


def get_videos_stages_bulk(video_ids: list[str]) -> list[IngestStatusRecord]:
    return _status_repo().get_for_videos(video_ids)


def get_completed_video_ids() -> set[str]:
    return _status_repo().get_completed_video_ids()


def list_videos_by_stage_status(stage: str, status: str) -> list[IngestStatusRecord]:
    return _status_repo().list_by_stage_status(stage, status)


# ---------------------------------------------------------------------------
# Events (SSE)
# ---------------------------------------------------------------------------

def append_ingest_event(job_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    try:
        _event_repo().add_event(job_id, event_type, data or {})
    except Exception:
        logger.warning("Failed to append ingest event job_id=%s type=%s", job_id, event_type, exc_info=True)


def load_ingest_events(job_id: str) -> list[IngestEventRecord]:
    return _event_repo().get_events(job_id)


def load_ingest_events_after(job_id: str, after_id: int = 0) -> list[IngestEventRecord]:
    return _event_repo().get_events_after(job_id, after_id)
