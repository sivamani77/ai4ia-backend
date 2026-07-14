from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.db.repositories.job_repository import JobRepository
from app.db.repositories.event_repository import EventRepository

logger = logging.getLogger("ai4ia.services.job_store")


@lru_cache(maxsize=1)
def get_job_repo() -> JobRepository:
    return JobRepository()


@lru_cache(maxsize=1)
def get_event_repo() -> EventRepository:
    return EventRepository()


def _to_supabase_job(job_id: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("payload", {})
    request = data.get("request", {})

    prompt = (
        request.get("prompt")
        or payload.get("prompt")
        or ""
    )

    return {
        "id": job_id,
        "status": data.get("status", "queued"),
        "progress": data.get("progress_percent", 0),
        "job_type": data.get("type", "video_processing"),
        "source_type": "prompt",
        "source_value": prompt,
        "user_id": data.get("user_id", ""),
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "completed_at": data.get("completed_at"),
        "error": data.get("error"),
        "metadata": data,
    }


def save_job(job_id: str, data: dict[str, Any]) -> None:
    repo = get_job_repo()
    row = _to_supabase_job(job_id, data)
    repo.upsert_job(row)
    logger.info("Job persisted id=%s status=%s", job_id, data.get("status", "unknown"))


def load_job(job_id: str) -> dict[str, Any]:
    repo = get_job_repo()

    row = repo.get_job(job_id)

    if not row:
        raise FileNotFoundError(f"Job not found: {job_id}")

    metadata = dict(row.get("metadata", {}) or {})

    metadata["job_id"] = row["id"]
    metadata["status"] = row.get("status")
    metadata["progress_percent"] = row.get("progress", 0)
    metadata["user_id"] = metadata.get("user_id") or row.get("user_id", "")

    logger.info("Job loaded id=%s", job_id)

    return metadata


def append_event(
    job_id: str,
    event_type: str,
    stage: str,
    payload: dict[str, Any],
) -> None:
    repo = get_event_repo()

    repo.add_event(
        {
            "job_id": job_id,
            "event_type": event_type,
            "stage": stage,
            "payload": payload,
        }
    )


def load_events(job_id: str) -> list[dict[str, Any]]:
    repo = get_event_repo()

    return repo.get_events(job_id)


def load_events_after(job_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    repo = get_event_repo()

    return repo.get_events_after(job_id, after_id)


def cancel_job(job_id: str) -> bool:
    """Mark a job as cancelled.

    Returns True if the job was in a cancellable state (queued or running),
    False if not found or already in a terminal state.

    Writes "cancelled" directly — the jobs table's status check constraint
    (db/init/001_schema.sql) does not allow an intermediate "cancelling" value.
    """
    try:
        job = load_job(job_id)
    except FileNotFoundError:
        return False

    if job.get("status") not in ("queued", "running"):
        return False

    repo = get_job_repo()
    repo.update_job(
        job_id,
        {"status": "cancelled", "metadata": {"status": "cancelled"}},
    )
    return True


def list_jobs() -> list[dict[str, Any]]:
    repo = get_job_repo()

    rows = repo.list_jobs()

    jobs = []

    for row in rows:
        metadata = dict(row.get("metadata", {}) or {})

        metadata["job_id"] = row["id"]
        metadata["status"] = row.get("status")
        metadata["progress_percent"] = row.get("progress", 0)
        metadata["user_id"] = metadata.get("user_id") or row.get("user_id", "")

        jobs.append(metadata)

    return jobs
