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


def _to_job_row(job_id: str, data: dict[str, Any]) -> dict[str, Any]:
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
    row = _to_job_row(job_id, data)
    repo.upsert_job(row)
    logger.info("Job persisted id=%s status=%s", job_id, data.get("status", "unknown"))


def create_job_within_quota(
    job_id: str, data: dict[str, Any], *, user_id: str, max_running: int, max_queued: int, max_active: int,
) -> str | None:
    """Save a new job only if the quota allows it; None on success, otherwise
    the limit that was hit (see JobRepository.create_job_within_quota)."""
    row = _to_job_row(job_id, data)
    reason = get_job_repo().create_job_within_quota(
        row, user_id=user_id, max_running=max_running, max_queued=max_queued, max_active=max_active,
    )
    if reason is None:
        logger.info("Job persisted id=%s status=%s", job_id, data.get("status", "unknown"))
    return reason


def list_user_jobs(user_id: str, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    """One page of a user's jobs (newest first) and the user's total."""
    rows, total = get_job_repo().list_by_user_page(user_id, limit, offset)
    return [_row_to_metadata(row) for row in rows], total


def queue_position(job_id: str) -> int:
    return get_job_repo().queue_position(job_id)


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


def _row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}) or {})

    metadata["job_id"] = row["id"]
    metadata["status"] = row.get("status")
    metadata["progress_percent"] = row.get("progress", 0)
    metadata["user_id"] = metadata.get("user_id") or row.get("user_id", "")

    return metadata


def list_jobs() -> list[dict[str, Any]]:
    repo = get_job_repo()

    rows = repo.list_jobs()

    return [_row_to_metadata(row) for row in rows]


def list_by_user(user_id: str) -> list[dict[str, Any]]:
    """All jobs (any status) created by a single user, newest first.

    Used by the admin per-user generation feed so an admin can see queued/
    running/failed jobs too, not just completed ones.
    """
    repo = get_job_repo()

    rows = repo.list_by_user(user_id)
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)

    return [_row_to_metadata(row) for row in rows]
