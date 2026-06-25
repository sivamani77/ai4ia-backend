# DEPRECATED: This module is a compatibility shim. All callers should import
# directly from app.services.job_store. This file is kept only in case anything
# outside backend/ still references it. Do not add logic here.

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.paths import LOGS_DIR, ensure_runtime_dirs
from app.services.job_store import (
    save_job as save_job_db,
    load_job as load_job_db,
    list_jobs as list_jobs_db,
)


def create_job(job_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "progress_percent": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "payload": payload or {},
        "log_path": str(LOGS_DIR / "jobs" / f"{job_id}.log"),
        "result_path": None,
        "error": None,
    }
    save_job_db(job_id, job)
    return job


def save_job(job: dict[str, Any]) -> None:
    save_job_db(job["job_id"], job)


def get_job(job_id: str) -> dict[str, Any] | None:
    try:
        return load_job_db(job_id)
    except FileNotFoundError:
        return None


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    job = load_job_db(job_id)
    job.update(updates)
    save_job_db(job_id, job)
    return job


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    return list_jobs_db()[:limit]
