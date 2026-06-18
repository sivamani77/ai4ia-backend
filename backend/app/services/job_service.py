import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.paths import JOBS_DIR, LOGS_DIR, ensure_runtime_dirs


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def create_job(job_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()

    job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    log_path = LOGS_DIR / "jobs" / f"{job_id}.log"

    job = {
        "job_id": job_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "payload": payload or {},
        "log_path": str(log_path),
        "result_path": None,
        "error": None,
    }

    save_job(job)
    return job


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def save_job(job: dict[str, Any]) -> None:
    ensure_runtime_dirs()
    job_path(job["job_id"]).write_text(
        json.dumps(job, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_job(job_id: str) -> dict[str, Any] | None:
    path = job_path(job_id)

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    job = get_job(job_id)

    if not job:
        raise FileNotFoundError(f"Job not found: {job_id}")

    job.update(updates)
    save_job(job)
    return job


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    ensure_runtime_dirs()

    jobs = []

    for path in sorted(JOBS_DIR.glob("job_*.json"), reverse=True):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    return jobs[:limit]
