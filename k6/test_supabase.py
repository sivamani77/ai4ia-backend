from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from app.db import get_event_repository, get_job_repository, get_supabase_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai4ia.test.test_supabase")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_test_job(record_id: str) -> dict[str, object]:
    timestamp = utc_now()
    return {
        "id": record_id,
        "status": "queued",
        "source_type": "prompt",
        "source_value": "Supabase repository validation",
        "progress": 0,
        "created_at": timestamp,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "metadata": {
            "message": "Supabase validation job",
            "request": {"prompt": "Supabase repository validation"},
            "legacy_context": "test/test_supabase.py",
        },
        "job_type": "supabase_validation",
        "worker_id": "manual-test",
        "retry_count": 0,
    }


def build_test_event(record_id: str) -> dict[str, object]:
    return {
        "job_id": record_id,
        "created_at": utc_now(),
        "event_type": "validation_event",
        "stage": "supabase_test",
        "payload": {
            "message": "Supabase repository validation event",
        },
    }


def main() -> int:
    logger.info("Checking Supabase configuration")
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set before running this script."
        )

    client = get_supabase_client()
    logger.info("Supabase client initialized: %s", client.__class__.__name__)

    job_repository = get_job_repository()
    event_repository = get_event_repository()

    record_id = f"test_supabase_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    created_job = build_test_job(record_id)

    logger.info("Creating test job: %s", record_id)
    inserted = job_repository.create_job(created_job)
    assert inserted["id"] == record_id, "Inserted id mismatch"

    logger.info("Fetching test job: %s", record_id)
    fetched = job_repository.get_job(record_id)
    assert fetched is not None, "Expected inserted job to be returned"
    assert fetched["id"] == record_id, "Fetched id mismatch"

    logger.info("Updating test job: %s", record_id)
    updated = job_repository.update_job(
        record_id,
        {
            "status": "running",
            "progress": 25,
            "metadata": {
                "message": "Updated by test/test_supabase.py",
                "current_stage": "validation",
            },
            "worker_id": "manual-test-update",
            "retry_count": 1,
        },
    )
    assert updated is not None, "Expected updated job to be returned"
    assert updated["status"] == "running", "Updated job status mismatch"
    assert updated["progress"] == 25, "Updated progress mismatch"

    logger.info("Listing jobs")
    jobs = job_repository.list_jobs()
    assert any(job.get("id") == record_id for job in jobs), "Test job not found in list_jobs()"

    logger.info("Adding test event: %s", record_id)
    inserted_event = event_repository.add_event(build_test_event(record_id))
    assert inserted_event["job_id"] == record_id, "Inserted event job_id mismatch"

    logger.info("Fetching test events: %s", record_id)
    events = event_repository.get_events(record_id)
    assert events, "Expected at least one event for test job"
    assert events[-1]["job_id"] == record_id, "Fetched event job_id mismatch"

    logger.info("Cleaning up test rows")
    client.table("job_events").delete().eq("job_id", record_id).execute()
    deleted = job_repository.delete_job(record_id)
    assert deleted is True, "delete_job() did not return True"

    logger.info("Supabase validation completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
