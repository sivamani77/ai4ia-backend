"""
Celery tasks for the AI4IA video pipeline.

Tasks:
  - run_video_job_task: full documentary video generation pipeline
  - run_ingest_job_task: YouTube channel ingestion pipeline (scripts 01-06)
"""
from __future__ import annotations

import logging
from typing import Any

from app.worker.celery_app import celery_app
from app.services.video_job_runner import run_video_job
from app.services.ingest_job_runner import run_ingest_job

logger = logging.getLogger("ai4ia.worker")

# A job heavy enough to OOM-kill its worker is redelivered by the broker when
# the worker restarts (task_acks_late + task_reject_on_worker_lost), OOMs
# again, and loops forever — max_retries governs only explicit self.retry(),
# never broker-level redelivery. The pipeline is not idempotent, so every pass
# re-runs LLM calls and re-uploads to S3. Allow one retry after a worker loss,
# then refuse the job as a poison pill.
_MAX_DELIVERIES = 2

_TERMINAL_STATUSES = {"cancelled", "completed", "failed"}


def _claim_delivery(job_id: str) -> bool:
    """Record this delivery attempt on the job row.

    Returns False if the job must not run: either it is already terminal, or
    it has exhausted _MAX_DELIVERIES and is being abandoned. The counter lives
    on the job row rather than in the worker because the worker process is
    exactly what does not survive an OOM kill.
    """
    from app.core import metrics as _metrics
    from app.services.job_store import load_job, save_job
    from app.services.video_job_runner import _fail_job

    try:
        job = load_job(job_id)
    except Exception:  # noqa: BLE001
        # The counter is a safety net, not a correctness gate — a transient
        # store failure here must not block a legitimate job. run_video_job
        # does its own load and will fail properly if the store is really down.
        logger.warning("[%s] Could not read job row for delivery guard", job_id)
        return True

    if job.get("status") in _TERMINAL_STATUSES:
        logger.info("[%s] Skipping delivery — already terminal (status=%s)",
                    job_id, job.get("status"))
        return False

    attempts = int(job.get("delivery_attempts", 0)) + 1
    job["delivery_attempts"] = attempts
    save_job(job_id, job)

    if attempts > _MAX_DELIVERIES:
        logger.error(
            "[%s] Abandoning job — delivered %d times, worker lost on each. "
            "Likely OOM or a crash the pipeline cannot survive.",
            job_id, attempts,
        )
        _fail_job(
            job_id,
            "dispatch",
            "poison_pill",
            f"Abandoned after {attempts} deliveries — the worker died on each "
            "attempt, so the job was refused instead of retried indefinitely.",
        )
        _metrics.jobs_abandoned_total.inc()
        return False

    if attempts > 1:
        logger.warning("[%s] Redelivered by broker — delivery attempt %d of %d",
                       job_id, attempts, _MAX_DELIVERIES)
    return True


@celery_app.task(
    name="ai4ia.run_video_job",
    bind=True,
    max_retries=0,          # pipeline failures are surfaced via job status, not Celery retries
    acks_late=True,
    queue="video_jobs",
    track_started=True,
)
def run_video_job_task(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Execute the full video generation pipeline for *job_id*."""
    logger.info("[%s] Celery worker starting pipeline", job_id)
    if not _claim_delivery(job_id):
        return {"job_id": job_id, "status": "abandoned"}
    try:
        run_video_job(job_id, request)
        from app.services.job_store import load_job
        job = load_job(job_id)
        status = job.get("status", "completed") if job else "completed"
        if status == "failed":
            logger.error("[%s] Pipeline execution ended with status 'failed'", job_id)
        else:
            logger.info("[%s] Pipeline completed successfully", job_id)
        return {"job_id": job_id, "status": status}
    except Exception as exc:
        logger.exception("[%s] Pipeline raised unhandled exception", job_id)
        raise exc


@celery_app.task(
    name="ai4ia.run_ingest_job",
    bind=True,
    max_retries=0,
    acks_late=True,
    queue="ingest_jobs",
    track_started=True,
)
def run_ingest_job_task(self, job_id: str) -> dict[str, Any]:
    """Execute the YouTube channel ingestion pipeline for *job_id*."""
    logger.info("[%s] Celery worker starting ingest pipeline", job_id)
    try:
        run_ingest_job(job_id)
        logger.info("[%s] Ingest pipeline completed successfully", job_id)
        return {"job_id": job_id, "status": "completed"}
    except Exception as exc:
        logger.exception("[%s] Ingest pipeline raised unhandled exception", job_id)
        raise exc
