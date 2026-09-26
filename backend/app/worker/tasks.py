"""
Celery tasks for the AI4IA video pipeline.

Tasks:
  - run_video_job_task: full documentary video generation pipeline
  - run_ingest_job_task: YouTube channel ingestion pipeline (scripts 01-06)
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.settings import settings
from app.worker.celery_app import celery_app
from app.services.video_job_runner import run_video_job
from app.services.ingest_job_runner import run_ingest_job

logger = logging.getLogger("ai4ia.worker")

# An ingest job heavy enough to OOM-kill its worker is redelivered by the
# broker when the worker restarts (task_acks_late + task_reject_on_worker_lost),
# OOMs again, and loops forever — max_retries governs only explicit
# self.retry(), never broker-level redelivery. Allow one retry after a worker
# loss, then refuse the job as a poison pill. Video jobs use leases instead
# (app.services.job_lease).
_MAX_DELIVERIES = 2

_TERMINAL_STATUSES = {"cancelled", "completed", "failed"}


def _claim_ingest_delivery(job_id: str) -> bool:
    """Record this ingest delivery attempt; False if the job must not run.

    An ingest job downloading tens of GB of video can OOM-kill its worker,
    and the broker redelivers it on every restart (acks_late +
    task_reject_on_worker_lost). The counter lives on the job row, since the
    worker process is what doesn't survive. Ingest jobs have their own store
    (app.services.ingest_store) and don't use video job leases yet.
    """
    from app.core import metrics as _metrics
    from app.services.ingest_store import load_ingest_job, save_ingest_job
    from app.services.ingest_job_runner import _fail_job

    try:
        job = load_ingest_job(job_id)
    except Exception:  # noqa: BLE001
        logger.warning("[%s] Could not read ingest job row for delivery guard", job_id)
        return True

    if job.get("status") in _TERMINAL_STATUSES:
        logger.info("[%s] Skipping ingest delivery — already terminal (status=%s)",
                    job_id, job.get("status"))
        return False

    attempts = int(job.get("delivery_attempts", 0)) + 1
    job["delivery_attempts"] = attempts
    save_ingest_job(job_id, job)

    if attempts > _MAX_DELIVERIES:
        logger.error(
            "[%s] Abandoning ingest job — delivered %d times, worker lost on each.",
            job_id, attempts,
        )
        _fail_job(
            job_id,
            "dispatch",
            f"Abandoned after {attempts} deliveries — the worker died on each "
            "attempt, so the job was refused instead of retried indefinitely.",
        )
        _metrics.jobs_abandoned_total.inc()
        return False

    if attempts > 1:
        logger.warning("[%s] Ingest job redelivered by broker — delivery attempt %d of %d",
                       job_id, attempts, _MAX_DELIVERIES)
    return True


@celery_app.task(
    name="ai4ia.run_video_job",
    bind=True,
    max_retries=0,          # pipeline failures are surfaced via job status, not Celery retries
    acks_late=True,
    # Not a literal: release and uat share one Redis, so both workers consume
    # the same queue and whichever claims the task first runs the job. Routing
    # by settings lets each environment own a queue name.
    queue=settings.worker_queue,
    track_started=True,
    # Backstop for a hung run: the job's own limit (job_max_runtime_seconds)
    # fails it first. Stays below the broker visibility timeout, so a running
    # job is never handed to a second worker.
    time_limit=settings.job_max_runtime_seconds + 300,
    soft_time_limit=settings.job_max_runtime_seconds + 240,
)
def run_video_job_task(
    self, job_id: str, request: dict[str, Any], dispatch_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full video generation pipeline for *job_id*.

    The delivery runs only if it wins the job's lease (see job_lease.claim):
    duplicate, late or superseded deliveries are dropped here.
    """
    from app.services import job_lease

    logger.info("[%s] Celery worker starting pipeline", job_id)
    owner = job_lease.owner_id(self.request.id)
    outcome = job_lease.claim(job_id, dispatch_id, owner)
    if outcome != "run":
        return {"job_id": job_id, "status": outcome}
    job_lease.start(job_id, owner)
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
    finally:
        job_lease.release(job_id)
        ended = job_lease.ended_by(job_id)
        if ended != job_lease.STOP_SUPERSEDED:
            # A superseded run's recorded script may be the new run's.
            _kill_own_script(job_id)
        if ended not in (job_lease.STOP_SUPERSEDED, job_lease.STOP_HEARTBEAT_LOST):
            _fail_if_still_held(job_id, owner)


def _fail_if_still_held(job_id: str, owner: str) -> None:
    """Fail a job this run still holds after it ended, e.g. because its
    failure write raised. Left running, it would look like a lost worker and
    be retried."""
    from app.services import job_lease

    try:
        state = job_lease.job_store.job_lease_state(job_id)
        if not state or state["status"] != "running" or state["lease_owner"] != owner:
            return
        logger.error("[%s] Run ended without a result; failing the job", job_id)
        if job_lease.job_store.fail_job_if_lease(job_id, expected_owner=owner, error=job_lease.RUN_ENDED_ERROR):
            job_lease.record_failure(job_id, "pipeline_error", job_lease.RUN_ENDED_ERROR)
    except Exception:  # noqa: BLE001
        logger.warning("[%s] Could not check the job's final state", job_id, exc_info=True)


def _kill_own_script(job_id: str) -> None:
    """Kill this job's script group if it outlived the run (soft time limit, or
    an exception while a script ran). A no-op after a normal run."""
    from app.services import job_lease

    try:
        state = job_lease.job_store.job_lease_state(job_id)
        if state and state.get("active_host") == job_lease.HOST and str(state.get("active_pid") or "").isdigit():
            job_lease.kill_job_script_group(job_id, int(state["active_pid"]))
    except Exception:  # noqa: BLE001
        logger.warning("[%s] Could not check for a leftover script", job_id, exc_info=True)


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
    if not _claim_ingest_delivery(job_id):
        return {"job_id": job_id, "status": "abandoned"}
    try:
        run_ingest_job(job_id)
        # run_ingest_job catches its own pipeline failures internally and
        # calls _fail_job rather than raising -- so without this reload, a
        # failed job (e.g. 0/43 downloads) still fell through to this success
        # branch, logging "completed successfully" and returning
        # status: "completed" while the job row itself correctly said
        # "failed". Same fix run_video_job_task already has.
        from app.services.ingest_store import load_ingest_job
        job = load_ingest_job(job_id)
        status = job.get("status", "completed") if job else "completed"
        if status == "failed":
            logger.error("[%s] Ingest pipeline ended with status 'failed'", job_id)
        else:
            logger.info("[%s] Ingest pipeline completed successfully", job_id)
        return {"job_id": job_id, "status": status}
    except Exception as exc:
        logger.exception("[%s] Ingest pipeline raised unhandled exception", job_id)
        raise exc
