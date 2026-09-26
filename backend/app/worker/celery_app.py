"""
Celery application instance.

Import this module anywhere you need to enqueue or inspect tasks.
The broker and result backend are both Redis — configure via environment:

    REDIS_URL=redis://localhost:6379/0   (default)
"""
from __future__ import annotations

from celery import Celery

from app.core.otel import is_enabled as _otel_enabled, setup_tracing
from app.core.settings import settings

REDIS_URL = settings.redis_url

setup_tracing("ai4ia-worker")

if _otel_enabled():
    from opentelemetry.instrumentation.celery import CeleryInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    CeleryInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    # See app.main for why skip_dep_check is needed here (psycopg2-binary
    # vs. psycopg2 distribution-name mismatch).
    Psycopg2Instrumentor().instrument(skip_dep_check=True)

celery_app = Celery(
    "ai4ia",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

# Separate queue for ingest jobs so they don't block video generation workers
INGEST_QUEUE = "ingest_jobs"


def _crontab(spec: str):
    """Parse "minute hour dom month dow" into a crontab, or None if malformed."""
    from celery.schedules import crontab

    parts = spec.split()
    if len(parts) != 5:
        return None

    minute, hour, dom, month, dow = parts
    return crontab(
        minute=minute,
        hour=hour,
        day_of_month=dom,
        month_of_year=month,
        day_of_week=dow,
    )


def _configure_beat_schedule() -> None:
    """Register the periodic tasks whose settings enable them.

    Auto-sync is opt-in (ENABLE_AUTO_SYNC); the auth retention sweep is on by
    default, because leaving it off means session/token rows accumulate IP
    addresses indefinitely. Blank AUTH_PURGE_CRON to disable it.
    """
    schedule = {}

    if settings.enable_auto_sync:
        sync_cron = _crontab(settings.auto_sync_cron)
        if sync_cron is not None:
            schedule["auto-sync-youtube-channel"] = {
                "task": "ai4ia.auto_sync_channel",
                "schedule": sync_cron,
                "options": {"queue": INGEST_QUEUE},
            }

    if settings.auth_purge_cron.strip():
        purge_cron = _crontab(settings.auth_purge_cron)
        if purge_cron is not None:
            schedule["purge-expired-auth-data"] = {
                "task": "ai4ia.purge_expired_auth_data",
                "schedule": purge_cron,
                "options": {"queue": INGEST_QUEUE},
            }

    if settings.processing_purge_cron.strip():
        processing_cron = _crontab(settings.processing_purge_cron)
        if processing_cron is not None:
            schedule["purge-expired-processing-records"] = {
                "task": "ai4ia.purge_expired_processing_records",
                "schedule": processing_cron,
                "options": {"queue": INGEST_QUEUE},
            }

    if schedule:
        celery_app.conf.beat_schedule = schedule


@celery_app.task(name="ai4ia.purge_expired_processing_records", queue=INGEST_QUEUE)
def purge_expired_processing_records_task() -> dict:
    """Delete prompt and LLM-exchange records past their expires_at (privacy policy 6)."""
    import logging

    from app.services import processing_records

    results = processing_records.purge_expired()
    logging.getLogger("ai4ia.worker.processing_purge").info("Processing-record sweep: %s", results)
    return results


@celery_app.task(name="ai4ia.purge_expired_auth_data", queue=INGEST_QUEUE)
def purge_expired_auth_data_task() -> dict:
    """Delete auth rows past the retention window (AUTH_DATA_RETENTION_DAYS).

    user_sessions, refresh_tokens and auth_audit all store the client IP and
    user-agent, and every end-of-life path in the auth flow is an UPDATE
    (ended_at / revoked_at) rather than a DELETE — so without this task those
    rows, and the IPs in them, live as long as the database does.

    Only finished rows are eligible: an active session or a live refresh
    token is never purged, however old. Each table is swept independently so
    one failure does not abandon the rest.
    """
    import logging

    log = logging.getLogger("ai4ia.worker.auth_purge")

    from app.db.repositories import (
        get_auth_audit_repository,
        get_refresh_token_repository,
        get_session_repository,
    )

    days = settings.auth_data_retention_days
    results: dict[str, int | str] = {"retention_days": days}

    sweeps = (
        ("sessions", lambda: get_session_repository().purge_ended_before(retention_days=days)),
        ("refresh_tokens", lambda: get_refresh_token_repository().purge_spent_before(retention_days=days)),
        ("auth_audit", lambda: get_auth_audit_repository().purge_before(retention_days=days)),
    )

    for name, sweep in sweeps:
        try:
            results[name] = sweep()
        except Exception:
            log.exception("Auth retention sweep failed for %s", name)
            results[name] = "error"

    log.info("Auth retention sweep (%d-day window): %s", days, results)
    return results


@celery_app.task(name="ai4ia.auto_sync_channel", queue=INGEST_QUEUE)
def auto_sync_channel_task() -> dict:
    """Celery beat task: create and run a full channel ingest job."""
    import logging
    log = logging.getLogger("ai4ia.worker.auto_sync")
    log.info("Auto-sync triggered by Celery beat")

    from app.services.ingest_job_runner import create_ingest_job, run_ingest_job
    from app.services.ingest_store import list_ingest_jobs

    # Idempotency: skip if a sync is already running
    jobs = list_ingest_jobs(limit=5)
    for j in jobs:
        if j.get("status") in ("queued", "running"):
            log.info("Skipping auto-sync — job %s is already active", j.get("job_id"))
            return {"skipped": True, "active_job": j.get("job_id")}

    job_id = create_ingest_job()
    run_ingest_job(job_id)
    return {"job_id": job_id}


_configure_beat_schedule()

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Results TTL — keep task results for 24 h then discard
    result_expires=86400,

    # At most 1 video pipeline per worker process (heavy CPU/GPU work)
    worker_concurrency=1,

    # Acknowledge the task only after it finishes so a worker crash re-queues it
    task_acks_late=True,

    # NACK the task back to the queue if the worker process dies mid-execution
    task_reject_on_worker_lost=True,

    # Do not prefetch more than one task at a time
    worker_prefetch_multiplier=1,

    # How long an unacknowledged task waits before the broker hands it to
    # another worker. Must exceed the video task's time limit; the same Redis
    # database shares one window across every environment's workers.
    broker_transport_options={"visibility_timeout": settings.celery_visibility_timeout_seconds},

    # On a broker blip, keep running tasks going; a redelivered copy loses
    # the job lease and is dropped (app.services.job_lease).
    worker_cancel_long_running_tasks_on_connection_loss=False,

    # Route all tasks to a dedicated queue so we can scale workers independently
    # later, and so environments sharing one Redis do not consume each other's
    # jobs. See settings.worker_queue.
    task_default_queue=settings.worker_queue,
)
