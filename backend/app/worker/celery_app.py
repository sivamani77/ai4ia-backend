"""
Celery application instance.

Import this module anywhere you need to enqueue or inspect tasks.
The broker and result backend are both Redis — configure via environment:

    REDIS_URL=redis://localhost:6379/0   (default)
"""
from __future__ import annotations

from celery import Celery

from app.core.settings import settings

REDIS_URL = settings.redis_url

celery_app = Celery(
    "ai4ia",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

# Separate queue for ingest jobs so they don't block video generation workers
INGEST_QUEUE = "ingest_jobs"


def _configure_beat_schedule() -> None:
    """Register the periodic channel sync task only when ENABLE_AUTO_SYNC=true."""
    if not settings.enable_auto_sync:
        return

    from celery.schedules import crontab

    # Parse "minute hour dom month dow" from settings (e.g. "0 2 * * *")
    parts = settings.auto_sync_cron.split()
    if len(parts) != 5:
        return

    minute, hour, dom, month, dow = parts
    celery_app.conf.beat_schedule = {
        "auto-sync-youtube-channel": {
            "task": "ai4ia.auto_sync_channel",
            "schedule": crontab(
                minute=minute,
                hour=hour,
                day_of_month=dom,
                month_of_year=month,
                day_of_week=dow,
            ),
            "options": {"queue": INGEST_QUEUE},
        },
    }


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

    # Route all tasks to a dedicated queue so we can scale workers independently later
    task_default_queue="video_jobs",
)
