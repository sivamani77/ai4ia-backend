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
    try:
        run_video_job(job_id, request)
        logger.info("[%s] Pipeline completed successfully", job_id)
        return {"job_id": job_id, "status": "completed"}
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
