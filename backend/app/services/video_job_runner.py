"""
Video job orchestration service.

Thin orchestrator: creates a PipelineContext and delegates each stage to
app.pipeline.stage_* modules. All stage logic lives there.
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.pipeline.base import PipelineContext
from app.services.job_store import (
    save_job,
    load_job,
    append_event,
    load_events,
    list_jobs,
)

# ---------------------------------------------------------------------------
# Stage weights (must sum to 100)
# ---------------------------------------------------------------------------
STAGE_WEIGHTS: dict[str, int] = {
    "input_validation": 2,
    "07_storyline": 15,
    "07b_validate_storyline": 35,
    "08_assets": 25,
    "09_export": 15,
    "10_validate_export": 5,
    "11_subtitles": 3,
}

STAGE_DEFINITIONS = [
    {"stage": "input_validation",        "label": "Input validation"},
    {"stage": "07_storyline",            "label": "Initial storyline generation"},
    {"stage": "07b_validate_storyline",  "label": "LLM editorial validation"},
    {"stage": "08_assets",               "label": "Voiceover and visual asset generation"},
    {"stage": "09_export",               "label": "Final video export"},
    {"stage": "10_validate_export",      "label": "Export validation"},
    {"stage": "11_subtitles",            "label": "Subtitle finalization"},
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _job_logger(job_id: str, stage: str = "") -> logging.LoggerAdapter:
    return logging.LoggerAdapter(
        logging.getLogger("ai4ia.pipeline"),
        {"job_id": job_id, "stage": stage},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Progress / persistence
# ---------------------------------------------------------------------------

def compute_global_progress(stage_status: dict[str, Any]) -> int:
    total = 0
    for stage, weight in STAGE_WEIGHTS.items():
        s = stage_status.get(stage, {})
        st = s.get("status", "pending")
        sp = int(s.get("progress_percent", 0))
        if st == "completed":
            total += weight
        elif st == "running":
            total += weight * sp // 100
    return min(100, total)


def update_stage(
    job_id: str,
    stage: str,
    status: str,
    progress: int,
    message: str = "",
    **extra: Any,
) -> None:
    job = load_job(job_id)
    job.setdefault("stage_status", {})
    job["stage_status"][stage] = {
        "status": status,
        "progress_percent": progress,
        "summary": message,
        **extra,
    }
    job["current_stage"] = stage
    job["stage_progress_percent"] = progress
    job["message"] = message
    job["progress_percent"] = compute_global_progress(job["stage_status"])
    job["updated_at"] = _utc_now()
    if status == "running" and not job.get("started_at"):
        job["started_at"] = _utc_now()
    save_job(job_id, job)
    append_event(job_id, "progress", stage, {
        "stage": stage,
        "status": status,
        "progress": progress,
        "global_progress": job["progress_percent"],
        "message": message,
    })


def _fail_job(job_id: str, stage: str, error_type: str, message: str, details: str = "") -> None:
    job = load_job(job_id)
    job["status"] = "failed"
    job["current_stage"] = stage
    job["updated_at"] = _utc_now()
    job["completed_at"] = _utc_now()
    job["error"] = {
        "type": error_type,
        "stage": stage,
        "message": message,
        "details": details[:2000],
    }
    save_job(job_id, job)
    append_event(job_id, "failed", stage, {
        "error_type": error_type,
        "message": message,
    })


def _cancel_job(job_id: str, message: str = "") -> None:
    job = load_job(job_id)
    job["status"] = "cancelled"
    job["completed_at"] = _utc_now()
    job["message"] = message or "Job cancelled"
    job["updated_at"] = _utc_now()
    save_job(job_id, job)
    append_event(job_id, "cancelled", job.get("current_stage", "unknown"), {"message": message})

    pid = job.get("active_pid")
    if pid:
        try:
            from app.pipeline.base import _kill_process_group
            _kill_process_group(pid)
        except Exception:
            pass


def _is_cancelling(job_id: str) -> bool:
    try:
        job = load_job(job_id)
        return job.get("status") == "cancelled"
    except Exception:
        return False


def _cleanup_job_tmp_dirs(job_id: str) -> None:
    """Delete all pipeline /tmp stage directories for a job (safe, ignore_errors)."""
    for prefix in (f"07b_{job_id}", f"08_{job_id}", f"09_{job_id}", f"10_{job_id}"):
        p = Path(f"/tmp/ai4ia_pipeline/{prefix}")
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def _complete_job(job_id: str) -> None:
    job = load_job(job_id)
    job["status"] = "completed"
    job["current_stage"] = "completed"
    job["progress_percent"] = 100
    job["message"] = "Video generation completed"
    job["completed_at"] = _utc_now()
    job["updated_at"] = _utc_now()
    save_job(job_id, job)


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

def create_video_job(request: dict[str, Any], user_id: str = "") -> str:
    job_id = f"vjob_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "prompt": request.get("prompt", ""),
        "target_duration": request.get("target_duration", 150),
        "user_id": user_id,
        "current_stage": "queued",
        "progress_percent": 0,
        "stage_progress_percent": 0,
        "message": "Job created",
        "created_at": _utc_now(),
        "started_at": None,
        "updated_at": _utc_now(),
        "completed_at": None,
        "request": request,
        "paths": {
            "storyline_path": None,
            "validated_storyline_path": None,
            "asset_manifest_path": None,
            "export_manifest_path": None,
            "timeline_metadata_path": None,
            "video_path": None,
            "srt_path": None,
            "vtt_path": None,
        },
        "stage_status": {},
        "warnings": [],
        "error": None,
    }
    save_job(job_id, job)
    append_event(job_id, "created", "queued", {"prompt": request.get("prompt", "")})
    return job_id


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_video_job(job_id: str, request: dict[str, Any]) -> None:
    """Run the full pipeline via stage modules."""
    from app.pipeline import (
        stage_input_validation,
        stage_07_storyline,
        stage_07b_validate,
        stage_08_assets,
        stage_09_export,
        stage_10_validate,
        stage_11_subtitles,
    )

    log = _job_logger(job_id)

    try:
        job = load_job(job_id)
        job["status"] = "running"
        job["started_at"] = _utc_now()
        job["updated_at"] = _utc_now()
        save_job(job_id, job)
        log.info("Pipeline started")

        ctx = PipelineContext(job_id=job_id, request=request)

        stages = [
            stage_input_validation,
            stage_07_storyline,
            stage_07b_validate,
            stage_08_assets,
            stage_09_export,
            stage_10_validate,
            stage_11_subtitles,
        ]

        for stage_mod in stages:
            stage_name = getattr(stage_mod, "STAGE_NAME", stage_mod.__name__.split(".")[-1])
            stage_log = _job_logger(job_id, stage_name)
            stage_log.info("Stage starting")
            result = stage_mod.run(job_id, request, ctx)
            if result.status == "failed":
                stage_log.error("Stage failed: %s", result.message)
                _fail_job(job_id, result.stage, "pipeline_error", result.message)
                _cleanup_job_tmp_dirs(job_id)
                return
            stage_log.info("Stage %s", result.status)
            if _is_cancelling(job_id):
                log.info("Cancellation detected after stage %s", stage_name)
                _cancel_job(job_id, "cancelled between stages")
                _cleanup_job_tmp_dirs(job_id)
                return

        _complete_job(job_id)
        log.info("Pipeline completed")

        try:
            from app.services.media_registry import build_media_registry
            build_media_registry(job_id)
        except Exception:
            pass  # non-fatal; job is still complete

        try:
            from app.services.storage_service import STORAGE_BACKEND, upload_job_outputs_to_s3
            if STORAGE_BACKEND == "s3":
                upload_job_outputs_to_s3(job_id)
                # Clean up all stage tmp dirs now that everything is in S3
                from app.pipeline.s3_helpers import cleanup_tmp_dir
                for stage_prefix in (f"07b_{job_id}", f"08_{job_id}", f"09_{job_id}", f"10_{job_id}"):
                    cleanup_tmp_dir(stage_prefix)
        except Exception:
            log.warning("S3 upload failed for job %s — outputs remain on local disk", job_id, exc_info=True)

        job = load_job(job_id)
        vp = job["paths"].get("video_path", "")
        rel = Path(vp).name if vp else ""
        append_event(job_id, "completed", "completed", {
            "video_stream_url": f"/api/video-jobs/{job_id}/video/stream",
            "video_url": f"/exports/{rel}" if rel else None,
            "duration": None,
        })

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        stage = "unknown"
        try:
            jb = load_job(job_id)
            stage = jb.get("current_stage", "unknown")
        except Exception:
            pass
        log.exception("Pipeline error at stage %s: %s", stage, exc)
        _fail_job(job_id, stage, "pipeline_error", str(exc), tb)
        _cleanup_job_tmp_dirs(job_id)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_redis_cache: tuple[bool, float] = (False, 0.0)
_REDIS_CACHE_TTL = 10.0  # seconds


def _redis_reachable() -> bool:
    """Return True only if the Redis broker is actually listening.

    Result is cached for 10 seconds to avoid a blocking TCP connect on
    every request (the check runs in the FastAPI request/response cycle).
    """
    import socket
    global _redis_cache
    result, ts = _redis_cache
    if time.monotonic() - ts < _REDIS_CACHE_TTL:
        return result

    redis_url = settings.redis_url
    try:
        parts = redis_url.replace("redis://", "").split("/")[0].split(":")
        host = parts[0] or "localhost"
        port = int(parts[1]) if len(parts) > 1 else 6379
        with socket.create_connection((host, port), timeout=0.5):
            result = True
    except Exception:
        result = False

    _redis_cache = (result, time.monotonic())
    return result


def start_video_job_background(job_id: str, request: dict[str, Any]) -> None:
    """Dispatch to Celery. Raises HTTP 503 if Redis is unreachable — no thread fallback."""
    from fastapi import HTTPException

    if not (settings.redis_url and _redis_reachable()):
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable. Please try again in a moment.",
        )
    try:
        from app.worker.tasks import run_video_job_task
        run_video_job_task.delay(job_id, request)
    except Exception as exc:
        logger.error("Celery dispatch failed for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to queue job. Please try again.",
        ) from exc
