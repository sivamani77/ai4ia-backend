"""
Video job orchestration service.

Thin orchestrator: creates a PipelineContext and delegates each stage to
app.pipeline.stage_* modules. All stage logic lives there.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai4ia.services.video_job_runner")

from app.core import metrics as _metrics
from app.core.code_version import code_version
from app.core.config import VLLM_MODEL
from app.core.error_messages import friendly_message
from app.core.otel import get_tracer
from app.core.security import parse_timestamp
from app.core.settings import settings
from app.pipeline.base import PipelineContext

_tracer = get_tracer("ai4ia.pipeline")
from app.services import model_gate, processing_records
from app.services import job_lease
from app.services.job_store import (
    save_job,
    load_job,
    append_event,
    load_events,
    list_jobs,
    finish_job,
)

# Process-local stage start times for job_stage_duration_seconds — not
# persisted on the job (would leak internal bookkeeping into the
# user-facing JobResponse); mirrors the process-local _redis_cache pattern
# below.
_stage_start_times: dict[tuple[str, str], float] = {}

# ---------------------------------------------------------------------------
# Stage weights (must sum to 100)
# ---------------------------------------------------------------------------
STAGE_WEIGHTS: dict[str, int] = {
    "prompt_guardrail": 1,
    "prompt_resolve": 1,
    "input_validation": 1,
    "07_storyline": 15,
    "07b_validate_storyline": 34,
    "08_assets": 25,
    "09_export": 15,
    "10_validate_export": 5,
    "11_subtitles": 3,
}

STAGE_DEFINITIONS = [
    {"stage": "prompt_guardrail",        "label": "Prompt content check"},
    {"stage": "prompt_resolve",          "label": "Prompt check against the archive"},
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

    _metrics.job_stage_transitions_total.labels(stage=stage, status=status).inc()
    stage_key = (job_id, stage)
    if status == "running":
        _stage_start_times.setdefault(stage_key, time.monotonic())
    elif status in ("completed", "failed"):
        started = _stage_start_times.pop(stage_key, None)
        if started is not None:
            _metrics.job_stage_duration_seconds.labels(stage=stage).observe(time.monotonic() - started)

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


def _fail_job(
    job_id: str, stage: str, error_type: str, message: str, details: str = "",
    exc: Exception | None = None, *, user_message: str | None = None,
    reason_code: str | None = None, suggestions: list[str] | None = None,
) -> None:
    owner = job_lease.release(job_id)
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
        # user_message is what the client should display. message/details are
        # technical (exception text, tracebacks, ffmpeg output) for logs and
        # support — never render them directly to a user.
        "user_message": user_message or friendly_message(error_type, exc),
    }
    if reason_code:
        job["error"]["reason_code"] = reason_code
        job["error"]["suggestions"] = suggestions or []
    save_job(job_id, job)
    if not finish_job(job_id, "failed", owner):
        logger.warning("[%s] Not recording failure: the job is no longer this run's", job_id)
        return
    append_event(job_id, "failed", stage, {
        "error_type": error_type,
        "message": message,
    })
    _metrics.job_completions_total.labels(status="failed").inc()
    processing_records.finish(job_id, "failed")


def _cancel_job(job_id: str, message: str = "", *, kill_pid: bool = True) -> None:
    job_lease.release(job_id)
    job = load_job(job_id)
    job["status"] = "cancelled"
    job["completed_at"] = _utc_now()
    job["message"] = message or "Job cancelled"
    job["updated_at"] = _utc_now()
    save_job(job_id, job)
    append_event(job_id, "cancelled", job.get("current_stage", "unknown"), {"message": message})
    _metrics.job_completions_total.labels(status="cancelled").inc()
    processing_records.finish(job_id, "cancelled")

    # PID kill is only valid in the WORKER process (where the subprocess and its
    # active_pid actually live). The API's DELETE handler runs in a different
    # container/namespace, so a PID from the worker is meaningless — and could
    # even hit a recycled PID. The API therefore passes kill_pid=False and relies
    # on the worker's in-stage cancellation poll (run_script) to do the kill.
    if kill_pid:
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


def _complete_job(job_id: str) -> bool:
    """Mark the job completed; False if the job is no longer this run's."""
    # Stop the heartbeat first, so a finished job never raises a stop signal
    # that would cut the post-completion housekeeping short.
    owner = job_lease.release(job_id)
    job = load_job(job_id)
    job["status"] = "completed"
    job["current_stage"] = "completed"
    job["progress_percent"] = 100
    job["message"] = "Video generation completed"
    job["completed_at"] = _utc_now()
    job["updated_at"] = _utc_now()
    save_job(job_id, job)
    if not finish_job(job_id, "completed", owner):
        logger.warning("[%s] Not marking completed: the job is no longer this run's", job_id)
        return False
    _metrics.job_completions_total.labels(status="completed").inc()
    processing_records.finish(job_id, "completed")
    created_at = job.get("created_at")
    if created_at:
        try:
            duration = (parse_timestamp(job["completed_at"]) - parse_timestamp(created_at)).total_seconds()
            _metrics.job_duration_seconds.observe(duration)
        except (ValueError, TypeError):
            logger.warning("Could not compute job_duration_seconds for job_id=%s", job_id)
    return True


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

class JobQuotaExceeded(Exception):
    """A submission was refused by the job quota; `limit` names which one."""

    def __init__(self, limit: str) -> None:
        super().__init__(limit)
        self.limit = limit


def create_video_job(
    request: dict[str, Any],
    user_id: str = "",
    flags: dict[str, Any] | None = None,
    *,
    enforce_quota: bool = False,
) -> str:
    """Create a queued job. With enforce_quota, the quota check and the insert
    are one atomic step and JobQuotaExceeded is raised when a limit is hit."""
    job_id = f"vjob_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "prompt": request.get("prompt", ""),
        "target_duration": request["target_duration"],
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
        "api_code_version": code_version(),
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
    if enforce_quota:
        from app.services import job_store

        # The dispatch id marks the one queue message allowed to run this job
        # (docs/design/job-recovery-plan.md); the queue name marks the environment.
        job["dispatch_id"] = uuid.uuid4().hex
        limit = job_store.create_job_within_quota(
            job_id, job,
            user_id=user_id,
            max_running=settings.video_job_max_running_per_user,
            max_queued=settings.video_job_max_queued_per_user,
            max_active=settings.video_job_max_active,
            dispatch_id=job["dispatch_id"],
            worker_queue=settings.worker_queue,
        )
        if limit is not None:
            raise JobQuotaExceeded(limit)
    else:
        save_job(job_id, job)
    try:
        append_event(job_id, "created", "queued", {"prompt": request.get("prompt", "")})
        processing_records.start_request(job_id, user_id, request.get("prompt", ""), flags=flags)
    except Exception as exc:
        # The row exists but will never be dispatched; fail it so it does not
        # hold one of the owner's queued slots.
        fail_undispatched_job(job_id, f"Job setup failed: {exc}")
        raise
    _metrics.jobs_created_total.inc()
    return job_id


def fail_undispatched_job(job_id: str, message: str) -> None:
    """Fail a job that was created but never reached the worker. Best effort:
    a failure here is logged, never raised over the original error."""
    try:
        _fail_job(
            job_id, "queued", "dispatch_failed", message,
            user_message="The job could not be queued. Please try again.",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Could not mark undispatched job %s failed", job_id)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _apply_cached_storyline(
    job_id: str,
    ctx: "PipelineContext",
    cached_storyline: dict[str, Any],
) -> None:
    """Materialize a storyline-cache hit as if 07_storyline + 07b_validate_storyline
    had just run: write a real validated-storyline file (08_assets reads it from
    disk, same as a normal run), point the context at it, and mark both stages
    completed so progress/events stay coherent for anyone polling job status."""
    from app.core.paths import STORYLINES_DIR
    from app.pipeline.s3_helpers import use_s3, upload

    STORYLINES_DIR.mkdir(parents=True, exist_ok=True)
    validated_path = STORYLINES_DIR / f"storyline_cache_{job_id}_validated.json"
    validated_path.write_text(json.dumps(cached_storyline), encoding="utf-8")

    s3_key = f"raw/storylines/{validated_path.name}"
    if use_s3():
        upload(validated_path, s3_key)

    ctx.storyline_path = str(validated_path)
    ctx.validated_storyline_path = str(validated_path)
    ctx.validated_storyline_data = cached_storyline

    job = load_job(job_id)
    job["paths"]["storyline_path"] = str(validated_path)
    job["paths"]["validated_storyline_path"] = str(validated_path)
    if use_s3():
        # Mirrors stage_07_storyline.py / stage_07b_validate.py: without this,
        # a mindmap/decision-record lookup on a different pod (or after this
        # worker restarts) can't resolve the file via S3 key and falls back to
        # a full bucket listing + substring match on job_id — fragile, and the
        # cause of "mind map not displayed" for cache-served jobs.
        job["paths"]["storyline_s3_key"] = s3_key
        job["paths"]["validated_storyline_s3_key"] = s3_key
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    for stage_name in ("07_storyline", "07b_validate_storyline"):
        update_stage(job_id, stage_name, "completed", 100, "Served from storyline cache")
        append_event(job_id, "stage_completed", stage_name, {"cache_hit": True})


def _maybe_cache_storyline_result(job_id: str, request: dict[str, Any], ctx: "PipelineContext") -> None:
    """After a freshly-run (non-cached) 07b_validate_storyline, write the
    result to the storyline cache — but only if it passed its quality gates.
    No point serving a known-weak result fast to the next identical query.
    Best-effort: any failure here is logged, never fails the pipeline."""
    from app.services import storyline_cache as _storyline_cache

    try:
        # Read from the in-memory copy, not ctx.validated_storyline_path —
        # in S3 mode that file is already deleted by the time this runs
        # (stage_07b_validate.py's cleanup_tmp_dir happens before it returns).
        vdata = ctx.validated_storyline_data
        if not vdata:
            return
        quality_ok = bool((vdata.get("editorial_quality_gates") or {}).get("final_timeline_approved", False))
        _storyline_cache.store_storyline(request, vdata, quality_ok=quality_ok)
    except Exception:
        logger.warning("storyline cache: failed to store result for job %s (non-fatal)", job_id, exc_info=True)


def run_video_job(job_id: str, request: dict[str, Any]) -> None:
    """Run the full pipeline via stage modules."""
    from app.pipeline import (
        stage_guardrail,
        stage_input_validation,
        stage_resolve,
        stage_07_storyline,
        stage_07b_validate,
        stage_08_assets,
        stage_09_export,
        stage_10_validate,
        stage_11_subtitles,
    )
    from app.services import storyline_cache as _storyline_cache

    log = _job_logger(job_id)

    # Terminal-state guard. Celery uses acks_late + reject_on_worker_lost, so a
    # task in flight during a worker restart is REDELIVERED and this function
    # runs again. If the job was already cancelled (or finished) in the meantime,
    # re-running would restart the whole render — the "cancelled job keeps
    # processing" loop. Bail out immediately so redelivery is a harmless no-op.
    try:
        existing = load_job(job_id)
    except Exception:
        existing = None
    if existing and existing.get("status") in {"cancelled", "completed", "failed"}:
        log.info("Skipping run for %s — already terminal (status=%s)",
                 job_id, existing.get("status"))
        return

    try:
        job = load_job(job_id)
        job["status"] = "running"
        job["started_at"] = _utc_now()
        job["updated_at"] = _utc_now()
        # The worker's version, which can differ from the API that queued it.
        job["code_version"] = code_version()
        save_job(job_id, job)
        log.info("Pipeline started (code %s)", job["code_version"])

        ctx = PipelineContext(job_id=job_id, request=request)

        # Gate: nothing below this point may touch Ollama/vLLM until both are
        # confirmed reachable. Unlike the interactive-endpoint gate (api/deps.py),
        # this is a queued background job with nobody holding an HTTP connection
        # open, so a long blocking wait here is the right trade — it's visible
        # to the user as a job event instead of failing a request outright.
        update_stage(job_id, "model_warmup", "running", 0,
                     "Waiting for AI model hosts to come online")
        readiness = model_gate.ensure_ready(
            model_gate.EMBEDDING, model_gate.LLM,
            wait_seconds=settings.model_warm_deadline_seconds,
        )
        not_ready = [key for key, ok in readiness.items() if not ok]
        if not_ready:
            states = {key: model_gate.state(key) for key in not_ready}
            log.error("Model host(s) never became ready: %s", states)
            _fail_job(
                job_id, "model_warmup", "model_unavailable",
                f"AI model host(s) did not come online in time: {', '.join(not_ready)}",
                details=str(states),
            )
            _cleanup_job_tmp_dirs(job_id)
            return
        update_stage(job_id, "model_warmup", "completed", 100, "AI model hosts ready")

        stages = [
            stage_guardrail,
            stage_resolve,
            stage_input_validation,
            stage_07_storyline,
            stage_07b_validate,
            stage_08_assets,
            stage_09_export,
            stage_10_validate,
            stage_11_subtitles,
        ]

        from celery.exceptions import SoftTimeLimitExceeded

        from app.pipeline.base import JobCancelledException, JobDeadlineExceeded, JobSupersededException

        if not request.get("force_refresh"):
            cached_storyline = _storyline_cache.get_cached_storyline(request)
            if cached_storyline is not None:
                _apply_cached_storyline(job_id, ctx, cached_storyline)
                stages = [stage_guardrail, stage_resolve, stage_08_assets, stage_09_export,
                          stage_10_validate, stage_11_subtitles]

        for stage_mod in stages:
            stage_name = getattr(stage_mod, "STAGE_NAME", stage_mod.__name__.split(".")[-1])
            stage_log = _job_logger(job_id, stage_name)
            stage_log.info("Stage starting")
            try:
                job_lease.raise_if_stopped(job_id, stage_name)
                with _tracer.start_as_current_span(stage_name, attributes={"job_id": job_id}):
                    result = stage_mod.run(job_id, request, ctx)
            except JobSupersededException as exc:
                # The job belongs to another run now (or will be failed by the
                # sweeper); record nothing.
                stage_log.warning("Run stopped during %s: %s", stage_name, exc.reason)
                job_lease.release(job_id)
                _cleanup_job_tmp_dirs(job_id)
                return
            except (JobDeadlineExceeded, SoftTimeLimitExceeded):
                stage_log.error("Job passed its %ss limit during %s", settings.job_max_runtime_seconds, stage_name)
                _fail_job(
                    job_id, stage_name, "deadline_exceeded",
                    f"Job exceeded {settings.job_max_runtime_seconds}s during {stage_name}",
                    user_message="Your video took too long to generate and was stopped. "
                                 "Please try again, or choose a shorter duration.",
                )
                _cleanup_job_tmp_dirs(job_id)
                return
            except JobCancelledException:
                # Cancelled mid-stage: the subprocess was already killed inside
                # run_script. Finalize as cancelled (not failed) and stop.
                stage_log.info("Cancellation detected during stage %s", stage_name)
                _cancel_job(job_id, "cancelled during stage")
                _cleanup_job_tmp_dirs(job_id)
                return
            if result.status == "failed":
                stage_log.error("Stage failed: %s", result.message)
                data = result.data or {}
                _fail_job(job_id, result.stage, data.get("error_code") or "pipeline_error", result.message,
                          user_message=data.get("user_message"), reason_code=data.get("reason_code"),
                          suggestions=data.get("suggestions"))
                _cleanup_job_tmp_dirs(job_id)
                return
            stage_log.info("Stage %s", result.status)

            # Bug: `stage_name` here is derived from the module's __name__
            # (e.g. "stage_07b_validate"), never "07b_validate_storyline" —
            # no stage module sets STAGE_NAME, so a string match against
            # stage_name never fires. result.stage is the actual semantic
            # name each stage module returns (e.g. StageResult(stage=
            # "07b_validate_storyline", ...) in stage_07b_validate.py) —
            # compare against that instead.
            if result.stage == "07b_validate_storyline":
                _maybe_cache_storyline_result(job_id, request, ctx)

            if _is_cancelling(job_id):
                log.info("Cancellation detected after stage %s", stage_name)
                _cancel_job(job_id, "cancelled between stages")
                _cleanup_job_tmp_dirs(job_id)
                return

        if not _complete_job(job_id):
            return
        log.info("Pipeline completed")

        # ------------------------------------------------------------------
        # Post-completion housekeeping (decision record, media registry, S3
        # upload, completion event). The job is ALREADY marked completed and
        # that is durable — so NOTHING in this block may re-raise into the
        # outer handler and flip a finished job to "failed". A transient
        # Supabase disconnect here (the HTTP/2 keepalive dropped during the
        # long render) previously did exactly that AND skipped the S3 upload.
        # Each step is isolated; failures are logged, not fatal.
        # ------------------------------------------------------------------
        try:
            _emit_job_decision_record(job_id, outcome="video_generated")
        except Exception:
            log.warning("decision record failed for job %s (non-fatal)", job_id, exc_info=True)

        # Reconnect after the long run; retry-enabled load_job rides through a
        # dropped keepalive connection instead of raising.
        job = None
        try:
            job = load_job(job_id)
        except Exception:
            log.warning("post-complete load_job failed for job %s (non-fatal)", job_id, exc_info=True)

        try:
            from app.services.media_registry import build_media_registry
            build_media_registry(job_id)
        except Exception:
            log.warning("media registry build failed for job %s (non-fatal)", job_id, exc_info=True)

        try:
            from app.services.storage_service import STORAGE_BACKEND, upload_job_outputs_to_s3
            if STORAGE_BACKEND == "s3":
                upload_job_outputs_to_s3(job_id)
                from app.pipeline.s3_helpers import cleanup_tmp_dir
                for stage_prefix in (f"07b_{job_id}", f"08_{job_id}", f"09_{job_id}", f"10_{job_id}"):
                    cleanup_tmp_dir(stage_prefix)
        except Exception:
            log.warning("S3 upload failed for job %s — outputs remain on local disk", job_id, exc_info=True)

        try:
            vp = (job or {}).get("paths", {}).get("video_path", "") if job else ""
            rel = Path(vp).name if vp else ""
            append_event(job_id, "completed", "completed", {
                "video_stream_url": f"/api/video-jobs/{job_id}/video/stream",
                "video_url": f"/exports/{rel}" if rel else None,
                "duration": None,
            })
        except Exception:
            log.warning("completion event emit failed for job %s (non-fatal)", job_id, exc_info=True)

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
        _fail_job(job_id, stage, "pipeline_error", str(exc), tb, exc=exc)
        _cleanup_job_tmp_dirs(job_id)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _emit_job_decision_record(job_id: str, outcome: str) -> None:
    """Fire-and-forget Layer 1 decision record for a completed video job."""
    import threading as _threading

    def _insert() -> None:
        try:
            from app.db.repositories.decision_record_repository import DecisionRecordRepository
            from datetime import timedelta
            from app.pipeline.s3_helpers import use_s3 as _use_s3

            job = load_job(job_id)
            paths = job.get("paths", {})

            def _read_storyline(path_key: str, s3_key_key: str) -> dict:
                """Read storyline JSON from S3 (preferred in S3 mode) or local path."""
                s3_key = paths.get(s3_key_key)
                local = paths.get(path_key)
                # S3 first — local /tmp paths for validated storyline get cleaned up after stage 07b
                if _use_s3() and s3_key:
                    try:
                        from app.pipeline.s3_helpers import _s3, _bucket
                        resp = _s3().get_object(Bucket=_bucket(), Key=s3_key)
                        return json.loads(resp["Body"].read().decode("utf-8"))
                    except Exception:
                        pass
                if local and Path(local).exists():
                    try:
                        return json.loads(Path(local).read_text(encoding="utf-8"))
                    except Exception:
                        pass
                return {}

            # Validated storyline has editorial gates; fall back to initial if missing
            storyline_data = _read_storyline(
                "validated_storyline_path", "validated_storyline_s3_key"
            ) or _read_storyline("storyline_path", "storyline_s3_key")

            models = storyline_data.get("models") or {}
            planning = storyline_data.get("planning_summary") or {}
            story = storyline_data.get("storyline") or {}
            director = story.get("director_plan") or {}
            bucket = storyline_data.get("candidate_bucket_summary") or {}
            gates = storyline_data.get("editorial_quality_gates") or {}
            # Script writes VLLM_MODEL under "strong_llm_model" key
            llm_model = models.get("strong_llm_model") or VLLM_MODEL

            started_at = job.get("started_at")
            completed_at = job.get("completed_at")
            duration_ms: int | None = None
            if started_at and completed_at:
                try:
                    from datetime import datetime as _dt
                    t0 = _dt.fromisoformat(started_at.replace("Z", "+00:00"))
                    t1 = _dt.fromisoformat(completed_at.replace("Z", "+00:00"))
                    duration_ms = int((t1 - t0).total_seconds() * 1000)
                except Exception:
                    pass

            expires_at = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()

            DecisionRecordRepository().create({
                "id": f"dr_{job_id}",
                "job_id": job_id,
                "user_id": job.get("user_id"),
                "schema_version": "osi-1.0",
                "subject": "documentary_generation",
                "user_prompt": job.get("prompt"),
                "accountability": "ai4ia-pipeline-v2",
                "pipeline_version": job.get("code_version"),
                "agents_executed": [
                    "intent", "retrieval", "classification",
                    "director", "synthesis", "storyline", "validation",
                ],
                "decisions": {
                    "intent_type": (storyline_data.get("prompt_profile") or {}).get("intent_type"),
                    "video_mode": planning.get("video_mode") or director.get("video_mode"),
                    "evidence_strength": planning.get("evidence_strength") or director.get("evidence_strength"),
                    "section_count": len(story.get("sections") or []),
                    "fragments_selected": bucket.get("selected", 0),
                    "fragments_rejected": bucket.get("rejected", 0),
                    "validation_passed": bool(gates.get("final_timeline_approved")),
                },
                "models_used": {
                    "embedding": models.get("embedding_model", "nomic-embed-text:v1.5"),
                    "director": llm_model,
                    "synthesis": llm_model,
                    "storyline": llm_model,
                    "validation": llm_model,
                },
                "outcome": outcome,
                "total_duration_ms": duration_ms,
                "expires_at": expires_at,
                "created_at": completed_at or _utc_now(),
            })
        except Exception:
            pass

    _threading.Thread(target=_insert, daemon=True).start()


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
        from urllib.parse import urlsplit

        parts = urlsplit(redis_url)
        host = parts.hostname or "localhost"
        port = parts.port or 6379
        with socket.create_connection((host, port), timeout=0.5):
            result = True
    except Exception:
        result = False

    _redis_cache = (result, time.monotonic())
    _metrics.redis_reachable.set(1 if result else 0)
    return result


def start_video_job_background(job_id: str, request: dict[str, Any], dispatch_id: str | None = None) -> None:
    """Dispatch to Celery. Raises HTTP 503 if Redis is unreachable — no thread fallback.
    The message carries the job's dispatch id; only a matching message may run it."""
    from fastapi import HTTPException

    if not (settings.redis_url and _redis_reachable()):
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable. Please try again in a moment.",
        )
    try:
        from app.worker.tasks import run_video_job_task
        if dispatch_id is None:
            dispatch_id = load_job(job_id).get("dispatch_id")
        kwargs = {"dispatch_id": dispatch_id} if dispatch_id else {}
        run_video_job_task.apply_async(args=[job_id, request], kwargs=kwargs)
    except Exception as exc:
        logger.error("Celery dispatch failed for job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to queue job. Please try again.",
        ) from exc
