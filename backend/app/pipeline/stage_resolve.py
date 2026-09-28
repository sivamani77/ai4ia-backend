from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from app.pipeline.base import PipelineContext, StageResult
from app.services import intent_router, processing_records
from app.services.corpus_catalog import get_catalog
from app.services.job_store import append_event, load_job, save_job
from app.services.prompt_resolver import Decline, resolve
from app.services.video_job_runner import update_stage

logger = logging.getLogger("ai4ia.pipeline.resolve")

STAGE_NAME = "prompt_resolve"


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _route(job_id: str, prompt: str, catalog, context: PipelineContext) -> intent_router.Route:
    """The job's route. A rerun after a lost worker reuses the first run's, so it
    reads the prompt the same way; a storyline-cache hit needs none."""
    if context.storyline_cached:
        return intent_router.Route("cached")
    try:
        job = load_job(job_id)
    except Exception:
        job = {}
    saved = job.get("prompt_route") or {}
    if saved.get("prompt_hash") == _prompt_hash(prompt) and saved.get("router_version") == intent_router.router_version():
        return intent_router.Route.from_dict(saved.get("route") or {})
    route = intent_router.route(prompt, catalog, trace={
        "job_id": job_id, "user_id": job.get("user_id") or None, "stage": STAGE_NAME,
        "agent": "intent_router", "decision_label": "intent_routing"})
    if route.decline is None and job:
        try:
            job = load_job(job_id)  # fresh: the router call took seconds
            job["prompt_route"] = {"prompt_hash": _prompt_hash(prompt), "router_version": intent_router.router_version(),
                                   "route": route.to_dict()}
            save_job(job_id, job)
        except Exception:
            logger.warning("Job %s: could not save the route; a rerun will route again", job_id, exc_info=True)
    return route


def _declined(job_id: str, decline: Decline, slots: dict, dropped: list, timings: dict,
              resolution_method: str | None = None) -> StageResult:
    logger.info("Job %s declined: %s", job_id, decline.reason_code)
    processing_records.record_layer(
        job_id, outcome="declined", reason_code=decline.reason_code, user_message=decline.message,
        slots=slots, dropped_slots=dropped, suggestions=decline.suggestions, timings_ms=timings,
        resolution_method=resolution_method,
    )
    update_stage(job_id, STAGE_NAME, "failed", 100, decline.message)
    append_event(job_id, "stage_failed", STAGE_NAME, {"reason_code": decline.reason_code})
    return StageResult(stage=STAGE_NAME, status="failed", message=decline.message, data={
        "error_code": "declined", "reason_code": decline.reason_code,
        "user_message": decline.message, "suggestions": decline.suggestions,
    })


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    """Check the speaker, chapter and year a prompt names against the archive,
    then read what the prompt asks for (docs/design/intent-layer-design.md §6, §7.5).

    Declines here, before any retrieval or storyline call, when the archive
    clearly cannot answer.
    """
    update_stage(job_id, STAGE_NAME, "running", 50, "Checking the prompt against the archive")
    prompt = request.get("prompt") or ""
    started = time.monotonic()
    catalog = get_catalog()
    result = resolve(prompt, catalog)
    timings = {"l2a": int((time.monotonic() - started) * 1000)}

    if result.decline:
        # A clear rules decline stands; the router is not asked.
        return _declined(job_id, result.decline, result.slots, result.dropped_slots, timings)

    route = _route(job_id, prompt, catalog, context)
    if route.latency_ms is not None:
        timings["router"] = route.latency_ms
    dropped = result.dropped_slots + route.dropped
    slots = dict(result.slots)
    if route.used:
        slots["route"] = {k: v for k, v in route.to_dict().items()
                          if k in ("intent", "speakers", "topic_terms", "year", "chapter", "theme", "source_video",
                                   "exclude_terms", "excluded_speakers", "language", "aggregate", "count_unit",
                                   "unsupported") and v not in (None, [], "")}
    if route.decline:
        return _declined(job_id, route.decline, slots, dropped, timings, resolution_method=route.method)

    context.resolved_speaker = result.speaker
    context.route = route.to_dict() if route.used else {"method": route.method}
    processing_records.record_layer(job_id, slots=slots, dropped_slots=dropped, timings_ms=timings,
                                    resolution_method=route.method)
    update_stage(job_id, STAGE_NAME, "completed", 100, "Prompt matches the archive")
    append_event(job_id, "stage_completed", STAGE_NAME, {"slots": slots, "resolution_method": route.method})
    return StageResult(stage=STAGE_NAME, status="completed", message="Prompt matches the archive",
                       data={"slots": slots, "resolution_method": route.method})
