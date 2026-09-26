from __future__ import annotations

import logging
import time
from typing import Any

from app.pipeline.base import PipelineContext, StageResult
from app.services import processing_records
from app.services.corpus_catalog import get_catalog
from app.services.job_store import append_event
from app.services.prompt_resolver import resolve
from app.services.video_job_runner import update_stage

logger = logging.getLogger("ai4ia.pipeline.resolve")

STAGE_NAME = "prompt_resolve"


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    """Check the speaker, chapter and year a prompt names against the archive.

    Declines here, before any retrieval or storyline call, when the archive
    clearly cannot answer (docs/design/intent-layer-design.md §6).
    """
    update_stage(job_id, STAGE_NAME, "running", 50, "Checking the prompt against the archive")
    started = time.monotonic()
    result = resolve(request.get("prompt") or "", get_catalog())
    timings = {"l2a": int((time.monotonic() - started) * 1000)}

    if result.decline:
        decline = result.decline
        logger.info("Job %s declined: %s", job_id, decline.reason_code)
        processing_records.record_layer(
            job_id, outcome="declined", reason_code=decline.reason_code, user_message=decline.message,
            slots=result.slots, dropped_slots=result.dropped_slots, suggestions=decline.suggestions,
            timings_ms=timings,
        )
        update_stage(job_id, STAGE_NAME, "failed", 100, decline.message)
        append_event(job_id, "stage_failed", STAGE_NAME, {"reason_code": decline.reason_code})
        return StageResult(stage=STAGE_NAME, status="failed", message=decline.message, data={
            "error_code": "declined", "reason_code": decline.reason_code,
            "user_message": decline.message, "suggestions": decline.suggestions,
        })

    context.resolved_speaker = result.speaker
    processing_records.record_layer(job_id, slots=result.slots, dropped_slots=result.dropped_slots,
                                    timings_ms=timings)
    update_stage(job_id, STAGE_NAME, "completed", 100, "Prompt matches the archive")
    append_event(job_id, "stage_completed", STAGE_NAME, {"slots": result.slots})
    return StageResult(stage=STAGE_NAME, status="completed", message="Prompt matches the archive",
                       data={"slots": result.slots})
