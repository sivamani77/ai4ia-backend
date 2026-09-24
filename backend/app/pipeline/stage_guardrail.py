from __future__ import annotations

import logging
import time
from typing import Any

from app.core.settings import settings
from app.pipeline.base import PipelineContext, StageResult
from app.services import processing_records
from app.core.error_messages import GUARDRAIL_OFF_TOPIC_MESSAGE
from app.services.guardrail_service import GUARDRAIL_UNAVAILABLE_MESSAGE, VIDEO_POLICY, classify
from app.services.prompt_rules import check_prompt
from app.services.job_store import append_event, load_job
from app.services.video_job_runner import update_stage

logger = logging.getLogger("ai4ia.pipeline.guardrail")

STAGE_NAME = "prompt_guardrail"

REJECTION_MESSAGE = (
    "Your prompt couldn't be processed because it appears to violate our "
    "content guidelines. Please submit a prompt related to creating AI4IA "
    "video content."
)


def _job_user_id(job_id: str) -> str | None:
    try:
        return load_job(job_id).get("user_id") or None
    except Exception:
        return None


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    """Classify the job's prompt against the video content policy.

    The classification itself lives in guardrail_service so the archive-query
    routes can share it; this stage only maps the verdict onto job state.
    """
    prompt = (request.get("prompt") or "").strip()

    # L0 backstop for jobs created outside the API route; needs no model.
    rules = check_prompt(prompt)
    if rules.rejected:
        processing_records.record_layer(job_id, outcome="l0_rejected", reason_code=rules.reason_code,
                                        user_message=rules.message, flags=rules.flags)
        update_stage(job_id, STAGE_NAME, "failed", 100, rules.message)
        append_event(job_id, "stage_failed", STAGE_NAME, {"reason_code": rules.reason_code})
        return StageResult(stage=STAGE_NAME, status="failed", message=rules.message,
                           data={"reason_code": rules.reason_code, "error_code": "prompt_rejected",
                                 "user_message": rules.message})

    if not settings.prompt_guardrail_enabled:
        return StageResult(stage=STAGE_NAME, status="skipped", message="Guardrail disabled")

    update_stage(job_id, STAGE_NAME, "running", 50, "Checking prompt against content policy")
    append_event(job_id, "stage_started", STAGE_NAME, {})

    started = time.monotonic()
    verdict = classify(
        prompt,
        policy=VIDEO_POLICY,
        trace={"job_id": job_id, "user_id": _job_user_id(job_id), "stage": STAGE_NAME,
               "agent": "prompt_guardrail", "decision_label": "prompt_moderation"},
    )
    timings = {"l1": int((time.monotonic() - started) * 1000)}

    if verdict.failed_open:
        processing_records.record_layer(job_id, flags={"l1_failed_open": True}, timings_ms=timings)
        update_stage(job_id, STAGE_NAME, "completed", 100, "Guardrail check unavailable — allowed by default")
        append_event(job_id, "stage_completed", STAGE_NAME, {"fail_open": True})
        return StageResult(stage=STAGE_NAME, status="completed",
                           message="Guardrail check unavailable — allowed by default")

    if not verdict.allowed:
        unavailable = verdict.category == "guardrail_unavailable"
        if unavailable:
            user_message = GUARDRAIL_UNAVAILABLE_MESSAGE
        elif verdict.category == "off_topic":
            user_message = GUARDRAIL_OFF_TOPIC_MESSAGE
        else:
            user_message = REJECTION_MESSAGE
        processing_records.record_layer(
            job_id, outcome="l1_rejected", reason_code=verdict.category or "guardrail_rejected",
            user_message=user_message, timings_ms=timings,
        )
        logger.info("Prompt guardrail rejected job %s prompt (category=%s, reason=%s)",
                    job_id, verdict.category, verdict.reason)
        update_stage(job_id, STAGE_NAME, "failed", 100, user_message)
        append_event(job_id, "stage_failed", STAGE_NAME,
                     {"category": verdict.category, "reason": verdict.reason})
        return StageResult(
            stage=STAGE_NAME,
            status="failed",
            message=user_message,
            data={"category": verdict.category, "reason": verdict.reason,
                  "error_code": "guardrail_unavailable" if unavailable else "guardrail_rejected",
                  "user_message": user_message},
        )

    processing_records.record_layer(job_id, timings_ms=timings)
    update_stage(job_id, STAGE_NAME, "completed", 100, "Prompt passed content policy check")
    append_event(job_id, "stage_completed", STAGE_NAME, {})
    return StageResult(stage=STAGE_NAME, status="completed", message="Prompt passed content policy check")
