from __future__ import annotations

import logging
from typing import Any

from app.core.settings import settings
from app.pipeline.base import PipelineContext, StageResult
from app.services.guardrail_service import VIDEO_POLICY, classify
from app.services.job_store import append_event
from app.services.video_job_runner import update_stage

logger = logging.getLogger("ai4ia.pipeline.guardrail")

STAGE_NAME = "prompt_guardrail"

REJECTION_MESSAGE = (
    "Your prompt couldn't be processed because it appears to violate our "
    "content guidelines. Please submit a prompt related to creating AI4IA "
    "video content."
)


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    """Classify the job's prompt against the video content policy.

    The classification itself lives in guardrail_service so the archive-query
    routes can share it; this stage only maps the verdict onto job state.
    """
    prompt = (request.get("prompt") or "").strip()

    if not settings.prompt_guardrail_enabled:
        return StageResult(stage=STAGE_NAME, status="skipped", message="Guardrail disabled")

    update_stage(job_id, STAGE_NAME, "running", 50, "Checking prompt against content policy")
    append_event(job_id, "stage_started", STAGE_NAME, {})

    verdict = classify(
        prompt,
        policy=VIDEO_POLICY,
        trace={"job_id": job_id, "stage": STAGE_NAME, "agent": "prompt_guardrail",
               "decision_label": "prompt_moderation"},
    )

    if verdict.failed_open:
        update_stage(job_id, STAGE_NAME, "completed", 100, "Guardrail check unavailable — allowed by default")
        append_event(job_id, "stage_completed", STAGE_NAME, {"fail_open": True})
        return StageResult(stage=STAGE_NAME, status="completed",
                           message="Guardrail check unavailable — allowed by default")

    if not verdict.allowed:
        if verdict.category == "guardrail_unavailable":
            update_stage(job_id, STAGE_NAME, "failed", 100, REJECTION_MESSAGE)
            append_event(job_id, "stage_failed", STAGE_NAME, {"error": "guardrail_unavailable"})
            return StageResult(stage=STAGE_NAME, status="failed", message=REJECTION_MESSAGE)

        logger.info("Prompt guardrail rejected job %s prompt (category=%s, reason=%s)",
                    job_id, verdict.category, verdict.reason)
        update_stage(job_id, STAGE_NAME, "failed", 100, REJECTION_MESSAGE)
        append_event(job_id, "stage_failed", STAGE_NAME,
                     {"category": verdict.category, "reason": verdict.reason})
        return StageResult(
            stage=STAGE_NAME,
            status="failed",
            message=REJECTION_MESSAGE,
            data={"category": verdict.category, "reason": verdict.reason},
        )

    update_stage(job_id, STAGE_NAME, "completed", 100, "Prompt passed content policy check")
    append_event(job_id, "stage_completed", STAGE_NAME, {})
    return StageResult(stage=STAGE_NAME, status="completed", message="Prompt passed content policy check")
