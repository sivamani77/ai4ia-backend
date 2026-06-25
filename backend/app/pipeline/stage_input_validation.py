from __future__ import annotations

from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, STORYLINES_DIR, EXPORTS_DIR
from app.pipeline.base import PipelineContext, StageResult
from app.services.job_store import append_event
from app.services.video_job_runner import update_stage


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    update_stage(job_id, "input_validation", "running", 10, "Checking required paths")
    issues = []
    for d in [STORYLINES_DIR, EXPORTS_DIR, DATA_DIR]:
        if not d.exists():
            issues.append(f"Directory not found: {d}")
    for s in ["07_generate_storyline.py", "07b_validate_storyline_with_llm.py",
              "08_generate_assets.py", "09_export_video.py"]:
        if not (SCRIPTS_DIR / s).exists():
            issues.append(f"Script not found: {s}")
    if not request.get("prompt", "").strip():
        issues.append("prompt is required")
    if issues:
        raise RuntimeError("; ".join(issues))
    update_stage(job_id, "input_validation", "completed", 100, "Validation passed")
    append_event(job_id, "stage_completed", "input_validation", {"issues": []})
    return StageResult(
        stage="input_validation",
        status="completed",
        message="Validation passed",
    )
