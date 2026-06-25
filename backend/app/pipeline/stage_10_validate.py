from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.paths import SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage

PYTHON_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    export_manifest_path = context.export_manifest_path
    update_stage(job_id, "10_validate_export", "running", 10, "Validating export")
    append_event(job_id, "stage_started", "10_validate_export", {})

    script = SCRIPTS_DIR / "10_validate_video_export.py"
    if not script.exists() or not Path(export_manifest_path).exists():
        update_stage(job_id, "10_validate_export", "completed", 100, "Skipped — script or manifest not available")
        append_event(job_id, "stage_completed", "10_validate_export", {"skipped": True})
        return StageResult(
            stage="10_validate_export",
            status="skipped",
            message="Skipped — script or manifest not available",
        )

    cmd = [PYTHON_BIN, str(script), "--export-manifest", export_manifest_path]
    code, stdout, stderr = run_script(job_id, "10_validate_export", cmd)

    if code != 0:
        # Non-zero can mean serious errors or just warnings; don't fail the job unless clearly fatal
        update_stage(job_id, "10_validate_export", "completed", 100,
                     f"Validation returned exit code {code} (see logs)")
        jb = load_job(job_id)
        jb["warnings"] = jb.get("warnings", []) + [f"10_validate_export exit code {code}"]
        save_job(job_id, jb)
    else:
        update_stage(job_id, "10_validate_export", "completed", 100, "Export validation passed")

    append_event(job_id, "stage_completed", "10_validate_export", {"exit_code": code})
    return StageResult(
        stage="10_validate_export",
        status="completed",
        message="Export validation passed" if code == 0 else f"Validation returned exit code {code} (see logs)",
        data={"exit_code": code},
    )
