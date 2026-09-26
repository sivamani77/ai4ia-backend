from __future__ import annotations
import sys

from pathlib import Path
from typing import Any

from app.core.paths import SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.pipeline.s3_helpers import use_s3, download, tmp_dir, cleanup_tmp_dir
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage

PYTHON_BIN = sys.executable

_S3_EXPORTS_PREFIX = "exports"


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    export_manifest_path = context.export_manifest_path
    update_stage(job_id, "10_validate_export", "running", 10, "Validating export")
    append_event(job_id, "stage_started", "10_validate_export", {})
    # A rerun after a lost worker must not reuse the lost run's partial files.
    cleanup_tmp_dir(f"10_{job_id}")

    script = SCRIPTS_DIR / "10_validate_video_export.py"
    if not script.exists():
        update_stage(job_id, "10_validate_export", "completed", 100, "Skipped — validation script not available")
        append_event(job_id, "stage_completed", "10_validate_export", {"skipped": True})
        return StageResult(
            stage="10_validate_export",
            status="skipped",
            message="Skipped — validation script not available",
        )

    # Resolve local path to manifest — in S3 mode it may be in /tmp from stage 09
    manifest_local = Path(export_manifest_path) if export_manifest_path else None

    if use_s3() and (not manifest_local or not manifest_local.exists()):
        # Try downloading from S3
        if export_manifest_path:
            fname = Path(export_manifest_path).name
            work = tmp_dir(f"10_{job_id}")
            manifest_local = work / fname
            try:
                download(f"{_S3_EXPORTS_PREFIX}/{fname}", manifest_local)
            except Exception:
                manifest_local = None

    if not manifest_local or not manifest_local.exists():
        update_stage(job_id, "10_validate_export", "completed", 100, "Skipped — export manifest not available")
        append_event(job_id, "stage_completed", "10_validate_export", {"skipped": True})
        return StageResult(
            stage="10_validate_export",
            status="skipped",
            message="Skipped — export manifest not available",
        )

    cmd = [PYTHON_BIN, str(script), "--export-manifest", str(manifest_local)]
    code, stdout, stderr = run_script(job_id, "10_validate_export", cmd)

    if code != 0:
        update_stage(job_id, "10_validate_export", "failed", 100,
                     f"Export validation failed (exit code {code})")
        append_event(job_id, "stage_failed", "10_validate_export", {"exit_code": code, "error": stderr[-1000:]})
        raise RuntimeError(f"10_validate_video_export.py failed with exit code {code}:\n{stdout[-1000:]}\n{stderr[-1000:]}")

    update_stage(job_id, "10_validate_export", "completed", 100, "Export validation passed")
    append_event(job_id, "stage_completed", "10_validate_export", {"exit_code": code})
    return StageResult(
        stage="10_validate_export",
        status="completed",
        message="Export validation passed",
        data={"exit_code": code},
    )
