from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.paths import EXPORTS_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    manifest_path = context.asset_manifest_path
    update_stage(job_id, "09_export", "running", 5, "Starting final video export")
    append_event(job_id, "stage_started", "09_export", {"manifest_path": manifest_path})

    # Determine output path using job_id for predictability
    try:
        mdata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        raw_title = mdata.get("title") or mdata.get("storyline", {}).get("title") or ""
        if raw_title:
            import re
            safe_title = re.sub(r"[^a-z0-9]+", "_", raw_title.lower()).strip("_")[:40]
        else:
            safe_title = ""
    except Exception:
        safe_title = ""

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if safe_title:
        video_filename = f"{job_id}_{safe_title}.mp4"
    else:
        video_filename = f"{job_id}.mp4"
    video_path = EXPORTS_DIR / video_filename

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "09_export_video.py"),
        "--manifest", manifest_path,
        "--output", str(video_path),
    ]
    if request.get("burn_subtitles"):
        cmd.append("--burn-subtitles")

    update_stage(job_id, "09_export", "running", 15, "Rendering video segments")
    append_event(job_id, "export", "09_export", {"progress": 15, "message": "Rendering video segments"})

    code, stdout, stderr = run_script(job_id, "09_export", cmd)

    if code != 0:
        raise RuntimeError(f"09_export_video.py exited {code}: {stderr[-1000:]}")

    if not video_path.exists():
        raise RuntimeError(f"Expected export video not found: {video_path}")

    export_manifest_path = EXPORTS_DIR / f"{video_path.stem}_export_manifest.json"
    timeline_metadata_path = EXPORTS_DIR / f"{video_path.stem}_timeline_metadata.json"
    srt_path = EXPORTS_DIR / f"{video_path.stem}.srt"
    vtt_path = EXPORTS_DIR / f"{video_path.stem}.vtt"

    # Parse export manifest
    try:
        emdata = json.loads(export_manifest_path.read_text(encoding="utf-8")) if export_manifest_path.exists() else {}
        qc = emdata.get("quality_checks") or {}
        duration = emdata.get("subtitles", {}).get("coverage", {})
        gen_vis = emdata.get("generated_visual_segments", 0)
        muted_arc = emdata.get("muted_archive_narration_segments", 0)
        sub_cov = emdata.get("subtitles", {}).get("coverage", {})
    except Exception:
        qc, gen_vis, muted_arc, sub_cov = {}, 0, 0, {}

    job = load_job(job_id)
    job["paths"]["video_path"] = str(video_path)
    job["paths"]["export_manifest_path"] = str(export_manifest_path) if export_manifest_path.exists() else None
    job["paths"]["timeline_metadata_path"] = str(timeline_metadata_path) if timeline_metadata_path.exists() else None
    job["paths"]["srt_path"] = str(srt_path) if srt_path.exists() else None
    job["paths"]["vtt_path"] = str(vtt_path) if vtt_path.exists() else None
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    update_stage(job_id, "09_export", "completed", 100,
                 f"Export complete — generated_visuals={gen_vis}, muted_archive={muted_arc}")
    append_event(job_id, "export", "09_export", {
        "progress": 100,
        "message": "Video export complete",
        "video_path": str(video_path),
        "generated_visual_segments": gen_vis,
        "muted_archive_narration_segments": muted_arc,
        "subtitle_coverage": sub_cov,
    })
    append_event(job_id, "stage_completed", "09_export", {
        "video_path": str(video_path),
        "quality_checks": qc,
    })

    context.video_path = str(video_path)
    context.export_manifest_path = str(export_manifest_path)
    context.srt_path = str(srt_path) if srt_path.exists() else None
    context.vtt_path = str(vtt_path) if vtt_path.exists() else None
    return StageResult(
        stage="09_export",
        status="completed",
        message=f"Export complete — generated_visuals={gen_vis}, muted_archive={muted_arc}",
        data={
            "video_path": str(video_path),
            "export_manifest_path": str(export_manifest_path),
            "generated_visual_segments": gen_vis,
            "muted_archive_narration_segments": muted_arc,
        },
    )
