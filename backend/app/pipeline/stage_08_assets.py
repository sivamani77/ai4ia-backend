from __future__ import annotations

import json
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "python")
LATEST_ASSET_MANIFEST = DATA_DIR / "generated_assets" / "latest_asset_manifest.json"


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    validated_path = context.validated_storyline_path
    update_stage(job_id, "08_assets", "running", 5, "Starting voiceover and visual asset generation")
    append_event(job_id, "stage_started", "08_assets", {"validated_path": validated_path})

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "08_generate_assets.py"),
        "--storyline", validated_path,
    ]

    update_stage(job_id, "08_assets", "running", 10, "Generating voiceovers and visual assets")
    code, stdout, stderr = run_script(job_id, "08_assets", cmd)

    if code != 0:
        raise RuntimeError(f"08_generate_assets.py exited {code}: {stderr[-1000:]}")

    manifest_path = LATEST_ASSET_MANIFEST
    if not manifest_path.exists():
        raise RuntimeError(f"Asset manifest not found at {manifest_path}")

    # Parse manifest for details
    try:
        mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
        tl = mdata.get("export_timeline") or []
        nb_items = [x for x in tl if (x.get("type") or x.get("segment_type")) == "narration_broll"]
        visuals = [
            {"section_id": x.get("section_id"), "type": x.get("visual_asset_type"),
             "title": x.get("visual_title"), "path": x.get("render_visual_path") or x.get("visual_path")}
            for x in nb_items if x.get("render_visual_path") or x.get("visual_path")
        ]
        svos = mdata.get("voice", {}).get("section_voiceovers") or []
        vqg = mdata.get("visual_quality_gates") or {}
    except Exception:
        visuals, svos, vqg = [], [], {}

    job = load_job(job_id)
    job["paths"]["asset_manifest_path"] = str(manifest_path)
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    for v in visuals:
        append_event(job_id, "asset", "08_assets", {
            "section": v.get("section_id"),
            "type": v.get("type"),
            "title": v.get("title"),
            "status": "generated",
        })

    update_stage(job_id, "08_assets", "completed", 100,
                 f"{len(visuals)} visuals, {len(svos)} voiceovers generated",
                 visual_quality_gates=vqg)
    append_event(job_id, "stage_completed", "08_assets", {
        "manifest_path": str(manifest_path),
        "visuals_count": len(visuals),
        "voiceovers_count": len(svos),
        "visual_quality_gates": vqg,
    })

    context.asset_manifest_path = str(manifest_path)
    return StageResult(
        stage="08_assets",
        status="completed",
        message=f"{len(visuals)} visuals, {len(svos)} voiceovers generated",
        data={
            "asset_manifest_path": str(manifest_path),
            "visuals_count": len(visuals),
            "voiceovers_count": len(svos),
        },
    )
