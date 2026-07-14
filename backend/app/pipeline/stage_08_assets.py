from __future__ import annotations
import sys

import json
from pathlib import Path
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.pipeline.s3_helpers import use_s3, download, upload, upload_dir, tmp_dir, cleanup_tmp_dir
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = sys.executable

_S3_ASSETS_PREFIX = "generated_assets"


def _prepare_storyline(validated_path: str, job_id: str) -> Path:
    """Return local path to validated storyline JSON, downloading from S3 if needed."""
    if not use_s3():
        return Path(validated_path)

    vp = Path(validated_path)
    work = tmp_dir(f"08_{job_id}")
    local = work / vp.name
    if not local.exists():
        download(f"raw/storylines/{vp.name}", local)
    return local


def _assets_dir(job_id: str) -> Path:
    """Job-scoped generated-assets directory — never shared between concurrent
    jobs. Must always be passed to 08_generate_assets.py via AI4IA_LOCAL_ASSETS_DIR
    (see run() below) — this function only computing the path is not enough,
    the subprocess needs to actually be told to write there."""
    if use_s3():
        return tmp_dir(f"08_{job_id}") / "generated_assets"
    job_assets = DATA_DIR / "jobs" / job_id / "generated_assets"
    job_assets.mkdir(parents=True, exist_ok=True)
    return job_assets


def _manifest_path(job_id: str) -> Path:
    return _assets_dir(job_id) / "latest_asset_manifest.json"


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    validated_path = context.validated_storyline_path
    update_stage(job_id, "08_assets", "running", 5, "Starting voiceover and visual asset generation")
    append_event(job_id, "stage_started", "08_assets", {"validated_path": validated_path})

    if use_s3():
        update_stage(job_id, "08_assets", "running", 8, "Downloading storyline from S3")

    storyline_local = _prepare_storyline(validated_path, job_id)

    # Always override GENERATED_ASSETS_DIR to a job-scoped directory — in S3
    # mode that's a /tmp dir uploaded and discarded after; in local mode it's
    # DATA_DIR/jobs/{job_id}/generated_assets. Without this override the
    # script falls back to its own shared default (data/generated_assets),
    # which _manifest_path() below never looks at, and two concurrent jobs
    # would also collide on that shared dir's fixed-name manifest file.
    assets_dir = _assets_dir(job_id)
    env_overrides: dict[str, str] = {"AI4IA_LOCAL_ASSETS_DIR": str(assets_dir)}

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "08_generate_assets.py"),
        "--storyline", str(storyline_local),
    ]

    update_stage(job_id, "08_assets", "running", 10, "Generating voiceovers and visual assets")
    code, stdout, stderr = run_script(job_id, "08_assets", cmd, extra_env=env_overrides or None)

    if code != 0:
        raise RuntimeError(f"08_generate_assets.py exited {code}: {stderr[-1000:]}")

    # Find the manifest — script writes it into assets_dir (AI4IA_LOCAL_ASSETS_DIR above)
    manifests = list(assets_dir.rglob("latest_asset_manifest.json"))
    if not manifests:
        raise RuntimeError(f"Asset manifest not found under {assets_dir}")
    manifest_path = manifests[0]

    if use_s3():
        update_stage(job_id, "08_assets", "running", 85, "Uploading generated assets to S3")
        # Upload all generated assets (audio, cards, visual PNGs, manifest)
        upload_dir(assets_dir, _S3_ASSETS_PREFIX)
        # Upload manifest under a job-scoped key so concurrent jobs don't collide
        upload(manifest_path, f"{_S3_ASSETS_PREFIX}/{job_id}_manifest.json")

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

    if use_s3():
        cleanup_tmp_dir(f"08_{job_id}")

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
