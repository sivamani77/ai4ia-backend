from __future__ import annotations
import sys

import json
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, STORYLINES_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script, _newest_file_after
from app.pipeline.s3_helpers import use_s3, upload
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = sys.executable


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    update_stage(job_id, "07_storyline", "running", 5, "Starting storyline generation")
    append_event(job_id, "stage_started", "07_storyline", {"prompt": request.get("prompt", "")})

    import time
    ts_before = time.time()

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "07_generate_storyline.py"),
        "--prompt", request["prompt"],
        "--duration", str(request.get("target_duration", 150)),
        "--style", request.get("style", "documentary"),
    ]

    update_stage(job_id, "07_storyline", "running", 20, "Running storyline generation script")
    code, stdout, stderr = run_script(job_id, "07_storyline", cmd)

    if code != 0:
        raise RuntimeError(f"07_generate_storyline.py exited {code}: {stderr[-1000:]}")

    update_stage(job_id, "07_storyline", "running", 85, "Locating generated storyline")
    storyline_path = _newest_file_after(STORYLINES_DIR, ts_before, "storyline_*.json")
    if storyline_path is None:
        raise RuntimeError("No storyline JSON found after 07_generate_storyline.py ran")

    # Parse for summary
    try:
        sdata = json.loads(storyline_path.read_text(encoding="utf-8"))
        story = sdata.get("storyline") or sdata
        title = story.get("title") or sdata.get("title") or ""
        cands = len(sdata.get("selected_candidates") or sdata.get("export_timeline") or [])
        logline = story.get("logline") or ""
        sections = [s["section_id"] for s in story.get("sections") or []] if story.get("sections") else []
    except Exception:
        title, cands, logline, sections = "", 0, "", []

    # Upload storyline to S3 so downstream stages (07b, 08) can access it
    if use_s3():
        upload(storyline_path, f"raw/storylines/{storyline_path.name}")

    job = load_job(job_id)
    job["paths"]["storyline_path"] = str(storyline_path)
    if use_s3():
        job["paths"]["storyline_s3_key"] = f"raw/storylines/{storyline_path.name}"
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    update_stage(job_id, "07_storyline", "completed", 100, f"Storyline created — {cands} candidate fragments")
    append_event(job_id, "storyline", "07_storyline", {
        "title": title,
        "logline": logline,
        "candidate_fragments": cands,
        "sections": sections,
        "storyline_path": str(storyline_path),
    })
    append_event(job_id, "stage_completed", "07_storyline", {"storyline_path": str(storyline_path)})

    context.storyline_path = str(storyline_path)
    return StageResult(
        stage="07_storyline",
        status="completed",
        message=f"Storyline created — {cands} candidate fragments",
        data={"storyline_path": str(storyline_path), "title": title, "candidate_fragments": cands},
    )
