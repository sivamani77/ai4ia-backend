from __future__ import annotations
import sys
import threading
import uuid

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.pipeline.s3_helpers import use_s3, download, download_prefix, key_exists, tmp_dir, cleanup_tmp_dir
from app.pipeline.timeline_budget import SOURCE_DURATION_INDEX_S3_KEY
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = sys.executable


def _prepare_inputs(storyline_path: str, job_id: str) -> tuple[Path, Path, Path, Path]:
    """
    In S3 mode: download storyline JSON + transcripts from S3 to /tmp.
    In local mode: use paths as-is.
    Returns (storyline_local, transcripts_dir, fallback_transcripts_dir, output_path).
    """
    if not use_s3():
        sp = Path(storyline_path)
        validated = sp.parent / (sp.stem + "_validated.json")
        return sp, DATA_DIR / "transcripts_cleaned_aligned", DATA_DIR / "transcripts", validated

    # S3 mode — work entirely in /tmp
    work = tmp_dir(f"07b_{job_id}")

    # Download storyline JSON from S3
    # storyline_path is a local path like /app/data/storylines/storyline_xxx.json
    # In S3 mode, stage_07 wrote the file to S3 at raw/storylines/<filename>
    sp_local = Path(storyline_path)
    storyline_local = work / sp_local.name
    if not storyline_local.exists():
        s3_key = f"raw/storylines/{sp_local.name}"
        download(s3_key, storyline_local)

    # Download transcripts from S3 to /tmp
    transcripts_dir = work / "transcripts_cleaned_aligned"
    fallback_dir = work / "transcripts"
    if not transcripts_dir.exists() or not any(transcripts_dir.iterdir()):
        download_prefix("raw/transcripts_cleaned/", transcripts_dir)
    if not fallback_dir.exists() or not any(fallback_dir.iterdir()):
        download_prefix("raw/transcripts/", fallback_dir)

    validated_local = work / (sp_local.stem + "_validated.json")
    return storyline_local, transcripts_dir, fallback_dir, validated_local


def _prepare_source_duration_index(job_id: str) -> dict[str, str]:
    """Download the small source-video-duration sidecar (see stage_07_storyline
    for why this stage needs it without downloading full video files). Optional —
    the duration top-up loop just treats lengths as unknown if it's missing."""
    if not use_s3():
        return {}
    if not key_exists(SOURCE_DURATION_INDEX_S3_KEY):
        return {}
    work = tmp_dir(f"07b_{job_id}")
    local = work / "source_video_durations.json"
    if not local.exists():
        download(SOURCE_DURATION_INDEX_S3_KEY, local)
    return {"AI4IA_SOURCE_DURATION_INDEX": str(local)}


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    storyline_path = context.storyline_path
    update_stage(job_id, "07b_validate_storyline", "running", 5, "Starting LLM editorial validation")
    append_event(job_id, "stage_started", "07b_validate_storyline", {"storyline_path": storyline_path})

    if use_s3():
        update_stage(job_id, "07b_validate_storyline", "running", 8, "Downloading inputs from S3")

    sp, transcripts_dir, fallback_dir, validated_path = _prepare_inputs(storyline_path, job_id)
    max_iter = int(request.get("max_validation_iterations", 3))
    env_overrides = _prepare_source_duration_index(job_id)

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "07b_validate_storyline_with_llm.py"),
        "--storyline", str(sp),
        "--transcripts-dir", str(transcripts_dir),
        "--fallback-transcripts-dir", str(fallback_dir),
        "--output", str(validated_path),
        "--max-iterations", str(max_iter),
    ]

    update_stage(job_id, "07b_validate_storyline", "running", 15, "Validating fragments with LLM")
    code, stdout, stderr = run_script(job_id, "07b_validate_storyline", cmd, extra_env=env_overrides or None)

    if code != 0:
        raise RuntimeError(f"07b_validate_storyline_with_llm.py exited {code}: {stderr[-1000:]}")

    if not validated_path.exists():
        raise RuntimeError(f"Validated storyline not found: {validated_path}")

    # Upload validated storyline to S3
    if use_s3():
        from app.pipeline.s3_helpers import upload
        s3_key = f"raw/storylines/{validated_path.name}"
        upload(validated_path, s3_key)

    # Parse result
    vdata = None
    try:
        vdata = json.loads(validated_path.read_text(encoding="utf-8"))
        tl = vdata.get("final_timeline") or []
        gates = vdata.get("editorial_quality_gates") or {}
        warnings_list = []
        for item in tl:
            if item.get("type") == "speaker_clip" and not item.get("quote_complete"):
                warnings_list.append({
                    "fragment_id": item.get("fragment_id", ""),
                    "issue": "quote_incomplete",
                    "note": item.get("quote_completeness_note", ""),
                })
        approved = sum(1 for x in tl if x.get("type") == "speaker_clip")
        repair_iters = vdata.get("repair_iterations", 1)
        sections_covered = gates.get("sections_covered", [])
        nb_count = sum(1 for x in tl if x.get("type") == "narration_broll")
    except Exception:
        gates, warnings_list, approved, repair_iters, sections_covered, nb_count = {}, [], 0, 1, [], 0

    # Stash the parsed content in-memory: in S3 mode, cleanup_tmp_dir below
    # deletes validated_path from disk before this function returns, so
    # anything needing the content afterward (the storyline cache write)
    # must read it from here, not re-read the (by-then-gone) file.
    context.validated_storyline_data = vdata

    job = load_job(job_id)
    job["paths"]["validated_storyline_path"] = str(validated_path)
    if use_s3():
        job["paths"]["validated_storyline_s3_key"] = f"raw/storylines/{validated_path.name}"
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    if warnings_list:
        jb = load_job(job_id)
        jb["warnings"] = jb.get("warnings", []) + [w["note"] or w["issue"] for w in warnings_list]
        save_job(job_id, jb)

    update_stage(job_id, "07b_validate_storyline", "completed", 100,
                 f"{approved} speaker clips, {nb_count} narration brolls, {len(sections_covered)} sections covered")

    append_event(job_id, "validation", "07b_validate_storyline", {
        "approved_fragments": approved,
        "narration_broll_count": nb_count,
        "repair_iterations": repair_iters,
        "sections_covered": sections_covered,
        "quality_gates": gates,
    })
    for w in warnings_list:
        append_event(job_id, "validation_warning", "07b_validate_storyline", w)
    append_event(job_id, "quality_gate", "07b_validate_storyline", gates)
    append_event(job_id, "stage_completed", "07b_validate_storyline", {"validated_path": str(validated_path)})

    if use_s3():
        cleanup_tmp_dir(f"07b_{job_id}")

    _emit_validation_trace(job_id, repair_iters)

    context.validated_storyline_path = str(validated_path)
    return StageResult(
        stage="07b_validate_storyline",
        status="completed",
        message=f"{approved} speaker clips, {nb_count} narration brolls, {len(sections_covered)} sections covered",
        data={
            "validated_storyline_path": str(validated_path),
            "approved_fragments": approved,
            "narration_broll_count": nb_count,
            "repair_iterations": repair_iters,
            "sections_covered": sections_covered,
        },
    )


def _emit_validation_trace(job_id: str, repair_iterations: int) -> None:
    """Emit a post-hoc LLM trace for the A7 Validation Agent (subprocess call)."""
    def _insert() -> None:
        try:
            from app.core.config import VLLM_MODEL
            from app.db.repositories.llm_trace_repository import LLMTraceRepository
            job = load_job(job_id)
            for i in range(max(1, repair_iterations)):
                LLMTraceRepository().create({
                    "id": f"lt_{uuid.uuid4().hex}",
                    "job_id": job_id,
                    "user_id": job.get("user_id"),
                    "stage": "07b_validate_storyline",
                    "agent": "validation",
                    "model": VLLM_MODEL,
                    "call_type": "generate_json",
                    "decision_label": f"editorial_validation_iteration_{i + 1}",
                    "retry_count": 0,
                    "called_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            pass

    threading.Thread(target=_insert, daemon=True).start()
