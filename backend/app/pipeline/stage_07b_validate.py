from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = str(PROJECT_ROOT / ".venv" / "bin" / "python")
TRANSCRIPTS_DIR_ALIGNED = DATA_DIR / "transcripts_cleaned_aligned"
TRANSCRIPTS_DIR_FALLBACK = DATA_DIR / "transcripts"


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    storyline_path = context.storyline_path
    update_stage(job_id, "07b_validate_storyline", "running", 5, "Starting LLM editorial validation")
    append_event(job_id, "stage_started", "07b_validate_storyline", {"storyline_path": storyline_path})

    sp = Path(storyline_path)
    validated_path = sp.parent / (sp.stem + "_validated.json")
    max_iter = int(request.get("max_validation_iterations", 3))

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "07b_validate_storyline_with_llm.py"),
        "--storyline", str(sp),
        "--transcripts-dir", str(TRANSCRIPTS_DIR_ALIGNED),
        "--fallback-transcripts-dir", str(TRANSCRIPTS_DIR_FALLBACK),
        "--output", str(validated_path),
        "--max-iterations", str(max_iter),
    ]

    update_stage(job_id, "07b_validate_storyline", "running", 15, "Validating fragments with LLM")
    code, stdout, stderr = run_script(job_id, "07b_validate_storyline", cmd)

    if code != 0:
        raise RuntimeError(f"07b_validate_storyline_with_llm.py exited {code}: {stderr[-1000:]}")

    if not validated_path.exists():
        raise RuntimeError(f"Validated storyline not found: {validated_path}")

    # Parse result
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

    job = load_job(job_id)
    job["paths"]["validated_storyline_path"] = str(validated_path)
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    # Store warnings in job
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
