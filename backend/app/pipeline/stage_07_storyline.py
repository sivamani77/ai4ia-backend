from __future__ import annotations
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import json
from datetime import datetime, timezone
from typing import Any

from app.core.paths import DATA_DIR, SCRIPTS_DIR, STORYLINES_DIR, PROJECT_ROOT
from app.core.settings import settings
from app.pipeline.base import PipelineContext, StageResult, run_script, _newest_file_after
from app.pipeline.s3_helpers import use_s3, upload, download_if_missing
from app.pipeline.timeline_budget import SOURCE_DURATION_INDEX_S3_KEY
from app.services import processing_records
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = sys.executable

_S3_EMBEDDINGS_PREFIX = "raw/embeddings"
_EMBEDDINGS_FILES = [
    "classified_fragments_embeddings.jsonl",
    "fragment_embedding_matrix.npy",
    "fragment_embedding_metadata.json",
]


def _prepare_embeddings(job_id: str) -> dict[str, str]:
    """Download embeddings from S3 to /tmp and return AI4IA_EMBEDDINGS_DIR env override."""
    return download_if_missing("07", job_id, _S3_EMBEDDINGS_PREFIX, _EMBEDDINGS_FILES, "AI4IA_EMBEDDINGS_DIR")


def _prepare_source_duration_index(job_id: str) -> dict[str, str]:
    """Download the small source-video-duration sidecar so plan-time footage-length
    checks (clamp_timeline_windows_to_source / fill_timeline_to_target) have real
    numbers even though this stage never downloads the full video files.

    Optional: the index may not exist yet (first deploy) — proceed without it,
    the pipeline falls back to treating lengths as unknown, same as before.
    """
    prefix, _, filename = SOURCE_DURATION_INDEX_S3_KEY.rpartition("/")
    return download_if_missing(
        "07", job_id, prefix, [filename], "AI4IA_SOURCE_DURATION_INDEX",
        required_key=SOURCE_DURATION_INDEX_S3_KEY,
    )


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    update_stage(job_id, "07_storyline", "running", 5, "Starting storyline generation")
    append_event(job_id, "stage_started", "07_storyline", {"prompt": request.get("prompt", "")})

    import time
    ts_before = time.time()

    if use_s3():
        update_stage(job_id, "07_storyline", "running", 8, "Downloading embeddings from S3")
    env_overrides = _prepare_embeddings(job_id)
    env_overrides.update(_prepare_source_duration_index(job_id))

    target_duration = request.get("target_duration")
    if not target_duration:
        raise ValueError(
            "target_duration is required — the pipeline treats it as the minimum "
            "final video duration and will not assume a default."
        )

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "07_generate_storyline.py"),
        "--prompt", request["prompt"],
        "--duration", str(target_duration),
        "--style", request.get("style", "documentary"),
        "--job-id", job_id,
        # Passed explicitly, although the script defaults to the same setting,
        # so the corpus a job actually retrieved from is recorded in its
        # script_log rather than being an invisible default. Omitting the flag
        # is how stage 07 previously inherited "original_v1" and silently
        # retrieved from 220 of the 309 videos.
        "--corpus-version", settings.storyline_corpus_version,
    ]
    resolved_speaker = getattr(context, "resolved_speaker", None)
    if resolved_speaker:
        cmd += ["--requested-speaker", resolved_speaker]
    coverage_path = Path(tempfile.gettempdir()) / f"ai4ia_coverage_{job_id}.json"
    coverage_path.unlink(missing_ok=True)
    cmd += ["--coverage-file", str(coverage_path)]

    update_stage(job_id, "07_storyline", "running", 20, "Running storyline generation script")
    with processing_records.exchange_log(job_id, "07_storyline", env_overrides):
        code, stdout, stderr = run_script(job_id, "07_storyline", cmd, extra_env=env_overrides)

    if code != 0:
        raise RuntimeError(f"07_generate_storyline.py exited {code}: {stderr[-1000:]}")

    coverage = _read_coverage(coverage_path)
    if coverage:
        intent_name, intent_version = coverage.pop("intent", None), coverage.pop("intent_version", None)
        processing_records.record_layer(job_id, coverage=coverage, intent=intent_name, intent_version=intent_version)
        append_event(job_id, "coverage", "07_storyline", {
            k: coverage.get(k) for k in ("verdict", "method", "top_similarity", "missing_aspects")})
    if coverage.get("verdict") == "not_covered" and settings.coverage_decline_enabled:
        return _decline(job_id, coverage)

    update_stage(job_id, "07_storyline", "running", 85, "Locating generated storyline")
    # Job-scoped lookup first — the script names its output storyline_{job_id}_*.json
    # when given --job-id (always true here). This is what actually prevents two
    # concurrent jobs' stage-07 subprocesses from picking up each other's file;
    # _newest_file_after (pure "newest mtime in the shared directory") is kept
    # only as a fallback for old storyline files that predate job_id scoping.
    job_scoped_matches = sorted(
        STORYLINES_DIR.glob(f"storyline_{job_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    storyline_path = job_scoped_matches[-1] if job_scoped_matches else \
        _newest_file_after(STORYLINES_DIR, ts_before, "storyline_*.json")
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

    intent = (sdata.get("storyline") or {}).get("intent") if "sdata" in dir() else None
    if intent:
        processing_records.record_layer(job_id, intent=intent.get("name"), intent_version=intent.get("version"),
                                        template_hash=intent.get("template_hash"))

    # Emit post-hoc LLM traces from the storyline JSON (subprocess calls can't be
    # intercepted live, so we reconstruct them from what the script recorded).
    _emit_storyline_traces(job_id, sdata if "sdata" in dir() else {}, request)

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


def _read_coverage(path: Path) -> dict[str, Any]:
    """The script's coverage verdict; empty when it wrote none."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    finally:
        path.unlink(missing_ok=True)


def _decline(job_id: str, coverage: dict[str, Any]) -> StageResult:
    """The archive cannot answer the prompt: fail the job with a specific message (§8.3)."""
    message = coverage.get("user_message") or "The AI4IA archive doesn't have talks on this topic."
    processing_records.record_layer(job_id, outcome="declined", reason_code=coverage.get("reason_code"),
                                    user_message=message, suggestions=coverage.get("suggestions") or [])
    update_stage(job_id, "07_storyline", "failed", 100, message)
    append_event(job_id, "stage_failed", "07_storyline", {"reason_code": coverage.get("reason_code")})
    return StageResult(stage="07_storyline", status="failed", message=message, data={
        "error_code": "declined", "reason_code": coverage.get("reason_code"),
        "user_message": message, "suggestions": coverage.get("suggestions") or [],
    })


def _emit_storyline_traces(job_id: str, sdata: dict, request: dict) -> None:
    """Emit post-hoc LLM call traces for the agents that ran inside the subprocess.

    Stage 07 runs 07_generate_storyline.py as a subprocess so we can't intercept
    its LLM calls live. We reconstruct trace rows from what the script recorded
    in the storyline JSON.

    Only real LLM/model calls get trace rows:
      A2 Retrieval    — nomic-embed-text (embedding)
      A4 Director     — strong LLM (video mode + section structure decision)
      A6 Storyline    — strong LLM (full structured storyline generation)

    NOT traced (rule-based, no model call):
      A3 Classification   — scored heuristic, no LLM
      A5 Theme Synthesis  — _synthesize_documentary_context/tension/evidence are
                            keyword-template functions, zero LLM involvement
      Relay Assembler     — enrich_storyline_with_director_plan assigns clip-slots
                            from a lookup table after A4 decides the mode
    """
    def _insert() -> None:
        try:
            from app.db.repositories.llm_trace_repository import LLMTraceRepository
            repo = LLMTraceRepository()
            now = datetime.now(timezone.utc).isoformat()
            models = sdata.get("models") or {}
            planning = sdata.get("planning_summary") or {}
            story = sdata.get("storyline") or {}
            director = story.get("director_plan") or {}
            job = load_job(job_id)
            user_id = job.get("user_id")

            base = {
                "job_id": job_id,
                "user_id": user_id,
                "stage": "07_storyline",
                "retry_count": 0,
                "called_at": now,
            }

            # A2 — Retrieval Agent: embedding call to nomic-embed-text
            if models.get("embedding_model"):
                repo.create({**base, "id": f"lt_{uuid.uuid4().hex}", "agent": "retrieval",
                              "model": models["embedding_model"], "call_type": "embed",
                              "decision_label": "fragment_retrieval"})

            # A4 — Director Agent: LLM decides video_mode and recommended section structure
            video_mode = planning.get("video_mode") or director.get("video_mode")
            if models.get("strong_llm_model"):
                repo.create({**base, "id": f"lt_{uuid.uuid4().hex}", "agent": "director",
                              "model": models["strong_llm_model"], "call_type": "generate_json",
                              "decision_label": f"video_mode:{video_mode or 'unknown'}"})

            # A6 — Storyline Agent: LLM generates the full structured storyline JSON
            if models.get("strong_llm_model"):
                repo.create({**base, "id": f"lt_{uuid.uuid4().hex}", "agent": "storyline",
                              "model": models["strong_llm_model"], "call_type": "generate_json",
                              "decision_label": "final_storyline_generation"})
        except Exception:
            pass

    threading.Thread(target=_insert, daemon=True).start()
