"""
Video job orchestration API.

POST   /api/video-jobs              — start a new job
GET    /api/video-jobs              — list all jobs
GET    /api/video-jobs/{job_id}     — job status + stage progress
DELETE /api/video-jobs/{job_id}     — cancel a running or queued job
GET    /api/video-jobs/{job_id}/events            — SSE stream
GET    /api/video-jobs/{job_id}/storyline         — parsed storyline data
GET    /api/video-jobs/{job_id}/provenance        — full two-layer provenance (decision record + LLM traces)
GET    /api/video-jobs/{job_id}/validation-report — validation gates & decisions
GET    /api/video-jobs/{job_id}/assets            — generated visuals and voiceovers
GET    /api/video-jobs/{job_id}/result            — final video URLs and quality summary
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user
from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket

from app.schemas.pipeline import JobResponse

from app.services.video_job_runner import (
    STAGE_DEFINITIONS,
    create_video_job,
    load_job,
    load_events,
    list_jobs,
    start_video_job_background,
    _cancel_job,
    _redis_reachable,
    _utc_now,
)
from app.services.job_store import load_events_after

router = APIRouter(prefix="/video-jobs", tags=["video-jobs"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class VideoJobRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000, description="Documentary prompt")
    # Required: the pipeline treats this as a minimum duration (floor) — the final
    # video may run somewhat longer, never shorter. No silent default.
    target_duration: int = Field(..., ge=30, le=600)
    style: str = Field("documentary", max_length=50)
    validation_mode: str = Field("iterative_llm", max_length=50)
    visual_mode: str = Field("generated_assets", max_length=50)
    subtitle_mode: str = Field("external", max_length=50)
    max_validation_iterations: int = Field(3, ge=1, le=5)
    burn_subtitles: bool = Field(False)
    # Bypass the storyline cache lookup for this request (still writes a
    # fresh result back to cache afterward, refreshing the cached entry).
    force_refresh: bool = Field(False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RUNNING_PER_USER = 1   # only 1 active render per user at a time
_MAX_QUEUED_PER_USER  = 3   # up to 3 more waiting behind it
_MAX_GLOBAL_QUEUED    = 50  # hard cap on total active+queued jobs


# ---------------------------------------------------------------------------
# In-process rate limiter for job creation (no external dependency)
# ---------------------------------------------------------------------------

# 5 job-creation requests per user per 60 seconds
_create_limiter = TokenBucket(rate=5, per=60.0)


def _assert_owns(job: dict[str, Any], current_user: CurrentUser) -> None:
    """Raise 404 (not 403) to avoid leaking job existence to other users.

    Admins bypass this — same rationale as the identical bypass in
    video_media.py: the admin per-user generations feed needs to read job
    detail for jobs it doesn't own.
    """
    if current_user.persona == "admin":
        return
    if job.get("user_id") and current_user.id and job["user_id"] != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job['job_id']}")


def _build_stages_list(stage_status: dict[str, Any]) -> list[dict[str, Any]]:
    stages = []
    for defn in STAGE_DEFINITIONS:
        key = defn["stage"]
        s = stage_status.get(key, {})
        stages.append({
            "stage": key,
            "label": defn["label"],
            "status": s.get("status", "pending"),
            "progress_percent": s.get("progress_percent", 0),
            "summary": s.get("summary", ""),
        })
    return stages


def _job_response(job: dict[str, Any]) -> JobResponse:
    paths = job.get("paths", {}) or {}
    vp = paths.get("video_path")
    video_url = f"/exports/{Path(vp).name}" if vp else None

    # prompt lives top-level in vjob_ jobs; inside payload for legacy job_ jobs
    prompt = job.get("prompt") or (job.get("payload") or {}).get("prompt")

    data = {
        "job_id": job["job_id"],
        "status": job.get("status", "queued"),
        "current_stage": job.get("current_stage"),
        "progress_percent": job.get("progress_percent", 0),
        "stage_progress_percent": job.get("stage_progress_percent", 0),
        "message": job.get("message"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error") if isinstance(job.get("error"), dict) else None,
        "prompt": prompt,
        "stages": _build_stages_list(job.get("stage_status", {})),
        "video_url": video_url,
        "result_available": job.get("status") == "completed" and bool(vp),
        "warnings": job.get("warnings", []),
        "queue_position": job.get("queue_position"),
    }
    return JobResponse.model_validate(data)


def _require_job(job_id: str) -> dict[str, Any]:
    try:
        return load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", status_code=202)
def create_job(request: VideoJobRequest, current_user: CurrentUser = Depends(get_current_user)):
    effective_user_id = current_user.id

    # Rate limit: max 5 submission attempts per user per 60 s
    if not _create_limiter.allow(effective_user_id):
        _metrics.rate_limit_rejections_total.labels(limiter="job_creation").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before submitting another job.",
        )

    all_jobs = list_jobs()

    # Per-user quota: 1 running + 3 queued max
    if effective_user_id:
        user_running = [
            j for j in all_jobs
            if j.get("user_id") == effective_user_id and j.get("status") == "running"
        ]
        user_queued = [
            j for j in all_jobs
            if j.get("user_id") == effective_user_id and j.get("status") == "queued"
        ]
        if len(user_running) >= _MAX_RUNNING_PER_USER:
            raise HTTPException(
                status_code=429,
                detail="You already have a job running. Wait for it to complete before submitting another.",
            )
        if len(user_queued) >= _MAX_QUEUED_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"You already have {len(user_queued)} job(s) queued. "
                       "Wait for one to complete before submitting another.",
            )

    # Global queue cap — prevent VM overload
    global_active = [j for j in all_jobs if j.get("status") in ("queued", "running")]
    if len(global_active) >= _MAX_GLOBAL_QUEUED:
        raise HTTPException(
            status_code=503,
            detail="System is at capacity. Please try again in a few minutes.",
        )

    req_dict = request.model_dump()
    job_id = create_video_job(req_dict, user_id=effective_user_id)
    start_video_job_background(job_id, req_dict)

    # Queue position — count all queued jobs ahead of this one
    all_jobs = list_jobs()
    queued = [j for j in all_jobs if j.get("status") == "queued"]
    queue_position = len(queued)  # this job is already in the list
    _metrics.job_queue_position.observe(queue_position)

    return {
        "job_id": job_id,
        "status": "queued",
        "queue_position": queue_position,
        "message": f"Job created. Position in queue: {queue_position}",
    }


@router.get("", response_model=None)
def list_video_jobs(
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    jobs = [j for j in list_jobs() if j.get("user_id") == current_user.id]
    total = len(jobs)
    page = jobs[offset: offset + limit]
    return {
        "jobs": [_job_response(j).model_dump() for j in page],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{job_id}", response_model=None)
def get_video_job(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    return _job_response(job).model_dump()


@router.delete("/{job_id}", status_code=202)
def cancel_video_job(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)

    terminal = {"completed", "failed", "cancelled"}
    if job.get("status") in terminal:
        raise HTTPException(
            status_code=409,
            detail={"detail": "Job is not cancellable", "status": job["status"]},
        )

    # Revoke the Celery task if Redis is available
    if _redis_reachable():
        try:
            from celery.result import AsyncResult
            from app.worker.celery_app import celery_app
            AsyncResult(job_id, app=celery_app).revoke(terminate=True, signal="SIGTERM")
        except Exception:
            pass  # non-fatal; PID kill and status write are the primary mechanism

    # Write "cancelled" status and emit SSE event. This DB flag is the
    # cross-container cancellation signal the worker polls in run_script; the
    # worker performs the actual subprocess kill (kill_pid there, not here).
    _cancel_job(job_id, "Job cancelled by user", kill_pid=False)

    return {"job_id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# SSE events stream
# ---------------------------------------------------------------------------

_sse_logger = logging.getLogger("ai4ia.sse")


_TERMINAL_EVENT_TYPES = frozenset({"completed", "failed", "cancelled"})


async def _sse_generator(job_id: str, last_event_id: int = 0):
    """Yield SSE-formatted events using a cursor, polling until job is terminal.

    Terminal status is inferred from the event stream (completed/failed/cancelled
    event_type fields) to avoid a redundant load_job call on every tick.
    A single load_job is done only on the first tick when after_id=0 and no
    events exist yet, to catch jobs that were already terminal before streaming.
    """
    yield "retry: 2000\n\n"

    last_seen = last_event_id
    first_tick = last_event_id == 0
    while True:
        try:
            events = load_events_after(job_id, after_id=last_seen)
        except Exception:
            _sse_logger.exception("load_events_after failed for job %s after_id=%s", job_id, last_seen)
            events = []

        terminal_status: str | None = None
        for event in events:
            last_seen = event["id"]
            yield f"id: {last_seen}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("event_type") in _TERMINAL_EVENT_TYPES:
                terminal_status = event.get("event_type")

        if terminal_status:
            yield f"event: done\ndata: {json.dumps({'status': terminal_status})}\n\n"
            break

        # First tick with no events: job may have completed before this stream opened
        if first_tick and not events:
            try:
                job = load_job(job_id)
                status = job.get("status")
                if status in ("completed", "failed", "cancelled"):
                    yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                    break
            except FileNotFoundError:
                break

        first_tick = False
        await asyncio.sleep(2.0)


@router.get("/{job_id}/events")
async def stream_job_events(job_id: str, last_event_id: int = 0, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    return StreamingResponse(
        _sse_generator(job_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Storyline detail
# ---------------------------------------------------------------------------

@router.get("/{job_id}/storyline")
def get_job_storyline(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    paths = job.get("paths", {})
    vsp = paths.get("validated_storyline_path")
    isp = paths.get("storyline_path")

    if not vsp and not isp:
        return {"job_id": job_id, "status": "not_available", "message": "Storyline not yet generated"}

    target = vsp or isp
    try:
        from app.pipeline.s3_helpers import use_s3, _s3, _bucket
        s3_key = paths.get("validated_storyline_s3_key") or paths.get("storyline_s3_key")
        if use_s3() and s3_key:
            resp = _s3().get_object(Bucket=_bucket(), Key=s3_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
        else:
            data = json.loads(Path(target).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read storyline: {exc}")

    story = data.get("storyline") or data
    tl = data.get("final_timeline") or []
    gates = data.get("editorial_quality_gates") or {}

    sections = []
    for item in tl:
        t = item.get("type") or item.get("segment_type") or ""
        if t in ("title_card", "closing_card"):
            continue
        entry: dict[str, Any] = {
            "section_id": item.get("section_id", ""),
            "type": t,
        }
        if t == "speaker_clip":
            entry.update({
                "speaker": item.get("speaker", ""),
                "quote_text": item.get("quote_text") or item.get("moment_text") or "",
                "quote_complete": item.get("quote_complete"),
                "render_start": item.get("render_start") or item.get("start"),
                "render_end": item.get("render_end") or item.get("end"),
                "fragment_id": item.get("fragment_id", ""),
                "status": "approved",
            })
        elif t == "narration_broll":
            entry.update({
                "voiceover_text": item.get("voiceover_text") or item.get("subtitle_text") or "",
                "visual_title": item.get("visual_title", ""),
                "visual_asset_type": item.get("visual_asset_type", ""),
                "visual_asset_prompt": item.get("visual_asset_prompt", ""),
                "visual_keywords": item.get("visual_keywords", []),
                "visual_path": item.get("render_visual_path") or item.get("visual_path", ""),
                "audio_path": item.get("audio_path", ""),
            })
        sections.append(entry)

    return {
        "job_id": job_id,
        "status": "available",
        "initial_storyline_path": isp,
        "validated_storyline_path": vsp,
        "title": story.get("title") or data.get("title") or "",
        "logline": story.get("logline") or data.get("logline") or "",
        "sections": sections,
        "quality_gates": gates,
        "repair_iterations": data.get("repair_iterations", 0),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@router.get("/{job_id}/validation-report")
def get_job_validation_report(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    paths = job.get("paths", {})
    vsp = paths.get("validated_storyline_path")

    if not vsp or not Path(vsp).exists():
        return {"job_id": job_id, "status": "not_available", "message": "Validation not yet complete"}

    try:
        data = json.loads(Path(vsp).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read validated storyline: {exc}")

    tl = data.get("final_timeline") or []
    gates = data.get("editorial_quality_gates") or {}
    offending = gates.get("offending_items") or {}

    speaker_clips = [x for x in tl if x.get("type") == "speaker_clip"]
    narration_brolls = [x for x in tl if x.get("type") == "narration_broll"]

    # Derive changed decisions: narration_brolls that were originally speaker clips
    # (they typically have fragment_id and no speaker)
    changed_decisions = []
    for nb in narration_brolls:
        fid = nb.get("fragment_id", "")
        note = nb.get("demotion_reason") or nb.get("quote_completeness_note") or ""
        if note:
            changed_decisions.append({
                "fragment_id": fid,
                "before": "speaker_clip",
                "after": "narration_broll",
                "reason": note,
            })

    # Build canonical quality gates response
    qg_response = {
        "all_sections_covered": gates.get("all_sections_covered"),
        "all_speaker_labels_present": gates.get("all_speaker_labels_present"),
        "all_speaker_quotes_complete": gates.get("all_speaker_quotes_complete"),
        "no_instruction_leakage_in_voiceover": gates.get("no_instruction_leakage_in_voiceover"),
        "all_voiceovers_topic_specific": gates.get("all_voiceovers_topic_specific"),
        "final_timeline_approved": gates.get("final_timeline_approved"),
        "has_visual_asset_plan": gates.get("all_narration_broll_have_visual_asset_prompt"),
    }

    warnings_list = data.get("validated_narrative", {}).get("warnings") or data.get("warnings") or []

    return {
        "job_id": job_id,
        "validation_mode": job.get("request", {}).get("validation_mode", "iterative_llm"),
        "iterations": data.get("repair_iterations", 1),
        "approved_fragments": gates.get("speaker_clip_count", len(speaker_clips)),
        "narration_broll_count": gates.get("narration_broll_count", len(narration_brolls)),
        "rejected_fragments": 0,
        "changed_decisions": changed_decisions,
        "sections_covered": gates.get("sections_covered", []),
        "offending_items": offending,
        "quality_gates": qg_response,
        "warnings": [w for w in warnings_list if isinstance(w, str)],
    }


# ---------------------------------------------------------------------------
# Unified provenance endpoint
# ---------------------------------------------------------------------------

@router.get("/{job_id}/provenance")
def get_job_provenance(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Full two-layer provenance for a job.

    Layer 1 — decision_record: high-level osi-1.0 record (agents, models, outcome).
    Layer 2 — llm_call_traces: per-call rows (model, latency, decision_label).

    Also merges the prompt→retrieval→scoring trace (previously /explainability)
    and editorial gates (previously /validation-report) into a single response.
    The old endpoints are preserved for backwards compatibility.
    """
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    paths = job.get("paths", {})

    # Load storyline data (prefer validated). Try S3 key always if present —
    # the s3_key is the reliable path; local paths may be temp dirs that got
    # cleaned up (07b validated file) or inside a remote Docker container.
    src: dict[str, Any] = {}
    validated_data: dict[str, Any] = {}
    for key, dest in [
        ("validated_storyline_path", "validated"),
        ("storyline_path", "initial"),
    ]:
        p = paths.get(key)
        s3_key = paths.get(key.replace("_path", "_s3_key"))
        if not p and not s3_key:
            continue
        try:
            from app.pipeline.s3_helpers import _s3, _bucket
            if s3_key:
                resp = _s3().get_object(Bucket=_bucket(), Key=s3_key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
            elif p and Path(p).exists():
                data = json.loads(Path(p).read_text(encoding="utf-8"))
            else:
                continue
            if dest == "validated":
                validated_data = data
            if not src:
                src = data
        except Exception:
            pass

    if not src:
        src = validated_data

    # ── Layer 1: Decision record from Supabase ────────────────────────────────
    decision_record: dict[str, Any] | None = None
    try:
        from app.db.repositories.decision_record_repository import DecisionRecordRepository
        row = DecisionRecordRepository().get_by_job(job_id)
        decision_record = dict(row) if row else None
    except Exception:
        pass

    # ── Layer 2: LLM call traces from Supabase ───────────────────────────────
    llm_traces: list[dict[str, Any]] = []
    try:
        from app.db.repositories.llm_trace_repository import LLMTraceRepository
        llm_traces = [dict(r) for r in LLMTraceRepository().list_by_job(job_id)]
    except Exception:
        pass

    # ── Agent summary (group traces by agent) ────────────────────────────────
    agents_summary: dict[str, Any] = {}
    for t in llm_traces:
        agent = t.get("agent", "unknown")
        if agent not in agents_summary:
            agents_summary[agent] = {"llm_calls": 0, "models": set(), "decisions": []}
        agents_summary[agent]["llm_calls"] += 1
        if t.get("model"):
            agents_summary[agent]["models"].add(t["model"])
        if t.get("decision_label"):
            agents_summary[agent]["decisions"].append(t["decision_label"])
    # Serialize sets
    for a in agents_summary:
        agents_summary[a]["models"] = list(agents_summary[a]["models"])

    # Seed LLM agents from storyline models when no traces exist (pre-deployment jobs
    # or jobs where trace emission failed). Traces override these defaults if present.
    models_meta = src.get("models") or {}
    dr_models = (decision_record or {}).get("models_used") or {}
    embed_model = models_meta.get("embedding_model") or dr_models.get("embedding") or "nomic-embed-text:v1.5"
    strong_model = models_meta.get("strong_llm_model") or dr_models.get("director")
    if strong_model:
        agents_summary.setdefault("retrieval", {}).update({
            "rule_based": False,
            "description": "Embeds the user prompt and retrieves candidate fragments via cosine similarity.",
            "models": agents_summary.get("retrieval", {}).get("models") or [embed_model],
        })
        agents_summary.setdefault("director", {}).update({
            "rule_based": False,
            "description": "Decides video_mode and section structure from evidence strength.",
            "models": agents_summary.get("director", {}).get("models") or [strong_model],
        })
        agents_summary.setdefault("storyline", {}).update({
            "rule_based": False,
            "description": "Generates the full structured documentary storyline JSON.",
            "models": agents_summary.get("storyline", {}).get("models") or [strong_model],
        })
        agents_summary.setdefault("validation", {}).update({
            "rule_based": False,
            "description": "LLM editorial validation and repair of the timeline.",
            "models": agents_summary.get("validation", {}).get("models") or [strong_model],
        })

    # Enrich with rule-based agents (no LLM calls, so no traces)
    profile = src.get("prompt_profile") or src.get("prompt_intent") or {}
    agents_summary.setdefault("intent", {}).update({
        "rule_based": True,
        "description": "Parses prompt into intent_type, topic_terms, speaker, theme/chapter hints. No LLM.",
        "intent_type": profile.get("intent_type"),
        "scope": profile.get("scope"),
        "topic_terms": profile.get("topic_terms") or [],
        "detected_speaker": profile.get("requested_speaker"),
        "theme_hint": profile.get("theme_hint"),
    })
    bucket = src.get("candidate_bucket_summary") or {}
    agents_summary.setdefault("classification", {}).update({
        "rule_based": True,
        "description": "Scores each candidate fragment via semantic+keyword+theme heuristics. No LLM.",
        "exact": bucket.get("selected", 0),
        "contextual": bucket.get("context", 0),
        "rejected": bucket.get("rejected", 0),
    })
    # Theme Synthesis — _synthesize_documentary_context/tension/evidence are keyword-template
    # functions that return hardcoded strings. Zero LLM involvement.
    story = src.get("storyline") or {}
    agents_summary.setdefault("theme_synthesis", {}).update({
        "rule_based": True,
        "description": "Builds context/tension/evidence narrations via topic keyword templates. No LLM.",
        "topic_label": (profile.get("topic_terms") or [""])[0],
    })
    # Relay Assembler — assigns clip slots from a lookup table after Director decides mode
    director = story.get("director_plan") or {}
    video_mode = (src.get("planning_summary") or {}).get("video_mode") or director.get("video_mode")
    agents_summary.setdefault("relay_assembler", {}).update({
        "rule_based": True,
        "description": "Assigns fragments to relay/section slots based on video_mode lookup table. No LLM.",
        "video_mode": video_mode,
        "section_roles": director.get("recommended_structure") or [],
    })

    # ── Retrieval trace ───────────────────────────────────────────────────────
    ret_diag = src.get("retrieval_diagnostics") or {}
    corpus_diag = src.get("corpus_diagnostics") or {}
    retrieval = {
        "corpus_size": corpus_diag.get("total_embedding_records"),
        "embedding_model": (src.get("models") or {}).get("embedding_model"),
        "search_channels": {
            "semantic": ret_diag.get("semantic_count", 0),
            "keyword": ret_diag.get("keyword_count", 0),
            "chapter": ret_diag.get("chapter_count", 0),
            "speaker": ret_diag.get("speaker_count", 0),
        },
        "merged_candidates": ret_diag.get("merged_count", 0),
    }

    def _video_mode_rationale_inline(mode: str | None, planning: dict[str, Any]) -> str:
        exact = planning.get("exact_clip_count", 0)
        speaker = planning.get("usable_speaker_clip_count", 0)
        rationale_map = {
            "speaker_led_documentary": f"Speaker-led documentary: {exact} exact clips with {speaker} usable speaker soundbites.",
            "relay_of_ideas": f"Relay of ideas: {exact} exact clips from multiple speakers across themes.",
            "chronological_relay": "Chronological relay: clips span multiple years, ordered to show evolution over time.",
            "theme_summary_explainer": f"Theme-summary explainer: {exact} exact + strong related clips, narration-heavy.",
            "narration_led_explainer": f"Narration-led explainer: {exact} exact clips, supplemented by narration.",
            "speaker_led_theme_summary": "Speaker-led theme summary: speaker-constrained prompt with sufficient clips.",
            "low_evidence_brief": f"Low-evidence brief: only {exact} exact clips found — short narration-led format.",
            "low_evidence_speaker_brief": "Low-evidence speaker brief: speaker requested but few usable clips found.",
            "no_evidence": "No usable evidence found for the requested speaker — video could not be generated.",
        }
        return rationale_map.get(mode or "", f"Mode '{mode}' selected based on available evidence.")

    # ── Fragment decisions (per-candidate trace) ──────────────────────────────
    def _candidate_trace(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "fragment_id": c.get("fragment_id"),
            "source_video": c.get("source_video"),
            "speaker": c.get("speaker") or c.get("primary_speaker"),
            "primary_theme": c.get("primary_theme"),
            "primary_chapter": c.get("primary_chapter"),
            "relevance_role": c.get("relevance_role"),
            "relevance_reason": c.get("relevance_reason"),
            "use_as": c.get("use_as"),
            "final_prompt_score": c.get("final_prompt_score") or c.get("prompt_relevance_score"),
            "semantic_score": c.get("semantic_score"),
            "keyword_score": c.get("keyword_score"),
            "quote_strength": c.get("quote_strength"),
            "clarity_score": c.get("clarity_score"),
            "direct_evidence_terms": c.get("direct_evidence_terms") or c.get("meaningful_prompt_overlap") or [],
            "expanded_evidence_terms": c.get("expanded_evidence_terms") or [],
            "speaker_clip_allowed": bool(c.get("speaker_clip_allowed") or c.get("speaker_soundbite_allowed")),
            "final_video_allowed": bool(c.get("final_video_allowed")),
            "start": c.get("start"),
            "end": c.get("end"),
            "duration": c.get("duration"),
            "video_year": c.get("video_year"),
        }

    # ── Timeline trace ────────────────────────────────────────────────────────
    tl_repair = src.get("timeline_repair") or (src.get("storyline") or {}).get("timeline_repair") or {}
    export_tl = src.get("export_timeline") or (src.get("storyline") or {}).get("export_timeline") or []
    timeline = {
        "segment_count": len(export_tl),
        "repair_applied": bool(tl_repair.get("required")),
        "repair_reason": tl_repair.get("reason"),
        "validation_iterations": validated_data.get("repair_iterations", 0),
    }

    # ── Editorial gates ───────────────────────────────────────────────────────
    gates = validated_data.get("editorial_quality_gates") or {}
    editorial_gates = {
        "all_sections_covered": gates.get("all_sections_covered"),
        "all_speaker_labels_present": gates.get("all_speaker_labels_present"),
        "no_instruction_leakage_in_voiceover": gates.get("no_instruction_leakage_in_voiceover"),
        "final_timeline_approved": gates.get("final_timeline_approved"),
        "offending_items": gates.get("offending_items") or {},
    } if gates else None

    return {
        "job_id": job_id,
        "provenance_version": "2.0",
        "generated_at": _utc_now(),
        "decision_record": decision_record,
        "agents": agents_summary,
        "retrieval": retrieval,
        "fragment_decisions": {
            "total_candidates": sum([
                bucket.get("selected", 0),
                bucket.get("supporting", 0),
                bucket.get("context", 0),
                bucket.get("rejected", 0),
            ]),
            "exact": bucket.get("selected", 0),
            "supporting": bucket.get("supporting", 0),
            "contextual": bucket.get("context", 0),
            "rejected": bucket.get("rejected", 0),
            "selected_cap": bucket.get("selected_cap"),
            "cap_reason": bucket.get("cap_reason"),
            "selected": [_candidate_trace(c) for c in (src.get("selected_candidates") or [])],
            "supporting_candidates": [_candidate_trace(c) for c in (src.get("supporting_candidates") or [])],
        },
        "planning": {
            "video_mode": (src.get("planning_summary") or {}).get("video_mode") or (story.get("director_plan") or {}).get("video_mode"),
            "video_mode_rationale": _video_mode_rationale_inline(
                (src.get("planning_summary") or {}).get("video_mode") or (story.get("director_plan") or {}).get("video_mode"),
                src.get("planning_summary") or {},
            ),
            "evidence_strength": (src.get("planning_summary") or {}).get("evidence_strength"),
            "fallback_risk": (src.get("planning_summary") or {}).get("fallback_risk"),
            "planning_notes": (src.get("planning_summary") or {}).get("planning_notes") or (story.get("director_plan") or {}).get("planning_notes") or [],
            "theme_coverage": (story.get("director_plan") or {}).get("theme_coverage") or [],
            "chapter_coverage": (story.get("director_plan") or {}).get("chapter_coverage") or [],
        },
        "llm_call_traces": llm_traces,
        "timeline": timeline,
        "editorial_gates": editorial_gates,
    }


# ---------------------------------------------------------------------------
# Asset details
# ---------------------------------------------------------------------------

@router.get("/{job_id}/assets")
def get_job_assets(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    paths = job.get("paths", {})
    amp = paths.get("asset_manifest_path")

    if not amp or not Path(amp).exists():
        return {"job_id": job_id, "status": "not_available", "message": "Assets not yet generated"}

    try:
        mdata = json.loads(Path(amp).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read asset manifest: {exc}")

    tl = mdata.get("export_timeline") or []
    nb_items = [x for x in tl if (x.get("type") or x.get("segment_type")) == "narration_broll"]

    base = f"/api/video-jobs/{job_id}"

    voiceovers = []
    visuals = []
    for item in nb_items:
        sid = item.get("section_id", "")
        vo_text = item.get("voiceover_text") or item.get("subtitle_text") or ""
        audio_path = item.get("audio_path", "")
        v_path = item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path") or ""
        asset_id_vo = f"voiceover_{sid}" if sid else None
        asset_id_vis = f"visual_{sid}" if sid else None

        if audio_path:
            entry: dict[str, Any] = {
                "section_id": sid,
                "text": vo_text,
                "content_type": "audio/mpeg",
            }
            if asset_id_vo:
                entry["asset_id"] = asset_id_vo
                entry["audio_url"] = f"{base}/assets/{asset_id_vo}/stream"
                entry["download_url"] = f"{base}/assets/{asset_id_vo}/download"
            voiceovers.append(entry)

        if v_path:
            ventry: dict[str, Any] = {
                "section_id": sid,
                "visual_title": item.get("visual_title", ""),
                "visual_asset_type": item.get("visual_asset_type", ""),
                "visual_keywords": item.get("visual_keywords", []),
                "status": "generated" if Path(v_path).exists() else "missing",
                "content_type": "image/png",
            }
            if asset_id_vis:
                ventry["asset_id"] = asset_id_vis
                ventry["visual_url"] = f"{base}/assets/{asset_id_vis}/stream"
                ventry["download_url"] = f"{base}/assets/{asset_id_vis}/download"
            visuals.append(ventry)

    cards_data = mdata.get("cards") or {}
    cards = []
    if cards_data.get("title_card"):
        cards.append({
            "asset_id": "title_card",
            "type": "title_card",
            "image_url": f"{base}/assets/title_card/stream",
            "download_url": f"{base}/assets/title_card/download",
        })
    if cards_data.get("closing_card"):
        cards.append({
            "asset_id": "closing_card",
            "type": "closing_card",
            "image_url": f"{base}/assets/closing_card/stream",
            "download_url": f"{base}/assets/closing_card/download",
        })
    for sc in (cards_data.get("section_cards") or []):
        if isinstance(sc, dict):
            sc_id = sc.get("section_id", "")
            oid = f"section_card_{sc_id}" if sc_id else "section_card"
            cards.append({
                "asset_id": oid,
                "type": "section_card",
                "section_id": sc_id,
                "image_url": f"{base}/assets/{oid}/stream",
                "download_url": f"{base}/assets/{oid}/download",
            })

    vqg = mdata.get("visual_quality_gates") or {}

    return {
        "job_id": job_id,
        "status": "completed",
        "voiceovers": voiceovers,
        "visuals": visuals,
        "cards": cards,
        "visual_quality_gates": vqg,
        "visual_strategy": mdata.get("visual_strategy", ""),
    }


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@router.get("/{job_id}/result")
def get_job_result(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    if job.get("status") not in ("completed", "failed"):
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "message": "Job not yet complete",
            "progress_percent": job.get("progress_percent", 0),
        }

    paths = job.get("paths", {})
    vp = paths.get("video_path")
    emp = paths.get("export_manifest_path")
    srtp = paths.get("srt_path")
    vttp = paths.get("vtt_path")

    base = f"/api/video-jobs/{job_id}"

    # Check asset availability via registry (works for both local and S3)
    try:
        from app.services.storage_service import get_storage_service
        storage = get_storage_service()
        # Use S3-aware registry loader if available, else fall back to local
        if hasattr(storage, "_load_registry"):
            reg = storage._load_registry(job_id)
        else:
            from app.services.storage_service import load_registry
            reg = load_registry(job_id)
        reg_objects = reg.get("objects", {})
        def _asset_available(oid: str) -> bool:
            if oid not in reg_objects:
                return False
            try:
                obj = storage.resolve(job_id, oid)
                return storage.exists(obj)
            except Exception:
                return False
        has_video = _asset_available("video")
        has_srt = _asset_available("srt")
        has_vtt = _asset_available("vtt")
        has_manifest = _asset_available("manifest")
    except Exception:
        # Fall back to local path checks
        has_video = bool(vp and Path(vp).exists())
        has_srt = bool(srtp and Path(srtp).exists())
        has_vtt = bool(vttp and Path(vttp).exists())
        has_manifest = bool(emp and Path(emp).exists())

    # Live HLS subtitle check during rendering/streaming
    from app.core.paths import EXPORTS_DIR
    from app.services.storage_service import STORAGE_BACKEND
    _hls_subs_local = EXPORTS_DIR / "hls" / job_id / "subs.vtt"
    has_live_subs = _hls_subs_local.exists() if (job_id and STORAGE_BACKEND != "s3") else (job.get("status") in ("running", "completed"))
    if has_live_subs:
        has_vtt = True
        has_srt = True

    # Parse export manifest
    summary: dict[str, Any] = {}
    duration_check: dict[str, Any] = {}
    quality_checks: dict[str, Any] = {}
    title = ""
    warnings: list[str] = job.get("warnings", [])

    if has_manifest:
        try:
            from app.services.storage_service import get_storage_service as _gss
            _storage = _gss()
            _obj = _storage.resolve(job_id, "manifest")
            _local = _storage.get_local_path(_obj)
            if _local and _local.exists():
                emdata = json.loads(_local.read_text(encoding="utf-8"))
            else:
                emdata = json.loads(b"".join(_storage.open_stream(_obj)).decode("utf-8"))
            qc = emdata.get("quality_checks") or {}
            summary = {
                "speaker_clips": qc.get("speaker_clip_count", 0),
                "ai_voiceovers": qc.get("ai_voiceover_segments", 0),
                "generated_visuals": emdata.get("generated_visual_segments", 0),
                "muted_archive_narration": emdata.get("muted_archive_narration_segments", 0),
                "subtitle_segments": emdata.get("subtitles", {}).get("entry_count", 0),
            }
            sub_cov = emdata.get("subtitles", {}).get("coverage") or {}
            quality_checks = {
                "generated_visual_segments": emdata.get("generated_visual_segments", 0),
                "muted_archive_narration_segments": emdata.get("muted_archive_narration_segments", 0),
                "voiceover_segments_subtitled": sub_cov.get("voiceover_segments_subtitled", 0),
                "speaker_segments_subtitled": sub_cov.get("speaker_segments_subtitled", 0),
                "segments_without_subtitles": sub_cov.get("segments_without_subtitles", 0),
            }
            rhythm = emdata.get("rhythm") or {}
            duration_check = {
                "format_duration": rhythm.get("format_duration"),
                "video_stream_duration": rhythm.get("video_stream_duration"),
                "audio_stream_duration": rhythm.get("audio_stream_duration"),
                "duration_delta": rhythm.get("duration_delta"),
                "final_duration_seconds": rhythm.get("final_duration_seconds"),
            }
        except Exception:
            pass

    if not title and vp:
        title = Path(vp).stem

    subtitles: dict[str, str | None] = {
        "srt": f"{base}/subtitles/srt" if has_srt else None,
        "vtt": f"{base}/subtitles/vtt" if has_vtt else None,
    }

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "title": title,
        # Stable API URLs — use these in the frontend
        "video_url": f"{base}/video/stream" if has_video else None,
        "video_stream_url": f"{base}/video/stream" if has_video else None,
        "video_download_url": f"{base}/video/download" if has_video else None,
        "subtitles": subtitles,
        "manifest_url": f"{base}/manifest" if has_manifest else None,
        "assets_url": f"{base}/assets",
        "media_url": f"{base}/media",
        "thumbnail_url": f"{base}/thumbnail",
        "summary": summary,
        "duration_check": duration_check,
        "quality_checks": quality_checks,
        "warnings": warnings,
        "error": job.get("error"),
    }
