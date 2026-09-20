"""
YouTube channel ingestion API.

Endpoints:
  POST /api/ingest/sync                           — trigger a full channel sync
  POST /api/ingest/videos/{video_id}/sync         — re-ingest a single video
  POST /api/ingest/jobs/{job_id}/resume           — retry a failed job's failed videos from where it died
  GET  /api/ingest/jobs                           — list ingest jobs
  GET  /api/ingest/jobs/{job_id}                  — get job status
  GET  /api/ingest/jobs/{job_id}/events           — SSE stream of job events
  GET  /api/ingest/jobs/{job_id}/explainability   — structured stage-by-stage reasoning trace
  GET  /api/ingest/jobs/{job_id}/logs             — list available stage log files
  GET  /api/ingest/jobs/{job_id}/logs/{stage}     — full log text for a stage (stdout + stderr)
  GET  /api/ingest/jobs/{job_id}/logs/{stage}/stream — SSE tail of a live stage log
  GET  /api/ingest/videos                         — inventory of all known videos + ingest status
  GET  /api/ingest/videos/{video_id}              — single video full stage breakdown
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, require_role
from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket
from app.services.ingest_job_runner import (
    create_ingest_job,
    create_resume_job,
    resolve_resume_candidates,
    start_ingest_job_background,
)
from app.services.ingest_store import (
    load_ingest_job,
    list_ingest_jobs,
    load_ingest_events_after,
    get_video_stages,
    get_videos_stages_bulk,
    list_videos_by_stage_status,
)
from app.services.archive_service import list_videos as list_archive_videos

logger = logging.getLogger("ai4ia.api.ingest")

router = APIRouter(prefix="/ingest", tags=["Ingest"])

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SyncRequest(BaseModel):
    channel_url: str | None = Field(
        None,
        description="Override the default channel URL from settings",
    )
    video_limit: int | None = Field(
        None,
        ge=1,
        le=500,
        description="Limit the number of NEW videos to download in this sync",
    )


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str | None
    channel_url: str | None
    video_limit: int | None
    current_stage: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    summary: dict[str, Any] | None
    error: dict[str, Any] | None


class VideoIngestStatus(BaseModel):
    video_id: str
    title: str | None
    overall_status: str  # pending | partial | completed | failed
    stages: dict[str, dict[str, Any]]  # stage -> {status, error, metadata, updated_at}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 3 sync/re-sync triggers per admin per 5 minutes — the pipeline itself is
# also serialized to one job at a time (see _is_job_running), but that only
# limits concurrency, not request rate.
_trigger_limiter = TokenBucket(rate=3, per=300.0, name="ingest_trigger")


def _is_job_running() -> str | None:
    """Return job_id of any currently running/queued ingest job, or None."""
    jobs = list_ingest_jobs(limit=10)
    for j in jobs:
        if j.get("status") in ("queued", "running"):
            return j.get("job_id")
    return None


def _build_job_response(job: dict[str, Any]) -> IngestJobResponse:
    return IngestJobResponse(
        job_id=job.get("job_id", ""),
        status=job.get("status", "unknown"),
        progress=int(job.get("progress", 0)),
        message=job.get("message"),
        channel_url=job.get("channel_url"),
        video_limit=job.get("video_limit"),
        current_stage=job.get("current_stage"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        summary=job.get("summary"),
        error=job.get("error"),
    )


def _derive_overall_status(stages: dict[str, dict[str, Any]]) -> str:
    """
    Derive an overall ingest status for a video from its per-stage rows.
    - completed: embed stage is completed
    - failed: any stage is failed
    - partial: at least one stage completed but not all
    - pending: no stages started yet
    """
    if not stages:
        return "pending"

    statuses = {s: d.get("status", "pending") for s, d in stages.items()}

    if statuses.get("embed") == "completed":
        return "completed"
    if any(v == "failed" for v in statuses.values()):
        return "failed"
    if any(v == "completed" for v in statuses.values()):
        return "partial"
    if any(v == "running" for v in statuses.values()):
        return "running"
    return "pending"


# ---------------------------------------------------------------------------
# POST /ingest/sync — trigger full channel sync
# ---------------------------------------------------------------------------

@router.post("/sync", status_code=202)
def trigger_sync(
    body: SyncRequest | None = None,
    current_user: CurrentUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """
    Trigger a full YouTube channel sync.

    Returns immediately with a job_id. Use GET /ingest/jobs/{job_id} or
    GET /ingest/jobs/{job_id}/events to track progress.

    Only one sync job can run at a time. Returns 409 if one is already active.
    """
    if not _trigger_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="ingest_trigger").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many ingest triggers. Please wait before starting another sync.",
        )

    req = body or SyncRequest()

    running_id = _is_job_running()
    if running_id:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "A sync job is already active",
                "active_job_id": running_id,
            },
        )

    job_id = create_ingest_job(
        channel_url=req.channel_url,
        video_limit=req.video_limit,
    )
    start_ingest_job_background(job_id)

    logger.info("Ingest sync triggered job_id=%s", job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Channel sync job queued",
    }


# ---------------------------------------------------------------------------
# POST /ingest/videos/{video_id}/sync — re-ingest a single video
# ---------------------------------------------------------------------------

@router.post("/videos/{video_id}/sync", status_code=202)
def trigger_single_video_sync(
    video_id: str,
    current_user: CurrentUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """
    Re-trigger ingestion for a specific video_id (e.g. to retry after failure).

    This creates a new ingest job that will only process the given video.
    It bypasses the discovery stage and goes straight to download/process.
    """
    if not _trigger_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="ingest_trigger").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many ingest triggers. Please wait before starting another sync.",
        )

    from app.core.paths import RAW_VIDEO_DIR
    from app.core.settings import settings

    # Check if the video exists — S3 or local depending on backend
    already_downloaded = False
    if settings.ai4ia_storage_backend == "s3":
        try:
            from app.services.ingest_job_runner import _s3_list_keys
            s3_stems = _s3_list_keys("raw/youtube/ai4ia_channel/")
            already_downloaded = any(video_id in s for s in s3_stems)
        except Exception:
            already_downloaded = False
    else:
        mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{video_id}_*.mp4"))
        already_downloaded = len(mp4_files) > 0

    if not already_downloaded:
        # Not downloaded yet, so there is nothing to resume from — a normal
        # sync (scoped to at most 1 new video) is the only correct path,
        # since discover/download aren't part of the resumable stage set.
        job_id = create_ingest_job(video_limit=1)
    else:
        # Previously this reset 5 ingest_status rows to "pending" and then
        # started a plain job, which begins at discover -- discover only
        # finds videos not yet in yt-dlp's archive, so an already-downloaded
        # video is never "new" and the pending rows were never read by
        # anything. Route through the same resume engine as POST
        # /jobs/{id}/resume instead: start straight at extract_audio for
        # exactly this video.
        job_id = create_resume_job(resume_from_stage="extract_audio", resume_video_ids=[video_id])

    start_ingest_job_background(job_id)

    logger.info("Single video sync triggered video_id=%s job_id=%s", video_id, job_id)
    return {
        "job_id": job_id,
        "video_id": video_id,
        "status": "queued",
        "already_downloaded": already_downloaded,
        "message": "Video re-ingest job queued",
    }


# ---------------------------------------------------------------------------
# POST /ingest/jobs/{job_id}/resume — retry a failed job from where it died
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_ingest_job(
    job_id: str,
    current_user: CurrentUser = Depends(require_role("admin")),
) -> dict[str, Any]:
    """
    Retry a failed ingest job's failed videos from the stage they failed at,
    instead of a plain POST /sync (which starts at discover and never sees
    an already-downloaded video again).

    Only jobs that failed at extract_audio, transcribe, create_fragments,
    classify or embed are resumable — a discover or download failure means
    the videos were never fully downloaded, so a plain sync already retries
    them correctly on its own.
    """
    if not _trigger_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="ingest_trigger").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many ingest triggers. Please wait before starting another sync.",
        )

    try:
        job = load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    if job.get("status") != "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Only a failed job can be resumed (status is '{job.get('status')}')",
        )

    try:
        from_stage, video_ids = resolve_resume_candidates(job)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    running_id = _is_job_running()
    if running_id:
        raise HTTPException(
            status_code=409,
            detail={"message": "A sync job is already active", "active_job_id": running_id},
        )

    new_job_id = create_resume_job(
        resume_from_stage=from_stage,
        resume_video_ids=video_ids,
        channel_url=job.get("channel_url"),
        resumed_from_job_id=job_id,
    )
    start_ingest_job_background(new_job_id)

    logger.info("Ingest job resumed original=%s new=%s from_stage=%s videos=%d",
               job_id, new_job_id, from_stage, len(video_ids))
    return {
        "job_id": new_job_id,
        "resumed_from_job_id": job_id,
        "from_stage": from_stage,
        "video_count": len(video_ids),
        "status": "queued",
        "message": f"Resuming {len(video_ids)} video(s) from '{from_stage}'",
    }


# ---------------------------------------------------------------------------
# GET /ingest/jobs — list jobs
# ---------------------------------------------------------------------------

@router.get("/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    jobs = list_ingest_jobs(limit=limit)
    return {
        "count": len(jobs),
        "jobs": [_build_job_response(j) for j in jobs],
    }


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id} — job status
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> IngestJobResponse:
    try:
        job = load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")
    return _build_job_response(job)


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/events — SSE stream
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/events")
async def stream_job_events(
    job_id: str,
    after: int = Query(0, ge=0, description="Cursor: return events with id > after"),
) -> StreamingResponse:
    """
    Server-Sent Events stream of ingest job progress.

    Connect once; the stream closes automatically when the job reaches a
    terminal state (completed / failed / cancelled).

    Use the `after` query param with the last received event `id` for reconnect.
    """
    try:
        load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    async def _sse_generator():
        cursor = after
        terminal_types = {"completed", "failed", "cancelled"}
        poll_interval = 1.5  # seconds
        max_idle_polls = 200  # ~5 minutes of silence before giving up

        idle = 0
        while idle < max_idle_polls:
            try:
                events = await asyncio.to_thread(
                    load_ingest_events_after, job_id, cursor
                )
            except Exception:
                logger.warning("SSE event fetch failed for job %s", job_id, exc_info=True)
                await asyncio.sleep(poll_interval)
                continue

            for event in events:
                event_id = event.get("id", 0)
                event_type = event.get("type", "progress")
                data = event.get("data") or {}

                yield f"id: {event_id}\n"
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data)}\n\n"

                if event_id:
                    cursor = event_id
                idle = 0

                if event_type in terminal_types:
                    return

            if not events:
                idle += 1

            await asyncio.sleep(poll_interval)

        # Timed out — send a keepalive close
        yield "event: timeout\ndata: {}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/explainability
# ---------------------------------------------------------------------------

# Ordered pipeline stages with what each one does and what tool it uses
_STAGE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "discover": {
        "label": "Discover new channel videos",
        "tool": "yt-dlp --flat-playlist",
        "how": "Fetches the full channel playlist via yt-dlp, then diffs against the local download archive (data/metadata/youtube/download_archive.txt) and existing MP4 files (or S3 keys in cloud mode) to produce only net-new video IDs.",
        "output": "List of {video_id, title, index} dicts for videos not yet downloaded",
    },
    "download": {
        "label": "Download videos",
        "tool": "scripts/01_download_videos.py (yt-dlp)",
        "how": "Downloads each new video as MP4 with yt-dlp, using the download archive for deduplication. In S3 mode, files are uploaded and deleted locally immediately after download.",
        "output": "MP4 files in data/raw_videos/ (or S3 raw/youtube/ai4ia_channel/)",
    },
    "extract_audio": {
        "label": "Extract audio",
        "tool": "scripts/02_extract_audio.py (ffmpeg)",
        "how": "Runs ffmpeg to extract 16kHz mono WAV from each MP4. Verifies output WAV exists before marking the stage complete.",
        "output": "WAV files in data/audio/",
    },
    "transcribe": {
        "label": "Transcribe audio (Whisper)",
        "tool": "scripts/03_transcribe.py (OpenAI Whisper)",
        "how": "Runs Whisper on each WAV file to produce word-level JSON transcripts with speaker timestamps.",
        "output": "JSON transcript files in data/transcripts/",
    },
    "create_fragments": {
        "label": "Create speaker fragments",
        "tool": "scripts/04_create_fragments.py",
        "how": "Splits each transcript into semantically coherent speaker fragments (typically 30–120 seconds) using pause detection and topic-shift heuristics. Each fragment gets a stable fragment_id.",
        "output": "*.fragments.json files in data/fragments/",
    },
    "classify": {
        "label": "Classify fragments",
        "tool": "scripts/05_classify_fragments.py (openai/gpt-oss-120b)",
        "how": "Sends each fragment to openai/gpt-oss-120b for classification: assigns primary_theme, primary_chapter, speaker, quote_strength, clarity_score, and clip_usage_candidates.",
        "output": "*.classified.json files in data/fragments_classified/",
    },
    "embed": {
        "label": "Generate & import embeddings",
        "tool": "scripts/06_generate_embeddings.py (nomic-embed-text:v1.5) + import_embeddings_to_chroma.py",
        "how": "Generates 768-dim embeddings for each classified fragment using nomic-embed-text:v1.5 via Ollama, writes fragment_embedding_matrix.npy + classified_fragments_embeddings.jsonl, then imports into ChromaDB for retrieval.",
        "output": "data/embeddings/ (npy + jsonl) + ChromaDB collection 'fragments'",
    },
}

_STAGE_ORDER = ["discover", "download", "extract_audio", "transcribe", "create_fragments", "classify", "embed"]


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/provenance — unified two-layer provenance
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/provenance")
def get_ingest_provenance(job_id: str) -> dict[str, Any]:
    """Full two-layer provenance for an ingest job.

    Layer 1 — decision_record: osi-1.0 record (agents, models, fragment counts).
    Layer 2 — llm_call_traces: per-stage AI model call rows (transcription, embedding).

    Also includes the data funnel and per-stage trace from /explainability.
    The old /explainability endpoint is preserved for backwards compatibility.

    AI agents in the ingest pipeline:
      transcription — faster-whisper (speech-to-text, AI model)
      classification — rule-based keyword taxonomy (no AI)
      embedding     — nomic-embed-text:v1.5 (vector embedding, AI model)
    """
    try:
        job = load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    stage_status: dict[str, Any] = job.get("stage_status") or {}
    summary: dict[str, Any] = job.get("summary") or {}

    # ── Layer 1: Decision record ──────────────────────────────────────────────
    decision_record = None
    try:
        from app.db.repositories.decision_record_repository import DecisionRecordRepository
        row = DecisionRecordRepository().get_by_job(job_id)
        decision_record = dict(row) if row else None
    except Exception:
        pass

    # ── Layer 2: LLM/model call traces ───────────────────────────────────────
    llm_traces = []
    try:
        from app.db.repositories.llm_trace_repository import LLMTraceRepository
        llm_traces = [dict(r) for r in LLMTraceRepository().list_by_job(job_id)]
    except Exception:
        pass

    # ── Agent summary ─────────────────────────────────────────────────────────
    agents = {
        "transcription": {
            "rule_based": False,
            "model": "faster-whisper",
            "call_type": "speech_to_text",
            "description": "Converts audio to timestamped transcript using Whisper AI model.",
            "videos_processed": summary.get("transcribed", 0),
            "llm_traces": [t for t in llm_traces if t.get("agent") == "transcription"],
        },
        "classification": {
            "rule_based": True,
            "description": "Assigns primary_theme, chapter, and speaker_type via keyword taxonomy. No AI model.",
            "fragments_classified": summary.get("classified", 0),
        },
        "embedding": {
            "rule_based": False,
            "model": "nomic-embed-text:v1.5",
            "call_type": "embed",
            "description": "Generates 768-dim vectors for each fragment and imports into ChromaDB.",
            "fragments_embedded": summary.get("embedded", 0),
            "llm_traces": [t for t in llm_traces if t.get("agent") == "embedding"],
        },
    }

    # ── Data funnel ───────────────────────────────────────────────────────────
    funnel = {
        "videos_discovered": summary.get("new_videos_found", 0),
        "downloaded": summary.get("downloaded", 0),
        "audio_extracted": summary.get("audio_extracted", 0),
        "transcribed": summary.get("transcribed", 0),
        "fragmented": summary.get("fragmented", 0),
        "classified": summary.get("classified", 0),
        "embedded": summary.get("embedded", 0),
    }

    # ── Per-stage trace (reuse from explainability) ───────────────────────────
    stages_trace = []
    for stage in _STAGE_ORDER:
        s = stage_status.get(stage) or {}
        desc = _STAGE_DESCRIPTIONS.get(stage, {})
        stages_trace.append({
            "stage": stage,
            "label": desc.get("label", stage),
            "status": s.get("status", "pending"),
            "progress_percent": s.get("progress", 0),
            "message": s.get("message", ""),
            "tool": desc.get("tool"),
        })

    return {
        "job_id": job_id,
        "provenance_version": "2.0",
        "status": job.get("status"),
        "channel_url": job.get("channel_url"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "decision_record": decision_record,
        "agents": agents,
        "data_funnel": funnel,
        "llm_call_traces": llm_traces,
        "stages": stages_trace,
    }


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/logs — list available stage log files
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/logs")
def list_job_logs(job_id: str) -> dict[str, Any]:
    """List all available stage log files for an ingest job."""
    try:
        load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    from app.core.paths import DATA_DIR as _DATA_DIR
    log_dir = _DATA_DIR / "jobs" / job_id / "logs"

    available = []
    for stage in _STAGE_ORDER:
        stdout_path = log_dir / f"{stage}.stdout.log"
        stderr_path = log_dir / f"{stage}.stderr.log"
        stdout_exists = stdout_path.exists()
        stderr_exists = stderr_path.exists()
        if stdout_exists or stderr_exists:
            available.append({
                "stage": stage,
                "label": _STAGE_DESCRIPTIONS.get(stage, {}).get("label", stage),
                "stdout_bytes": stdout_path.stat().st_size if stdout_exists else 0,
                "stderr_bytes": stderr_path.stat().st_size if stderr_exists else 0,
                "log_url": f"/api/ingest/jobs/{job_id}/logs/{stage}",
                "stream_url": f"/api/ingest/jobs/{job_id}/logs/{stage}/stream",
            })

    return {
        "job_id": job_id,
        "log_dir": str(log_dir),
        "stages_with_logs": available,
    }


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/logs/{stage} — full log text
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/logs/{stage}")
def get_stage_log(
    job_id: str,
    stage: str,
    stream_type: str = Query("both", description="stdout | stderr | both"),
) -> PlainTextResponse:
    """Return the full log output for a stage as plain text.

    Use ?stream_type=stdout, stderr, or both (default).
    """
    try:
        load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    from app.core.paths import DATA_DIR as _DATA_DIR
    log_dir = _DATA_DIR / "jobs" / job_id / "logs"

    parts: list[str] = []

    if stream_type in ("stdout", "both"):
        p = log_dir / f"{stage}.stdout.log"
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            if stream_type == "both" and text:
                parts.append(f"=== STDOUT ===\n{text}")
            else:
                parts.append(text)

    if stream_type in ("stderr", "both"):
        p = log_dir / f"{stage}.stderr.log"
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            if stream_type == "both" and text:
                parts.append(f"=== STDERR ===\n{text}")
            else:
                parts.append(text)

    if not parts:
        raise HTTPException(
            status_code=404,
            detail=f"No logs found for stage '{stage}' in job '{job_id}'. "
                   "The stage may not have started yet.",
        )

    return PlainTextResponse("\n".join(parts))


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}/logs/{stage}/stream — SSE live tail
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/logs/{stage}/stream")
async def stream_stage_log(
    job_id: str,
    stage: str,
    stream_type: str = Query("stdout", description="stdout | stderr"),
) -> StreamingResponse:
    """SSE stream that tails a stage log file in real time.

    Emits 'log_line' events while the stage is running, then a 'done' event
    when the stage reaches a terminal state (completed/failed) and no new
    lines appear. Reconnect-safe: use ?after_byte=N to resume from offset.

    The log file grows line-by-line because _run_script uses Popen with
    live writing — so this stream delivers output as the subprocess produces it.
    """
    try:
        load_ingest_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Ingest job not found: {job_id}")

    from app.core.paths import DATA_DIR as _DATA_DIR
    log_dir = _DATA_DIR / "jobs" / job_id / "logs"
    log_path = log_dir / f"{stage}.{stream_type}.log"

    async def _generator():
        yield "retry: 1000\n\n"

        offset = 0
        idle_ticks = 0
        max_idle = 120  # 120 × 0.5s = 60s of silence after stage ends

        while idle_ticks < max_idle:
            # Read new content from the log file
            new_lines: list[str] = []
            try:
                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    new_chunk = text[offset:]
                    if new_chunk:
                        offset = len(text)
                        new_lines = new_chunk.splitlines()
            except Exception:
                pass

            for line in new_lines:
                payload = json.dumps({"stage": stage, "stream": stream_type, "line": line})
                yield f"event: log_line\ndata: {payload}\n\n"

            if new_lines:
                idle_ticks = 0
            else:
                idle_ticks += 1

            # Check if stage is terminal and no more new lines
            if idle_ticks >= 4:  # 2s of silence — check stage status
                try:
                    job = load_ingest_job(job_id)
                    stage_st = (job.get("stage_status") or {}).get(stage, {}).get("status")
                    job_status = job.get("status")
                    is_terminal = (
                        stage_st in ("completed", "failed")
                        or job_status in ("completed", "failed", "cancelled")
                    )
                    if is_terminal and idle_ticks >= 6:
                        yield f"event: done\ndata: {json.dumps({'stage': stage, 'status': stage_st or job_status})}\n\n"
                        return
                except Exception:
                    pass

            await asyncio.sleep(0.5)

        yield f"event: timeout\ndata: {json.dumps({'stage': stage})}\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# GET /ingest/videos — inventory with ingest status
# ---------------------------------------------------------------------------

@router.get("/videos")
def list_ingest_videos(
    status: str | None = Query(
        None,
        description="Filter by overall status: pending | partial | completed | failed | running",
    ),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """
    List all known channel videos with their per-stage ingest status.

    Combines the archive service's file-system scan with ingest_status rows
    from Supabase to give a unified inventory view.
    """
    # Get all videos from disk (archive service)
    archive_videos = list_archive_videos()
    video_ids = [v["video_id"] for v in archive_videos if v.get("video_id")]

    # Bulk-fetch all ingest_status rows for these videos
    all_stage_rows = get_videos_stages_bulk(video_ids)

    # Group stage rows by video_id
    stages_by_video: dict[str, dict[str, dict[str, Any]]] = {}
    for row in all_stage_rows:
        vid = row.get("video_id", "")
        stage = row.get("stage", "")
        stages_by_video.setdefault(vid, {})[stage] = {
            "status": row.get("status", "pending"),
            "error": row.get("error"),
            "metadata": row.get("metadata", {}),
            "updated_at": row.get("updated_at"),
        }

    # Build response
    result: list[dict[str, Any]] = []
    for video in archive_videos:
        vid = video.get("video_id", "")
        stages = stages_by_video.get(vid, {})
        overall = _derive_overall_status(stages)

        if status and overall != status:
            continue

        result.append({
            "video_id": vid,
            "title": video.get("title"),
            "duration": video.get("duration"),
            "webpage_url": video.get("webpage_url"),
            "overall_status": overall,
            "stages": stages,
        })

        if len(result) >= limit:
            break

    return {
        "count": len(result),
        "total_on_disk": len(archive_videos),
        "videos": result,
    }


# ---------------------------------------------------------------------------
# GET /ingest/videos/{video_id} — single video full stage breakdown
# ---------------------------------------------------------------------------

@router.get("/videos/{video_id}")
def get_video_ingest_status(video_id: str) -> VideoIngestStatus:
    """
    Full per-stage ingest status for a single video.
    """
    stage_rows = get_video_stages(video_id)

    # Also check the archive service for basic metadata
    archive_videos = list_archive_videos()
    archive_meta = next((v for v in archive_videos if v.get("video_id") == video_id), None)

    stages: dict[str, dict[str, Any]] = {}
    for row in stage_rows:
        stage = row.get("stage", "")
        stages[stage] = {
            "status": row.get("status", "pending"),
            "error": row.get("error"),
            "metadata": row.get("metadata", {}),
            "updated_at": row.get("updated_at"),
            "job_id": row.get("job_id"),
        }

    overall = _derive_overall_status(stages)

    if not stage_rows and not archive_meta:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    return VideoIngestStatus(
        video_id=video_id,
        title=archive_meta.get("title") if archive_meta else None,
        overall_status=overall,
        stages=stages,
    )
