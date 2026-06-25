"""
YouTube channel ingestion API.

Endpoints:
  POST /api/ingest/sync                    — trigger a full channel sync
  POST /api/ingest/videos/{video_id}/sync  — re-ingest a single video
  GET  /api/ingest/jobs                    — list ingest jobs
  GET  /api/ingest/jobs/{job_id}           — get job status
  GET  /api/ingest/jobs/{job_id}/events    — SSE stream of job events
  GET  /api/ingest/videos                  — inventory of all known videos + ingest status
  GET  /api/ingest/videos/{video_id}       — single video full stage breakdown
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.ingest_job_runner import (
    create_ingest_job,
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
def trigger_sync(body: SyncRequest | None = None) -> dict[str, Any]:
    """
    Trigger a full YouTube channel sync.

    Returns immediately with a job_id. Use GET /ingest/jobs/{job_id} or
    GET /ingest/jobs/{job_id}/events to track progress.

    Only one sync job can run at a time. Returns 409 if one is already active.
    """
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
def trigger_single_video_sync(video_id: str) -> dict[str, Any]:
    """
    Re-trigger ingestion for a specific video_id (e.g. to retry after failure).

    This creates a new ingest job that will only process the given video.
    It bypasses the discovery stage and goes straight to download/process.
    """
    from app.core.paths import RAW_VIDEO_DIR
    from app.services.ingest_store import upsert_video_stage

    # Check if the video exists on disk already — if so, skip download
    mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{video_id}_*.mp4"))
    already_downloaded = len(mp4_files) > 0

    if not already_downloaded:
        # Full re-ingest: create a standard job but limit to this video
        job_id = create_ingest_job(video_limit=1)
        # Mark the target video so the runner knows what to focus on
        # (the discover stage will still run but should find exactly this video as new)
    else:
        # Video is already downloaded — re-run from extract_audio onwards
        job_id = create_ingest_job(video_limit=1)
        # Reset stages so runner picks it up again
        for stage in ["extract_audio", "transcribe", "create_fragments", "classify", "embed"]:
            upsert_video_stage(video_id, stage, "pending", job_id=job_id)

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
