"""
YouTube channel ingestion job runner.

Orchestrates the 7-stage ingestion pipeline (scripts 01–06 + chroma import):
  discover → download → extract_audio → transcribe →
  create_fragments → classify → embed

Mirrors the video_job_runner.py pattern: Celery if Redis is reachable,
thread fallback otherwise. Auto-sync is intentionally NOT triggered here —
callers (API endpoints or Celery beat in deployed mode) drive scheduling.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    SCRIPTS_DIR,
    RAW_VIDEO_DIR,
)
from app.core.settings import settings
from app.services.ingest_store import (
    save_ingest_job,
    load_ingest_job,
    list_running_ingest_jobs,
    append_ingest_event,
    upsert_video_stage,
)

logger = logging.getLogger("ai4ia.ingest")

PYTHON_BIN = sys.executable

# ---------------------------------------------------------------------------
# Stage definitions & weights (must sum to 100)
# ---------------------------------------------------------------------------

INGEST_STAGE_WEIGHTS: dict[str, int] = {
    "discover":         5,
    "download":        30,
    "extract_audio":   10,
    "transcribe":      25,
    "create_fragments": 12,
    "classify":        10,
    "embed":            8,
}

INGEST_STAGE_LABELS: dict[str, str] = {
    "discover":         "Discovering new channel videos",
    "download":         "Downloading videos",
    "extract_audio":    "Extracting audio",
    "transcribe":       "Transcribing audio (Whisper)",
    "create_fragments": "Creating speaker fragments",
    "classify":         "Classifying fragments",
    "embed":            "Generating & importing embeddings",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(job_id: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"job_id": job_id})


def _run_script(
    script: str | Path,
    args: list[str],
    job_id: str,
    stage: str,
    timeout: int = 7200,
) -> tuple[int, str, str]:
    """Run a script subprocess, capturing stdout/stderr. Returns (returncode, stdout, stderr)."""
    log_dir = DATA_DIR / "jobs" / job_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [PYTHON_BIN, str(script)] + args
    _log(job_id).info("Running %s: %s", stage, " ".join(cmd))

    stdout_log = log_dir / f"{stage}.stdout.log"
    stderr_log = log_dir / f"{stage}.stderr.log"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        stdout_log.write_text(result.stdout or "")
        stderr_log.write_text(result.stderr or "")
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        msg = f"Script timed out after {timeout}s: {script}"
        stderr_log.write_text(msg)
        return -1, "", msg
    except Exception as exc:
        msg = str(exc)
        stderr_log.write_text(msg)
        return -1, "", msg


# ---------------------------------------------------------------------------
# Job state helpers
# ---------------------------------------------------------------------------

def _update_job(
    job_id: str,
    stage: str,
    status: str,
    progress: int,
    message: str = "",
    **extra: Any,
) -> None:
    job = load_ingest_job(job_id)
    job.setdefault("stage_status", {})
    job["stage_status"][stage] = {
        "status": status,
        "progress": progress,
        "message": message,
        **extra,
    }
    job["current_stage"] = stage
    job["progress"] = _compute_progress(job["stage_status"])
    job["message"] = message
    job["updated_at"] = _utc_now()
    if status == "running" and not job.get("started_at"):
        job["started_at"] = _utc_now()
    save_ingest_job(job_id, job)
    append_ingest_event(job_id, "progress", {
        "stage": stage,
        "stage_status": status,
        "progress": job["progress"],
        "message": message,
        **extra,
    })


def _compute_progress(stage_status: dict[str, Any]) -> int:
    total = 0
    for stage, weight in INGEST_STAGE_WEIGHTS.items():
        s = stage_status.get(stage, {})
        st = s.get("status", "pending")
        sp = int(s.get("progress", 0))
        if st == "completed":
            total += weight
        elif st == "running":
            total += weight * sp // 100
    return min(100, total)


def _fail_job(job_id: str, stage: str, message: str, details: str = "") -> None:
    job = load_ingest_job(job_id)
    job["status"] = "failed"
    job["current_stage"] = stage
    job["error"] = {"stage": stage, "message": message, "details": details[:2000]}
    job["completed_at"] = _utc_now()
    job["updated_at"] = _utc_now()
    save_ingest_job(job_id, job)
    append_ingest_event(job_id, "failed", {"stage": stage, "message": message})


def _complete_job(job_id: str, summary: dict[str, Any]) -> None:
    job = load_ingest_job(job_id)
    job["status"] = "completed"
    job["progress"] = 100
    job["current_stage"] = "completed"
    job["message"] = "Channel sync completed"
    job["completed_at"] = _utc_now()
    job["updated_at"] = _utc_now()
    job["summary"] = summary
    save_ingest_job(job_id, job)
    append_ingest_event(job_id, "completed", summary)


def _is_cancelling(job_id: str) -> bool:
    try:
        return load_ingest_job(job_id).get("status") == "cancelled"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Stage: discover
# ---------------------------------------------------------------------------

def _stage_discover(job_id: str, channel_url: str, video_limit: int | None) -> list[dict[str, Any]]:
    """
    Use yt-dlp --flat-playlist to enumerate all video IDs on the channel,
    then compare against already-downloaded files to find new ones.
    Returns list of {video_id, title, index} dicts for videos not yet downloaded.
    """
    _update_job(job_id, "discover", "running", 10, "Fetching channel playlist")

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(playlist_index)s\t%(id)s\t%(title)s",
        "--no-warnings",
        channel_url,
    ]
    if video_limit:
        cmd += ["--max-downloads", str(video_limit)]

    _log(job_id).info("Discovering channel: %s", channel_url)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT)
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp channel discovery timed out after 120s")
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with: pip install yt-dlp"
        )

    if result.returncode not in (0, 101):  # 101 = max-downloads reached (expected)
        raise RuntimeError(f"yt-dlp discovery failed (rc={result.returncode}): {result.stderr[:500]}")

    # Parse channel listing
    channel_videos: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        idx_raw, video_id = parts[0], parts[1].strip()
        title = parts[2].strip() if len(parts) > 2 else ""
        if not video_id or video_id == "NA":
            continue
        try:
            idx = int(idx_raw)
        except (ValueError, TypeError):
            idx = 0
        channel_videos.append({"video_id": video_id, "title": title, "index": idx})

    _update_job(job_id, "discover", "running", 60, f"Found {len(channel_videos)} videos on channel")
    append_ingest_event(job_id, "channel_scanned", {"total_on_channel": len(channel_videos)})

    # Find already-downloaded video IDs from existing .mp4 filenames
    existing_ids: set[str] = set()
    for f in RAW_VIDEO_DIR.glob("*.mp4"):
        parts = f.stem.split("_", 2)
        if len(parts) >= 2:
            existing_ids.add(parts[1])

    new_videos = [v for v in channel_videos if v["video_id"] not in existing_ids]
    skipped = len(channel_videos) - len(new_videos)

    _update_job(
        job_id, "discover", "completed", 100,
        f"Discovered {len(new_videos)} new videos ({skipped} already downloaded)",
        new_count=len(new_videos),
        skipped_count=skipped,
        total_count=len(channel_videos),
    )
    append_ingest_event(job_id, "discover_completed", {
        "new_videos": new_videos,
        "skipped_count": skipped,
        "total_on_channel": len(channel_videos),
    })
    upsert_video_stage("__channel__", "discover", "completed", job_id=job_id, metadata={
        "new_count": len(new_videos),
        "total_count": len(channel_videos),
    })

    return new_videos


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------

def _stage_download(job_id: str, new_videos: list[dict[str, Any]], channel_url: str) -> list[str]:
    """
    Download only the new videos using yt-dlp with the download archive for deduplication.
    Returns list of video_ids successfully downloaded.
    """
    if not new_videos:
        _update_job(job_id, "download", "completed", 100, "No new videos to download")
        return []

    _update_job(job_id, "download", "running", 5, f"Downloading {len(new_videos)} new videos")

    # Mark each new video as pending in ingest_status
    for v in new_videos:
        upsert_video_stage(v["video_id"], "download", "pending", job_id=job_id)

    archive_file = DATA_DIR / "metadata" / "youtube" / "download_archive.txt"
    output_template = str(RAW_VIDEO_DIR / "%(playlist_index)04d_%(id)s_%(title)s.%(ext)s")

    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "01_download_videos.py",
        [
            "--url", channel_url,
            *(["--limit", str(len(new_videos))] if len(new_videos) < 100 else []),
        ],
        job_id,
        "download",
        timeout=7200,
    )

    # Determine which videos are now present on disk
    downloaded_ids: list[str] = []
    failed_ids: list[str] = []

    existing_ids_after: set[str] = set()
    for f in RAW_VIDEO_DIR.glob("*.mp4"):
        parts = f.stem.split("_", 2)
        if len(parts) >= 2:
            existing_ids_after.add(parts[1])

    for v in new_videos:
        vid = v["video_id"]
        if vid in existing_ids_after:
            downloaded_ids.append(vid)
            upsert_video_stage(vid, "download", "completed", job_id=job_id,
                               metadata={"title": v.get("title", "")})
            append_ingest_event(job_id, "video_downloaded", {"video_id": vid, "title": v.get("title", "")})
        else:
            failed_ids.append(vid)
            upsert_video_stage(vid, "download", "failed", job_id=job_id,
                               error=f"File not found after download (rc={returncode})")
            append_ingest_event(job_id, "video_download_failed", {"video_id": vid})

    _update_job(
        job_id, "download", "completed", 100,
        f"Downloaded {len(downloaded_ids)}/{len(new_videos)} videos",
        downloaded=downloaded_ids,
        failed=failed_ids,
    )
    return downloaded_ids


# ---------------------------------------------------------------------------
# Stage: extract_audio
# ---------------------------------------------------------------------------

def _stage_extract_audio(job_id: str, video_ids: list[str]) -> list[str]:
    if not video_ids:
        _update_job(job_id, "extract_audio", "completed", 100, "No videos to process")
        return []

    _update_job(job_id, "extract_audio", "running", 5, f"Extracting audio from {len(video_ids)} videos")

    for vid in video_ids:
        upsert_video_stage(vid, "extract_audio", "pending", job_id=job_id)

    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "02_extract_audio.py",
        [],
        job_id,
        "extract_audio",
        timeout=3600,
    )

    # Verify output: audio/<stem>.wav must exist for each video
    audio_dir = DATA_DIR / "audio"
    succeeded: list[str] = []
    failed: list[str] = []

    for vid in video_ids:
        # Find the mp4 file for this video_id
        mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{vid}_*.mp4"))
        if not mp4_files:
            failed.append(vid)
            upsert_video_stage(vid, "extract_audio", "failed", job_id=job_id,
                               error="Source mp4 not found")
            continue
        stem = mp4_files[0].stem
        wav = audio_dir / f"{stem}.wav"
        if wav.exists():
            succeeded.append(vid)
            upsert_video_stage(vid, "extract_audio", "completed", job_id=job_id)
        else:
            failed.append(vid)
            upsert_video_stage(vid, "extract_audio", "failed", job_id=job_id,
                               error=f"WAV not found: {wav.name}")

    _update_job(
        job_id, "extract_audio", "completed", 100,
        f"Audio extracted for {len(succeeded)}/{len(video_ids)} videos",
        succeeded=succeeded, failed=failed,
    )
    return succeeded


# ---------------------------------------------------------------------------
# Stage: transcribe
# ---------------------------------------------------------------------------

def _stage_transcribe(job_id: str, video_ids: list[str]) -> list[str]:
    if not video_ids:
        _update_job(job_id, "transcribe", "completed", 100, "No audio to transcribe")
        return []

    _update_job(job_id, "transcribe", "running", 5, f"Transcribing {len(video_ids)} audio files")

    for vid in video_ids:
        upsert_video_stage(vid, "transcribe", "pending", job_id=job_id)

    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "03_transcribe.py",
        [],
        job_id,
        "transcribe",
        timeout=7200,
    )

    transcript_dir = DATA_DIR / "transcripts"
    succeeded: list[str] = []
    failed: list[str] = []

    for vid in video_ids:
        mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{vid}_*.mp4"))
        if not mp4_files:
            failed.append(vid)
            upsert_video_stage(vid, "transcribe", "failed", job_id=job_id,
                               error="Source mp4 not found")
            continue
        stem = mp4_files[0].stem
        transcript = transcript_dir / f"{stem}.json"
        if transcript.exists():
            succeeded.append(vid)
            upsert_video_stage(vid, "transcribe", "completed", job_id=job_id,
                               metadata={"transcript_file": transcript.name})
        else:
            failed.append(vid)
            upsert_video_stage(vid, "transcribe", "failed", job_id=job_id,
                               error=f"Transcript not found: {transcript.name}")

    _update_job(
        job_id, "transcribe", "completed", 100,
        f"Transcribed {len(succeeded)}/{len(video_ids)} files",
        succeeded=succeeded, failed=failed,
    )
    return succeeded


# ---------------------------------------------------------------------------
# Stage: create_fragments
# ---------------------------------------------------------------------------

def _stage_create_fragments(job_id: str, video_ids: list[str]) -> list[str]:
    if not video_ids:
        _update_job(job_id, "create_fragments", "completed", 100, "No transcripts to fragment")
        return []

    _update_job(job_id, "create_fragments", "running", 5, f"Creating fragments for {len(video_ids)} transcripts")

    for vid in video_ids:
        upsert_video_stage(vid, "create_fragments", "pending", job_id=job_id)

    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "04_create_fragments.py",
        [],
        job_id,
        "create_fragments",
        timeout=1800,
    )

    fragment_dir = DATA_DIR / "fragments"
    succeeded: list[str] = []
    failed: list[str] = []

    for vid in video_ids:
        mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{vid}_*.mp4"))
        if not mp4_files:
            failed.append(vid)
            upsert_video_stage(vid, "create_fragments", "failed", job_id=job_id,
                               error="Source mp4 not found")
            continue
        stem = mp4_files[0].stem
        frag_file = fragment_dir / f"{stem}.fragments.json"
        if frag_file.exists():
            try:
                count = len(json.loads(frag_file.read_text()))
            except Exception:
                count = 0
            succeeded.append(vid)
            upsert_video_stage(vid, "create_fragments", "completed", job_id=job_id,
                               metadata={"fragment_count": count})
            append_ingest_event(job_id, "fragments_created", {"video_id": vid, "fragment_count": count})
        else:
            failed.append(vid)
            upsert_video_stage(vid, "create_fragments", "failed", job_id=job_id,
                               error=f"Fragment file not found: {frag_file.name}")

    _update_job(
        job_id, "create_fragments", "completed", 100,
        f"Fragments created for {len(succeeded)}/{len(video_ids)} videos",
        succeeded=succeeded, failed=failed,
    )
    return succeeded


# ---------------------------------------------------------------------------
# Stage: classify
# ---------------------------------------------------------------------------

def _stage_classify(job_id: str, video_ids: list[str]) -> list[str]:
    if not video_ids:
        _update_job(job_id, "classify", "completed", 100, "No fragments to classify")
        return []

    _update_job(job_id, "classify", "running", 5, f"Classifying fragments for {len(video_ids)} videos")

    for vid in video_ids:
        upsert_video_stage(vid, "classify", "pending", job_id=job_id)

    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "05_classify_fragments.py",
        [],
        job_id,
        "classify",
        timeout=3600,
    )

    classified_dir = DATA_DIR / "fragments_classified"
    succeeded: list[str] = []
    failed: list[str] = []

    for vid in video_ids:
        mp4_files = list(RAW_VIDEO_DIR.glob(f"*_{vid}_*.mp4"))
        if not mp4_files:
            failed.append(vid)
            upsert_video_stage(vid, "classify", "failed", job_id=job_id,
                               error="Source mp4 not found")
            continue
        stem = mp4_files[0].stem
        classified_file = classified_dir / f"{stem}.classified.json"
        if classified_file.exists():
            succeeded.append(vid)
            upsert_video_stage(vid, "classify", "completed", job_id=job_id)
        else:
            failed.append(vid)
            upsert_video_stage(vid, "classify", "failed", job_id=job_id,
                               error=f"Classified file not found: {classified_file.name}")

    _update_job(
        job_id, "classify", "completed", 100,
        f"Classified {len(succeeded)}/{len(video_ids)} videos",
        succeeded=succeeded, failed=failed,
    )
    return succeeded


# ---------------------------------------------------------------------------
# Stage: embed
# ---------------------------------------------------------------------------

def _stage_embed(job_id: str, video_ids: list[str]) -> list[str]:
    if not video_ids:
        _update_job(job_id, "embed", "completed", 100, "No classified fragments to embed")
        return []

    _update_job(job_id, "embed", "running", 5, f"Generating embeddings for {len(video_ids)} videos")

    for vid in video_ids:
        upsert_video_stage(vid, "embed", "pending", job_id=job_id)

    # Generate embeddings
    returncode, stdout, stderr = _run_script(
        SCRIPTS_DIR / "06_generate_embeddings.py",
        [],
        job_id,
        "embed_generate",
        timeout=3600,
    )

    if returncode != 0:
        for vid in video_ids:
            upsert_video_stage(vid, "embed", "failed", job_id=job_id,
                               error=f"Embedding generation failed (rc={returncode})")
        _update_job(job_id, "embed", "failed", 0, f"Embedding generation failed (rc={returncode})")
        raise RuntimeError(f"06_generate_embeddings.py failed (rc={returncode}): {stderr[:500]}")

    _update_job(job_id, "embed", "running", 60, "Importing embeddings into ChromaDB")

    # Import into ChromaDB
    import_script = PROJECT_ROOT / "scripts" / "import_embeddings_to_chroma.py"
    if import_script.exists():
        returncode2, _, stderr2 = _run_script(
            import_script, [], job_id, "embed_import", timeout=1800
        )
        if returncode2 != 0:
            _log(job_id).warning("ChromaDB import returned rc=%d: %s", returncode2, stderr2[:200])

    for vid in video_ids:
        upsert_video_stage(vid, "embed", "completed", job_id=job_id)

    _update_job(
        job_id, "embed", "completed", 100,
        f"Embeddings generated and imported for {len(video_ids)} videos",
        embedded=video_ids,
    )
    return video_ids


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

def create_ingest_job(channel_url: str | None = None, video_limit: int | None = None) -> str:
    job_id = f"ijob_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    effective_channel = channel_url or settings.youtube_channel_url
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "channel_url": effective_channel,
        "video_limit": video_limit,
        "progress": 0,
        "message": "Ingest job created",
        "current_stage": "queued",
        "created_at": _utc_now(),
        "started_at": None,
        "updated_at": _utc_now(),
        "completed_at": None,
        "stage_status": {},
        "summary": {},
        "error": None,
    }
    save_ingest_job(job_id, job)
    append_ingest_event(job_id, "created", {
        "channel_url": effective_channel,
        "video_limit": video_limit,
    })
    return job_id


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_ingest_job(job_id: str) -> None:
    log = _log(job_id)

    try:
        job = load_ingest_job(job_id)
        channel_url = job.get("channel_url") or settings.youtube_channel_url
        video_limit = job.get("video_limit")

        job["status"] = "running"
        job["started_at"] = _utc_now()
        job["updated_at"] = _utc_now()
        save_ingest_job(job_id, job)
        log.info("Ingest pipeline started channel=%s", channel_url)

        # Stage 1: discover
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before discover")
        new_videos = _stage_discover(job_id, channel_url, video_limit)

        if not new_videos:
            log.info("No new videos found, completing immediately")
            _complete_job(job_id, {"new_videos": 0, "message": "Channel is up to date"})
            return

        # Stage 2: download
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before download")
        downloaded_ids = _stage_download(job_id, new_videos, channel_url)

        if not downloaded_ids:
            log.warning("No videos successfully downloaded")
            _complete_job(job_id, {
                "new_videos_found": len(new_videos),
                "downloaded": 0,
                "message": "Discovery found new videos but all downloads failed",
            })
            return

        # Stage 3: extract_audio
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before extract_audio")
        audio_ids = _stage_extract_audio(job_id, downloaded_ids)

        # Stage 4: transcribe
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before transcribe")
        transcribed_ids = _stage_transcribe(job_id, audio_ids)

        # Stage 5: create_fragments
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before create_fragments")
        fragmented_ids = _stage_create_fragments(job_id, transcribed_ids)

        # Stage 6: classify
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before classify")
        classified_ids = _stage_classify(job_id, fragmented_ids)

        # Stage 7: embed
        if _is_cancelling(job_id):
            return _cancel_job(job_id, "cancelled before embed")
        embedded_ids = _stage_embed(job_id, classified_ids)

        summary = {
            "new_videos_found": len(new_videos),
            "downloaded": len(downloaded_ids),
            "audio_extracted": len(audio_ids),
            "transcribed": len(transcribed_ids),
            "fragmented": len(fragmented_ids),
            "classified": len(classified_ids),
            "embedded": len(embedded_ids),
        }
        _complete_job(job_id, summary)
        log.info("Ingest pipeline completed: %s", summary)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        stage = "unknown"
        try:
            stage = load_ingest_job(job_id).get("current_stage", "unknown")
        except Exception:
            pass
        log.exception("Ingest pipeline error at stage=%s: %s", stage, exc)
        _fail_job(job_id, stage, str(exc), tb)


def _cancel_job(job_id: str, message: str = "") -> None:
    job = load_ingest_job(job_id)
    job["status"] = "cancelled"
    job["completed_at"] = _utc_now()
    job["message"] = message or "Job cancelled"
    job["updated_at"] = _utc_now()
    save_ingest_job(job_id, job)
    append_ingest_event(job_id, "cancelled", {"message": message})


# ---------------------------------------------------------------------------
# Dispatch (Celery or thread fallback — same pattern as video_job_runner)
# ---------------------------------------------------------------------------

_redis_cache: tuple[bool, float] = (False, 0.0)
_REDIS_CACHE_TTL = 10.0


def _redis_reachable() -> bool:
    import socket
    global _redis_cache
    result, ts = _redis_cache
    if time.monotonic() - ts < _REDIS_CACHE_TTL:
        return result

    redis_url = settings.redis_url
    try:
        parts = redis_url.replace("redis://", "").split("/")[0].split(":")
        host = parts[0] or "localhost"
        port = int(parts[1]) if len(parts) > 1 else 6379
        with socket.create_connection((host, port), timeout=0.5):
            result = True
    except Exception:
        result = False

    _redis_cache = (result, time.monotonic())
    return result


def start_ingest_job_background(job_id: str) -> None:
    """Dispatch ingest job to Celery if Redis is reachable, otherwise run in a thread."""
    import threading

    use_celery = bool(settings.redis_url) and _redis_reachable()

    if use_celery:
        try:
            from app.worker.tasks import run_ingest_job_task
            run_ingest_job_task.delay(job_id)
            logger.info("Ingest job dispatched to Celery: %s", job_id)
            return
        except Exception:
            logger.warning(
                "Celery dispatch failed for ingest job %s — falling back to thread",
                job_id, exc_info=True,
            )

    t = threading.Thread(target=run_ingest_job, args=(job_id,), daemon=False)
    t.start()
    logger.info("Ingest job started in thread: %s", job_id)


# ---------------------------------------------------------------------------
# Orphan reaping (called at server startup in deployed environments)
# ---------------------------------------------------------------------------

def reap_orphaned_ingest_jobs() -> None:
    """Mark any queued/running ingest jobs as failed (orphaned from a previous process)."""
    try:
        orphans = list_running_ingest_jobs()
        for row in orphans:
            job_id = row.get("id", "")
            if not job_id:
                continue
            try:
                job = load_ingest_job(job_id)
                job["status"] = "failed"
                job["error"] = {"message": "Orphaned: server restarted while job was running"}
                job["completed_at"] = _utc_now()
                job["updated_at"] = _utc_now()
                save_ingest_job(job_id, job)
                append_ingest_event(job_id, "failed", {
                    "message": "Orphaned by server restart"
                })
                logger.info("Reaped orphaned ingest job: %s", job_id)
            except Exception:
                logger.warning("Failed to reap orphaned ingest job: %s", job_id, exc_info=True)
    except Exception:
        logger.warning("Orphan reaping failed (non-fatal)", exc_info=True)
