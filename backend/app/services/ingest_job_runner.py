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

# ---------------------------------------------------------------------------
# S3 helpers for ingest
# ---------------------------------------------------------------------------

def _s3_client():
    import boto3
    from app.services.storage_service import (
        _OBJECT_STORAGE_BUCKET, _OBJECT_STORAGE_REGION,
        _OBJECT_STORAGE_ACCESS_KEY, _OBJECT_STORAGE_SECRET_KEY, _OBJECT_STORAGE_ENDPOINT,
    )
    kwargs = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
    return boto3.client("s3", **kwargs), _OBJECT_STORAGE_BUCKET


def _s3_upload_file(local_path: Path, s3_key: str) -> None:
    s3, bucket = _s3_client()
    import mimetypes
    ct, _ = mimetypes.guess_type(str(local_path))
    s3.upload_file(str(local_path), bucket, s3_key,
                   ExtraArgs={"ContentType": ct or "application/octet-stream"})


def _s3_list_keys(prefix: str) -> set[str]:
    """Return set of stem names (without prefix) for files under a prefix."""
    s3, bucket = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.add(Path(obj["Key"]).stem)
    return keys


def _s3_upload_dir(local_dir: Path, s3_prefix: str, pattern: str = "*",
                   delete_after: bool = True) -> int:
    """Upload all files matching pattern from local_dir to S3, optionally delete local."""
    if not local_dir.exists():
        return 0
    count = 0
    for f in local_dir.glob(pattern):
        if not f.is_file():
            continue
        key = f"{s3_prefix}/{f.name}"
        try:
            _s3_upload_file(f, key)
            if delete_after:
                f.unlink()
            count += 1
        except Exception as exc:
            logger.warning("S3 upload failed %s: %s", f.name, exc)
    return count


def _video_id_from_name(name: str) -> str | None:
    """Pull the video id out of a pipeline filename.

    Every artifact is named {index}_{video_id}_{title}{suffix}, so the id sits
    in the same position whether the file is an mp4, a wav, a transcript or a
    .info.json.
    """
    parts = Path(name).stem.split("_", 2)
    return parts[1] if len(parts) >= 2 else None


def _s3_download_dir(s3_prefix: str, local_dir: Path,
                     video_ids: list[str] | None = None,
                     suffixes: tuple[str, ...] | None = None) -> int:
    """Pull a stage's inputs back down from S3. Counterpart to _s3_upload_dir.

    Every stage uploads its outputs and deletes the local copies to free disk,
    but the stage scripts only ever glob local directories — so without this
    the next stage starts against an empty directory and silently does
    nothing.

    video_ids restricts the fetch to those videos, which matters for the media
    stages where pulling the whole prefix would mean every mp4 in the corpus.
    Passing None fetches everything under the prefix, which the fragment
    stages need because they rebuild whole-corpus outputs (05's catalog, 06's
    embedding matrix) from the full directory. Files already on disk are left
    alone, so a partially-populated directory costs nothing to top up.
    """
    s3, bucket = _s3_client()
    local_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(video_ids) if video_ids is not None else None

    count = 0
    prefix = s3_prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = Path(key).name
            if not name:
                continue
            if suffixes and not name.endswith(suffixes):
                continue
            if wanted is not None and _video_id_from_name(name) not in wanted:
                continue
            target = local_dir / name
            if target.exists():
                continue
            try:
                s3.download_file(bucket, key, str(target))
                count += 1
            except Exception as exc:
                logger.warning("S3 download failed %s: %s", name, exc)
    return count


def _s3_cleanup_dir(local_dir: Path,
                    video_ids: list[str] | None = None,
                    suffixes: tuple[str, ...] | None = None) -> int:
    """Delete local scratch copies once a stage has uploaded its results.

    Only ever called behind _ingest_use_s3(). In s3 mode the bucket is the
    source of truth and these files are scratch; in local mode they ARE the
    source of truth and deleting them destroys data.

    Always scoped to one directory, never a sweep of DATA_DIR: data/metadata
    holds yt-dlp's download_archive.txt, and losing that re-downloads the whole
    channel — ~31GB onto a 30GiB volume. Returns bytes freed.
    """
    if not local_dir.exists():
        return 0
    wanted = set(video_ids) if video_ids is not None else None

    freed = 0
    for f in local_dir.iterdir():
        if not f.is_file():
            continue
        if suffixes and not f.name.endswith(suffixes):
            continue
        if wanted is not None and _video_id_from_name(f.name) not in wanted:
            continue
        try:
            size = f.stat().st_size
            f.unlink()
            freed += size
        except Exception as exc:
            logger.warning("Local cleanup failed %s: %s", f.name, exc)
    return freed


def _s3_sizes(s3_prefix: str,
              video_ids: list[str] | None = None,
              suffixes: tuple[str, ...] | None = None) -> dict[str, int]:
    """Bytes per video under a prefix, taken from the listing metadata.

    list_objects_v2 already returns Size, so budgeting a chunk costs no extra
    calls.
    """
    s3, bucket = _s3_client()
    wanted = set(video_ids) if video_ids is not None else None

    sizes: dict[str, int] = {}
    prefix = s3_prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = Path(obj["Key"]).name
            if suffixes and not name.endswith(suffixes):
                continue
            vid = _video_id_from_name(name)
            if vid is None or (wanted is not None and vid not in wanted):
                continue
            sizes[vid] = sizes.get(vid, 0) + int(obj.get("Size", 0))
    return sizes


def _chunk_by_bytes(video_ids: list[str], sizes: dict[str, int],
                    budget_bytes: int) -> list[list[str]]:
    """Group videos so each chunk's download stays under the budget.

    A video larger than the whole budget gets a chunk to itself rather than
    being skipped. Videos with no known size count as 0 so they still flow
    through and get recorded by the stage's verification step.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    running = 0

    for vid in video_ids:
        size = sizes.get(vid, 0)
        if current and running + size > budget_bytes:
            chunks.append(current)
            current, running = [], 0
        current.append(vid)
        running += size

    if current:
        chunks.append(current)
    return chunks


def _plan_chunks(video_ids: list[str], s3_prefix: str,
                 suffixes: tuple[str, ...] | None = None) -> list[list[str]]:
    """Split a stage's work into disk-bounded chunks.

    Local mode is always a single chunk: nothing is downloaded, so there is
    nothing to bound, and the stage behaves exactly as it did before chunking
    existed.
    """
    if not _ingest_use_s3():
        return [video_ids]

    budget = max(int(settings.ingest_batch_bytes), 1)
    try:
        sizes = _s3_sizes(s3_prefix, video_ids=video_ids, suffixes=suffixes)
    except Exception as exc:
        logger.warning("Could not size %s (%s) — processing in one chunk", s3_prefix, exc)
        return [video_ids]
    return _chunk_by_bytes(video_ids, sizes, budget)


def _assert_free_space(needed_bytes: int, path: Path) -> None:
    """Fail the job rather than fill the volume.

    The volume is shared with the API, which serves media off it, so running it
    to zero degrades more than this job. A failed job is recoverable; a full
    shared disk is not.
    """
    import shutil

    try:
        path.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free
    except Exception:
        return  # never block a job on an unreadable stat
    if free < needed_bytes:
        raise RuntimeError(
            f"Insufficient disk space: need ~{needed_bytes // (1024 * 1024)}MB, "
            f"only {free // (1024 * 1024)}MB free on {path}"
        )


def _ingest_use_s3() -> bool:
    return settings.ai4ia_storage_backend == "s3"


def _resolve_stem(video_id: str) -> str | None:
    """Find the filename stem (e.g. 0001_abc_Title) for a video_id.

    In S3 mode the MP4 was uploaded and deleted locally, so look up S3.
    Falls back to local disk glob.
    """
    if _ingest_use_s3():
        try:
            for stem in _s3_list_keys("raw/youtube/ai4ia_channel/"):
                parts = stem.split("_", 2)
                if len(parts) >= 2 and parts[1] == video_id:
                    return stem
        except Exception:
            pass
    # Local fallback
    matches = list(RAW_VIDEO_DIR.glob(f"*_{video_id}_*.mp4"))
    if matches:
        return matches[0].stem
    return None


from app.services.ingest_store import (
    save_ingest_job,
    load_ingest_job,
    list_running_ingest_jobs,
    append_ingest_event,
    upsert_video_stage,
)

logger = logging.getLogger("ai4ia.ingest")

PYTHON_BIN = sys.executable

# Batching for the script_log events _run_script emits. yt-dlp is chatty, so
# one event per line would hammer Postgres for no benefit.
_EVENT_BATCH_LINES = 50
_EVENT_FLUSH_SECONDS = 2.0
_EVENT_LINE_CHARS = 2000

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
    truncate: bool = True,
    cmd: list[str] | None = None,
) -> tuple[int, str, str]:
    """Run a script subprocess, streaming stdout/stderr line-by-line in real time.

    Lines go three places: the per-job log files, the server logger, and — in
    batches — job events in Postgres.

    The events matter because the log files are written by whichever container
    runs the stage, while the log-tail API endpoint reads its own filesystem.
    When those differ, every stage log looks like it never started even though
    the stage ran for a quarter of an hour. Postgres is the only channel both
    sides share, which is how the video pipeline has always done it
    (pipeline/base.py emits script_log the same way).

    Returns (returncode, stdout, stderr) for backward compat.
    """
    import threading as _threading

    log = _log(job_id)
    log_dir = DATA_DIR / "jobs" / job_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # cmd lets a caller run an arbitrary executable (e.g. the yt-dlp binary
    # directly, for discover) instead of a PYTHON_BIN-run .py script under
    # SCRIPTS_DIR — every other caller leaves it unset.
    if cmd is None:
        cmd = [PYTHON_BIN, str(script)] + args
    log.info("Running %s: %s", stage, " ".join(cmd))

    stdout_log = log_dir / f"{stage}.stdout.log"
    stderr_log = log_dir / f"{stage}.stderr.log"
    # Clear / create both log files up front so the tail endpoint can open them.
    # A chunked stage calls this once per chunk and passes truncate=False after
    # the first, so later chunks append instead of erasing earlier output.
    if truncate:
        stdout_log.write_text("")
        stderr_log.write_text("")
    else:
        stdout_log.touch()
        stderr_log.touch()

    def _drain(pipe, log_path: Path, level: int, stream_name: str = "stdout") -> str:
        """Read pipe line by line, appending to the log file, the server logger
        and (batched) the job's event stream."""
        buf: list[str] = []
        pending: list[str] = []
        last_flush = time.monotonic()

        def _flush() -> None:
            if not pending:
                return
            try:
                append_ingest_event(job_id, "script_log", {
                    "stage": stage,
                    "stream": stream_name,
                    # Long lines are almost always a yt-dlp progress bar, which
                    # emits \r rather than \n and can arrive as one enormous
                    # line. Cap it so a single row can't bloat the JSONB.
                    "lines": [ln[:_EVENT_LINE_CHARS] for ln in pending],
                })
            except Exception:  # noqa: BLE001
                pass  # observability must never break the pipeline
            pending.clear()

        try:
            with open(log_path, "a", buffering=1) as fh:
                for line in pipe:
                    fh.write(line)
                    buf.append(line)
                    stripped = line.rstrip("\n")
                    if stripped:
                        log.log(level, "[%s] %s", stage, stripped)
                        pending.append(stripped)
                    now = time.monotonic()
                    if len(pending) >= _EVENT_BATCH_LINES or (
                        pending and now - last_flush >= _EVENT_FLUSH_SECONDS
                    ):
                        _flush()
                        last_flush = now
        except Exception:
            pass
        _flush()
        return "".join(buf)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )

        stdout_buf: list[str] = []
        stderr_buf: list[str] = []

        def _drain_stdout():
            stdout_buf.append(_drain(proc.stdout, stdout_log, logging.INFO, "stdout"))

        def _drain_stderr():
            stderr_buf.append(_drain(proc.stderr, stderr_log, logging.WARNING, "stderr"))

        t_out = _threading.Thread(target=_drain_stdout, daemon=True)
        t_err = _threading.Thread(target=_drain_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            msg = f"Script timed out after {timeout}s: {script}"
            with open(stderr_log, "a") as fh:
                fh.write(f"\n{msg}\n")
            log.error("%s: %s", stage, msg)
            return -1, "".join(stdout_buf), msg

        t_out.join()
        t_err.join()
        stdout_all = "".join(stdout_buf)
        stderr_all = "".join(stderr_buf)

        if proc.returncode not in (0, 101):  # 101 = yt-dlp --max-downloads reached (expected)
            log.error(
                "%s exited rc=%d — last stderr:\n%s",
                stage, proc.returncode, stderr_all[-2000:] or "(no stderr captured)",
            )
        else:
            log.info("%s finished rc=%d", stage, proc.returncode)

        return proc.returncode, stdout_all, stderr_all

    except Exception as exc:
        msg = str(exc)
        stderr_log.write_text(msg)
        log.exception("%s: failed to launch subprocess: %s", stage, msg)
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


def _require_some_success(job_id: str, stage: str, video_ids: list[str],
                          succeeded: list[str], failed: list[str]) -> None:
    """Raise if every video in a non-empty batch failed this stage.

    A stage that verifies per video and still calls _update_job(...,
    "completed", ...) even when nothing survived is what let a 43/43 download
    failure sit in the UI as a green run with error: null -- and every stage
    downstream of download had the same shape, so the failure just moved one
    stage further before it went silently unnoticed. Mirrors how _stage_embed
    already handles its own rc!=0 failure: mark the stage failed with its
    counts, then raise so run_ingest_job's top-level except turns it into a
    real job failure via _fail_job.
    """
    if video_ids and not succeeded:
        msg = f"All {len(video_ids)} videos failed {stage}"
        _update_job(job_id, stage, "failed", 0, msg, succeeded=succeeded, failed=failed)
        raise RuntimeError(msg)


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
        *(["--cookies", settings.youtube_cookies_file] if settings.youtube_cookies_file else []),
        "--flat-playlist",
        "--print", "%(playlist_index)s\t%(id)s\t%(title)s",
        "--no-warnings",
        channel_url,
    ]
    if video_limit:
        cmd += ["--max-downloads", str(video_limit)]

    disc_log = _log(job_id)
    disc_log.info("Discovering channel: %s", channel_url)

    # Routed through _run_script (rather than a bare subprocess.run, as before)
    # so discovery's yt-dlp output reaches script_log events in Postgres like
    # every other stage's — previously a discovery failure had nothing an
    # operator could inspect beyond the container's own log stream.
    returncode, stdout, stderr = _run_script(
        "yt-dlp", [], job_id, "discover", timeout=120, cmd=cmd,
    )

    if stderr:
        disc_log.warning("yt-dlp discovery stderr:\n%s", stderr[-2000:])

    if returncode not in (0, 101):  # 101 = max-downloads reached (expected)
        raise RuntimeError(f"yt-dlp discovery failed (rc={returncode}): {stderr[:500]}")

    # Parse channel listing
    channel_videos: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
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

    # Find already-downloaded video IDs.
    # Always read the yt-dlp download archive first (most authoritative — survives local deletes).
    # Then supplement with local mp4 scan and, in s3 mode, S3 key scan.
    existing_ids: set[str] = set()

    archive_file = DATA_DIR / "metadata" / "youtube" / "download_archive.txt"
    if archive_file.exists():
        try:
            for line in archive_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    existing_ids.add(parts[1])  # format: "youtube <video_id>"
        except Exception as exc:
            logger.warning("Could not read download archive: %s", exc)

    if _ingest_use_s3():
        try:
            for stem in _s3_list_keys("raw/youtube/ai4ia_channel/"):
                parts = stem.split("_", 2)
                if len(parts) >= 2:
                    existing_ids.add(parts[1])
        except Exception as exc:
            logger.warning("S3 discover check failed, falling back to local: %s", exc)
            for f in RAW_VIDEO_DIR.glob("*.mp4"):
                parts = f.stem.split("_", 2)
                if len(parts) >= 2:
                    existing_ids.add(parts[1])
    else:
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
    disc_log.info(
        "Discover stage: %d total on channel, %d new, %d already downloaded",
        len(channel_videos), len(new_videos), skipped,
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
            *(["--cookies", settings.youtube_cookies_file] if settings.youtube_cookies_file else []),
        ],
        job_id,
        "download",
        timeout=7200,
    )

    # Upload downloaded videos to S3, then verify
    downloaded_ids: list[str] = []
    failed_ids: list[str] = []

    if _ingest_use_s3():
        # Upload all new mp4/info.json/jpg/vtt files to S3, delete local to free disk
        for f in list(RAW_VIDEO_DIR.glob("*")):
            if not f.is_file():
                continue
            key = f"raw/youtube/ai4ia_channel/{f.name}"
            try:
                _s3_upload_file(f, key)
                f.unlink()
            except Exception as exc:
                logger.warning("Failed to upload %s to S3: %s", f.name, exc)
        # Verify by checking S3
        s3_ids_after: set[str] = set()
        try:
            for stem in _s3_list_keys("raw/youtube/ai4ia_channel/"):
                parts = stem.split("_", 2)
                if len(parts) >= 2:
                    s3_ids_after.add(parts[1])
        except Exception:
            pass
        for v in new_videos:
            vid = v["video_id"]
            if vid in s3_ids_after:
                downloaded_ids.append(vid)
                upsert_video_stage(vid, "download", "completed", job_id=job_id,
                                   metadata={"title": v.get("title", "")})
                append_ingest_event(job_id, "video_downloaded", {"video_id": vid, "title": v.get("title", "")})
            else:
                failed_ids.append(vid)
                upsert_video_stage(vid, "download", "failed", job_id=job_id,
                                   error=f"Not found in S3 after upload (rc={returncode})")
                append_ingest_event(job_id, "video_download_failed", {"video_id": vid})
    else:
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
    if downloaded_ids:
        _log(job_id).info("Download stage: %d/%d videos downloaded", len(downloaded_ids), len(new_videos))
    else:
        _log(job_id).error(
            "Download stage: 0/%d videos downloaded (yt-dlp rc=%d) — see the [download] "
            "lines above for the actual yt-dlp error. Common causes: no JS runtime in the "
            "worker image for YouTube's challenge, or a bot-check/sign-in wall on this IP.",
            len(new_videos), returncode,
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

    # mp4s are the largest artifact in the pipeline, so this stage is chunked:
    # _stage_download uploaded them and deleted the local copies, and pulling
    # the whole corpus back would be ~31GB onto a 30GiB volume.
    audio_dir = DATA_DIR / "audio"
    succeeded: list[str] = []
    failed: list[str] = []

    chunks = _plan_chunks(video_ids, "raw/youtube/ai4ia_channel", (".mp4",))
    for idx, chunk in enumerate(chunks):
        if _ingest_use_s3():
            _assert_free_space(int(settings.ingest_batch_bytes) * 2, RAW_VIDEO_DIR)
            _s3_download_dir("raw/youtube/ai4ia_channel", RAW_VIDEO_DIR,
                             video_ids=chunk, suffixes=(".mp4",))

        returncode, stdout, stderr = _run_script(
            SCRIPTS_DIR / "02_extract_audio.py",
            [],
            job_id,
            "extract_audio",
            timeout=3600,
            truncate=(idx == 0),
        )

        # Verify output: audio/<stem>.wav must exist for each video
        for vid in chunk:
            stem = _resolve_stem(vid)
            if not stem:
                failed.append(vid)
                upsert_video_stage(vid, "extract_audio", "failed", job_id=job_id,
                                   error="Source video stem not found")
                continue
            wav = audio_dir / f"{stem}.wav"
            if wav.exists():
                succeeded.append(vid)
                upsert_video_stage(vid, "extract_audio", "completed", job_id=job_id)
            else:
                failed.append(vid)
                upsert_video_stage(vid, "extract_audio", "failed", job_id=job_id,
                                   error=f"WAV not found: {wav.name}")

        if _ingest_use_s3():
            _s3_upload_dir(DATA_DIR / "audio", "raw/audio", "*.wav", delete_after=True)
            freed = _s3_cleanup_dir(RAW_VIDEO_DIR, video_ids=chunk, suffixes=(".mp4",))
            _log(job_id).info("extract_audio chunk %d/%d done, freed %dMB",
                              idx + 1, len(chunks), freed // (1024 * 1024))

        if len(chunks) > 1:
            _update_job(
                job_id, "extract_audio", "running",
                5 + int(90 * (idx + 1) / len(chunks)),
                f"Extracted audio for {len(succeeded)}/{len(video_ids)} videos",
            )

    _require_some_success(job_id, "extract_audio", video_ids, succeeded, failed)
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

    # WAVs are uncompressed 16kHz mono PCM — 115.2MB per hour, and the longest
    # video in the corpus is a 444MB WAV — so this stage is chunked too.
    transcript_dir = DATA_DIR / "transcripts"
    audio_dir = DATA_DIR / "audio"
    succeeded: list[str] = []
    failed: list[str] = []

    chunks = _plan_chunks(video_ids, "raw/audio", (".wav",))
    for idx, chunk in enumerate(chunks):
        if _ingest_use_s3():
            _assert_free_space(int(settings.ingest_batch_bytes) * 2, audio_dir)
            _s3_download_dir("raw/audio", audio_dir,
                             video_ids=chunk, suffixes=(".wav",))

        returncode, stdout, stderr = _run_script(
            SCRIPTS_DIR / "03_transcribe.py",
            [],
            job_id,
            "transcribe",
            timeout=7200,
            truncate=(idx == 0),
        )

        for vid in chunk:
            stem = _resolve_stem(vid)
            if not stem:
                failed.append(vid)
                upsert_video_stage(vid, "transcribe", "failed", job_id=job_id,
                                   error="Source video stem not found")
                continue
            transcript = transcript_dir / f"{stem}.json"
            if transcript.exists():
                succeeded.append(vid)
                upsert_video_stage(vid, "transcribe", "completed", job_id=job_id,
                                   metadata={"transcript_file": transcript.name})
            else:
                failed.append(vid)
                upsert_video_stage(vid, "transcribe", "failed", job_id=job_id,
                                   error=f"Transcript not found: {transcript.name}")

        if _ingest_use_s3():
            _s3_upload_dir(DATA_DIR / "transcripts", "raw/transcripts", "*", delete_after=True)
            freed = _s3_cleanup_dir(audio_dir, video_ids=chunk, suffixes=(".wav",))
            _log(job_id).info("transcribe chunk %d/%d done, freed %dMB",
                              idx + 1, len(chunks), freed // (1024 * 1024))

        if len(chunks) > 1:
            _update_job(
                job_id, "transcribe", "running",
                5 + int(90 * (idx + 1) / len(chunks)),
                f"Transcribed {len(succeeded)}/{len(video_ids)} files",
            )

    _require_some_success(job_id, "transcribe", video_ids, succeeded, failed)
    _update_job(
        job_id, "transcribe", "completed", 100,
        f"Transcribed {len(succeeded)}/{len(video_ids)} files",
        succeeded=succeeded, failed=failed,
    )
    _emit_ingest_trace(job_id, agent="transcription", model="faster-whisper",
                       call_type="speech_to_text", decision_label="audio_transcription",
                       video_count=len(succeeded))
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

    if _ingest_use_s3():
        _s3_download_dir("raw/transcripts", DATA_DIR / "transcripts",
                         video_ids=video_ids, suffixes=(".json",))
        # 04 reads {stem}.info.json out of the raw video dir for video_year and
        # video_upload_date; without it those fields come back empty.
        _s3_download_dir("raw/youtube/ai4ia_channel", RAW_VIDEO_DIR,
                         video_ids=video_ids, suffixes=(".info.json",))

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
        stem = _resolve_stem(vid)
        if not stem:
            failed.append(vid)
            upsert_video_stage(vid, "create_fragments", "failed", job_id=job_id,
                               error="Source video stem not found")
            continue
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

    if _ingest_use_s3():
        _s3_upload_dir(DATA_DIR / "fragments", "raw/fragments", "*.fragments.json", delete_after=True)
        # Release the inputs this stage pulled down. The volume is persistent,
        # so anything left here survives every restart until deleted.
        _s3_cleanup_dir(DATA_DIR / "transcripts", video_ids=video_ids)
        _s3_cleanup_dir(RAW_VIDEO_DIR, video_ids=video_ids, suffixes=(".info.json",))

    _require_some_success(job_id, "create_fragments", video_ids, succeeded, failed)
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

    if _ingest_use_s3():
        # Whole prefix, not just this job's videos: 05 rewrites
        # classified_fragments_catalog.jsonl from whatever is in the directory,
        # so a partial fetch would truncate every other video out of it.
        _s3_download_dir("raw/fragments", DATA_DIR / "fragments")

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
        stem = _resolve_stem(vid)
        if not stem:
            failed.append(vid)
            upsert_video_stage(vid, "classify", "failed", job_id=job_id,
                               error="Source video stem not found")
            continue
        classified_file = classified_dir / f"{stem}.classified.json"
        if classified_file.exists():
            succeeded.append(vid)
            upsert_video_stage(vid, "classify", "completed", job_id=job_id)
        else:
            failed.append(vid)
            upsert_video_stage(vid, "classify", "failed", job_id=job_id,
                               error=f"Classified file not found: {classified_file.name}")

    if _ingest_use_s3():
        _s3_upload_dir(DATA_DIR / "fragments_classified", "raw/fragments_classified",
                       "*.classified.json", delete_after=True)
        _s3_cleanup_dir(DATA_DIR / "fragments", suffixes=(".fragments.json",))

    _require_some_success(job_id, "classify", video_ids, succeeded, failed)
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

    if _ingest_use_s3():
        # Whole prefix for the same reason as classify: 06 rebuilds the
        # embedding jsonl and matrix from the full directory.
        _s3_download_dir("raw/fragments_classified", DATA_DIR / "fragments_classified")
        # Seeds 06's hash cache so unchanged fragments are not re-embedded.
        _s3_download_dir("raw/embeddings", DATA_DIR / "embeddings", suffixes=(".jsonl",))

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
        # succeeded/failed recorded here (unlike before) so a resume can read
        # stage_status["embed"]["failed"] the same way it reads every other
        # tail stage's — see _resolve_resume_candidates.
        _update_job(job_id, "embed", "failed", 0, f"Embedding generation failed (rc={returncode})",
                   succeeded=[], failed=list(video_ids))
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

    if _ingest_use_s3():
        _s3_upload_dir(DATA_DIR / "embeddings", "raw/embeddings", "*.jsonl", delete_after=False)
        # stage_07_storyline's _prepare_embeddings requires all three of these
        # (see _EMBEDDINGS_FILES there) and raises if any is missing — without
        # uploading the matrix and metadata, ingest can finish clean and video
        # generation still cannot start.
        _s3_upload_dir(DATA_DIR / "embeddings", "raw/embeddings", "*.npy", delete_after=False)
        _s3_upload_dir(DATA_DIR / "embeddings", "raw/embeddings", "*.json", delete_after=False)
        _s3_upload_dir(DATA_DIR / "embeddings_cleaned", "raw/embeddings_cleaned", "*.jsonl", delete_after=False)
        # The classified fragments were only ever scratch for this stage. The
        # embeddings dir is left alone deliberately — it seeds 06's hash cache
        # so unchanged fragments are not re-embedded next run.
        _s3_cleanup_dir(DATA_DIR / "fragments_classified", suffixes=(".classified.json",))

    for vid in video_ids:
        upsert_video_stage(vid, "embed", "completed", job_id=job_id)

    _update_job(
        job_id, "embed", "completed", 100,
        f"Embeddings generated and imported for {len(video_ids)} videos",
        embedded=video_ids,
    )
    _emit_ingest_trace(job_id, agent="embedding", model="nomic-embed-text:v1.5",
                       call_type="embed", decision_label="fragment_embedding",
                       video_count=len(video_ids))
    return video_ids


# ---------------------------------------------------------------------------
# Resume — retry the tail of a job (or a single video) without repeating
# discover/download
# ---------------------------------------------------------------------------
#
# A stage failure used to mean re-running the whole job from discover, and
# discover only finds videos not yet in yt-dlp's download archive — so any
# video that *was* downloaded but failed a later stage became permanently
# invisible to every future sync. The five tail stages all share the shape
# (job_id, video_ids) -> list[str], with each stage's own script processing
# whatever it finds on disk/S3 regardless of video_ids (video_ids only scopes
# verification and per-video bookkeeping) — which is what makes resuming a
# subset safe: calling a stage with a different video_ids list cannot change
# what its script does to the filesystem, only which videos get checked and
# recorded.

RESUMABLE_STAGES: list[str] = ["extract_audio", "transcribe", "create_fragments", "classify", "embed"]

_RESUMABLE_STAGE_FNS: dict[str, Any] = {
    "extract_audio": _stage_extract_audio,
    "transcribe": _stage_transcribe,
    "create_fragments": _stage_create_fragments,
    "classify": _stage_classify,
    "embed": _stage_embed,
}

# Where to find "what fed this stage" when it needs to look at its
# predecessor rather than its own failed list — (predecessor stage,
# predecessor's success key). download uses "downloaded" instead of
# "succeeded" for historical reasons; every tail stage uses "succeeded".
_PREDECESSOR: dict[str, tuple[str, str]] = {
    "extract_audio": ("download", "downloaded"),
    "transcribe": ("extract_audio", "succeeded"),
    "create_fragments": ("transcribe", "succeeded"),
    "classify": ("create_fragments", "succeeded"),
    "embed": ("classify", "succeeded"),
}


def resolve_resume_candidates(job: dict[str, Any],
                              from_stage: str | None = None) -> tuple[str, list[str]]:
    """Work out where to resume a failed job and which videos to retry.

    Returns (from_stage, video_ids). Raises ValueError with an operator-facing
    message on anything that makes the job unresumable — the API layer turns
    that straight into a 400.

    from_stage, if not given, is the job's current_stage at the moment it
    failed (both _fail_job and _update_job set it). discover/download
    failures are refused: those videos were never fully downloaded, so a
    plain POST /sync already picks them back up correctly on its own.

    video_ids prefers stage_status[from_stage]["failed"] — the exact batch
    _require_some_success recorded as having failed that stage, scoped to
    this one job, not a cross-job query. Falls back to the predecessor
    stage's success list, which covers a stage that crashed before reaching
    _require_some_success (so never recorded its own failed list) but still
    left a record of what was handed to it.
    """
    stage_status = job.get("stage_status") or {}
    stage = from_stage or job.get("current_stage")

    if stage not in RESUMABLE_STAGES:
        raise ValueError(
            f"Cannot resume from stage '{stage}' — only {RESUMABLE_STAGES} are resumable. "
            "A discover or download failure needs a fresh POST /ingest/sync instead; "
            "download already retries anything not yet in the archive."
        )

    own_failed = (stage_status.get(stage) or {}).get("failed") or []
    if own_failed:
        return stage, list(own_failed)

    predecessor_stage, key = _PREDECESSOR[stage]
    predecessor_ids = (stage_status.get(predecessor_stage) or {}).get(key) or []
    if predecessor_ids:
        return stage, list(predecessor_ids)

    raise ValueError(
        f"Nothing to resume: no failed videos recorded for '{stage}' and its "
        f"predecessor '{predecessor_stage}' recorded no successes either."
    )


def create_resume_job(*, resume_from_stage: str, resume_video_ids: list[str],
                      channel_url: str | None = None,
                      resumed_from_job_id: str | None = None) -> str:
    """Create a job that starts at resume_from_stage for resume_video_ids
    instead of at discover. Used by both POST /jobs/{id}/resume (retrying a
    specific failed job's own failed videos) and the single-video retrigger
    endpoint (force-reprocessing one already-downloaded video regardless of
    job history) — they differ only in how the video list is sourced."""
    job_id = create_ingest_job(channel_url=channel_url)
    job = load_ingest_job(job_id)
    job["resume_from_stage"] = resume_from_stage
    job["resume_video_ids"] = list(resume_video_ids)
    if resumed_from_job_id:
        job["resumed_from_job_id"] = resumed_from_job_id
    save_ingest_job(job_id, job)
    return job_id


def _run_resumed_stages(job_id: str, from_stage: str, video_ids: list[str]) -> None:
    """Run RESUMABLE_STAGES[from_stage:] for exactly video_ids.

    Deliberately does not seed discover/download or earlier tail stages into
    stage_status as fake-completed: /provenance builds its stage trace
    straight from stage_status, and a resumed job that never ran discover
    should not claim it did. The visible cost is that a resumed run's
    progress percentage starts lower than 100 minus the remaining stages'
    weight — a cosmetic gap, not a correctness one.

    Raises on failure rather than calling _fail_job itself: run_ingest_job's
    caller already wraps this in the same top-level except that handles every
    other stage failure, so a second failure path here would risk a racy
    double-write to the job row instead of a single consistent one.
    """
    log = _log(job_id)
    if from_stage not in _RESUMABLE_STAGE_FNS:
        raise RuntimeError(f"Cannot resume from stage '{from_stage}'")
    if not video_ids:
        raise RuntimeError("Resume requested with no video_ids to process")

    names = RESUMABLE_STAGES
    ids = list(video_ids)
    for name in names[names.index(from_stage):]:
        if _is_cancelling(job_id):
            return _cancel_job(job_id, f"cancelled before {name}")
        ids = _RESUMABLE_STAGE_FNS[name](job_id, ids)

    summary = {
        "resumed_from_stage": from_stage,
        "resumed_video_count": len(video_ids),
        "embedded": len(ids),
    }
    _complete_job(job_id, summary)
    log.info("Resumed ingest pipeline completed: %s", summary)
    _emit_ingest_decision_record(job_id, summary)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _emit_ingest_decision_record(job_id: str, summary: dict) -> None:
    """Fire-and-forget Layer 1 decision record for a completed ingest job."""
    import threading

    def _insert() -> None:
        try:
            from app.db.repositories.decision_record_repository import DecisionRecordRepository
            from datetime import timedelta
            job = load_ingest_job(job_id)
            started_at = job.get("started_at")
            completed_at = job.get("completed_at")
            duration_ms: int | None = None
            if started_at and completed_at:
                try:
                    t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                    duration_ms = int((t1 - t0).total_seconds() * 1000)
                except Exception:
                    pass
            expires_at = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
            DecisionRecordRepository().create({
                "id": f"dr_{job_id}",
                "job_id": job_id,
                "schema_version": "osi-1.0",
                "subject": "video_ingestion",
                "user_prompt": job.get("channel_url"),
                "accountability": "ai4ia-pipeline-v2",
                "agents_executed": ["transcription", "classification", "embedding"],
                "decisions": {
                    "videos_discovered": summary.get("new_videos_found", 0),
                    "videos_transcribed": summary.get("transcribed", 0),
                    "fragments_classified": summary.get("classified", 0),
                    "fragments_embedded": summary.get("embedded", 0),
                },
                "models_used": {
                    "transcription": "faster-whisper",
                    "classification": "rule_based_keyword_taxonomy",
                    "embedding": "nomic-embed-text:v1.5",
                },
                "outcome": "ingestion_completed",
                "total_duration_ms": duration_ms,
                "expires_at": expires_at,
                "created_at": completed_at or datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    threading.Thread(target=_insert, daemon=True).start()


def _emit_ingest_trace(
    job_id: str,
    agent: str,
    model: str,
    call_type: str,
    decision_label: str,
    video_count: int = 0,
) -> None:
    """Fire-and-forget LLM/model trace for an ingest stage that ran as a subprocess."""
    import threading

    def _insert() -> None:
        try:
            from app.db.repositories.llm_trace_repository import LLMTraceRepository
            LLMTraceRepository().create({
                "id": f"lt_{uuid.uuid4().hex}",
                "job_id": job_id,
                "stage": "ingest",
                "agent": agent,
                "model": model,
                "call_type": call_type,
                "decision_label": f"{decision_label}:{video_count}_videos",
                "retry_count": 0,
                "called_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    threading.Thread(target=_insert, daemon=True).start()


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

        resume_from_stage = job.get("resume_from_stage")
        if resume_from_stage:
            _run_resumed_stages(job_id, resume_from_stage, job.get("resume_video_ids") or [])
            return

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
            # Every download failing is a failure, not a green run. Reporting
            # it as completed is what let a 43-of-43 cookie-wall failure sit in
            # the UI as a successful sync with error: null. The "nothing new to
            # do" case is handled above and never reaches here, so arriving
            # here means videos were found and all of them failed.
            job = load_ingest_job(job_id)
            job["summary"] = {
                "new_videos_found": len(new_videos),
                "downloaded": 0,
                "message": "Discovery found new videos but all downloads failed",
            }
            save_ingest_job(job_id, job)
            _fail_job(
                job_id, "download",
                f"All {len(new_videos)} downloads failed",
                "yt-dlp produced no usable files. On cloud/datacenter IPs the "
                "usual cause is YouTube's bot check — set YOUTUBE_COOKIES_FILE "
                "to a mounted cookies.txt. The download stage's script_log "
                "events carry yt-dlp's own error text.",
            )
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
        _emit_ingest_decision_record(job_id, summary)

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
        from urllib.parse import urlsplit

        parts = urlsplit(redis_url)
        host = parts.hostname or "localhost"
        port = parts.port or 6379
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

    # A job killed mid-chunk leaves its media behind, and the volume is
    # persistent, so nothing reclaims it on its own — repeated failures fill
    # the disk the API also serves from. Safe only in s3 mode, where these
    # directories hold nothing authoritative, and only after the loop above has
    # failed every in-flight job. data/metadata is never touched: it holds
    # yt-dlp's download archive.
    if not _ingest_use_s3():
        return
    try:
        freed = 0
        freed += _s3_cleanup_dir(RAW_VIDEO_DIR, suffixes=(".mp4",))
        freed += _s3_cleanup_dir(DATA_DIR / "audio", suffixes=(".wav",))
        if freed:
            logger.info("Reclaimed %dMB of orphaned ingest media", freed // (1024 * 1024))
    except Exception:
        logger.warning("Orphaned media sweep failed (non-fatal)", exc_info=True)
