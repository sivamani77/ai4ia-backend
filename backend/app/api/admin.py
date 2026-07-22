from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services import embedding_index, fragment_store

router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Export archive — every completed job's final video/subtitles/assets,
# across ALL users (including jobs orphaned under a pre-auth session UUID).
# Admin-only: gated by require_role("admin") at the router include in main.py.
# ---------------------------------------------------------------------------

@router.get("/exports/archive")
def list_export_archive(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str = Query("completed", description="Filter by job status, or 'all'"),
):
    """List every job's final exported assets (video, srt/vtt, manifest,
    thumbnail) for the admin export-archive screen. Includes jobs whose
    user_id is a legacy anonymous session UUID — those are only reachable
    through this admin view, never through the per-account /video-jobs
    endpoints."""
    from app.services.job_store import list_jobs
    from app.services.admin_content_service import job_to_generation_entry

    all_jobs = list_jobs()
    if status != "all":
        all_jobs = [j for j in all_jobs if j.get("status") == status]
    all_jobs.sort(key=lambda j: j.get("completed_at") or j.get("created_at") or "", reverse=True)

    total = len(all_jobs)
    page = all_jobs[offset: offset + limit]

    entries = [job_to_generation_entry(job) for job in page]

    return {
        "jobs": entries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# S3 helpers (lazy — only imported when storage backend is s3)
# ---------------------------------------------------------------------------

_S3_PREFIXES = {
    "videos":      "raw/youtube/ai4ia_channel/",
    "audio":       "raw/audio/",
    "transcripts": "raw/transcripts/",
    "fragments":   "raw/fragments/",
    "classified":  "raw/fragments_classified/",
    "embeddings":  "raw/embeddings/",
    "storylines":  "raw/storylines/",
    "assets":      "generated_assets/",
    "exports":     "exports/",
}


def _require_s3():
    from app.core.settings import settings
    if settings.ai4ia_storage_backend != "s3":
        raise HTTPException(status_code=400, detail="Storage backend is not S3")


def _s3_client():
    from app.services.storage_service import (
        _OBJECT_STORAGE_BUCKET,
        _OBJECT_STORAGE_REGION,
        _OBJECT_STORAGE_ACCESS_KEY,
        _OBJECT_STORAGE_SECRET_KEY,
        _OBJECT_STORAGE_ENDPOINT,
    )
    import boto3
    kwargs: dict[str, Any] = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
    return boto3.client("s3", **kwargs)


def _bucket() -> str:
    from app.services.storage_service import _OBJECT_STORAGE_BUCKET
    return _OBJECT_STORAGE_BUCKET


def _list_keys(prefix: str) -> list[str]:
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _list_objects_with_size(prefix: str) -> list[dict]:
    """Return list of {key, size} dicts for all objects under prefix."""
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[dict] = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj.get("Size", 0)})
    return objects


def _delete_keys(keys: list[str]) -> int:
    if not keys:
        return 0
    s3 = _s3_client()
    bucket = _bucket()
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i:i + 1000]]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
    return deleted


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------

@router.post("/reindex")
def api_reindex():
    """Reload fragment store count and embedding matrix from disk."""
    fs_result = fragment_store.reload()
    ei_result = embedding_index.reload()
    return {
        "status": "ok",
        "fragment_store": fs_result,
        "embedding_index": ei_result,
        "message": "Index reloaded from disk",
    }


# ---------------------------------------------------------------------------
# S3 storage — space usage
# ---------------------------------------------------------------------------

@router.get("/storage/space")
def storage_space():
    """
    Return total S3 storage used, broken down by directory.
    Each directory shows object count and total size in bytes + human-readable.
    """
    _require_s3()

    breakdown: dict[str, Any] = {}
    grand_total_bytes = 0
    grand_total_objects = 0

    for name, prefix in _S3_PREFIXES.items():
        objects = _list_objects_with_size(prefix)
        size_bytes = sum(o["size"] for o in objects)
        count = len(objects)
        breakdown[name] = {
            "prefix": prefix,
            "object_count": count,
            "size_bytes": size_bytes,
            "size_human": _fmt_bytes(size_bytes),
        }
        grand_total_bytes += size_bytes
        grand_total_objects += count

    return {
        "total_size_bytes": grand_total_bytes,
        "total_size_human": _fmt_bytes(grand_total_bytes),
        "total_object_count": grand_total_objects,
        "bucket": _bucket(),
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# S3 storage — list directories / objects
# ---------------------------------------------------------------------------

@router.get("/storage/directories")
def list_directories():
    """List all known S3 storage directories with object counts and sizes."""
    _require_s3()

    result: dict[str, Any] = {}
    for name, prefix in _S3_PREFIXES.items():
        objects = _list_objects_with_size(prefix)
        size_bytes = sum(o["size"] for o in objects)
        result[name] = {
            "prefix": prefix,
            "count": len(objects),
            "size_bytes": size_bytes,
            "size_human": _fmt_bytes(size_bytes),
        }
    return {"directories": result}


@router.get("/storage/directories/{directory}")
def list_directory_objects(directory: str):
    """List all objects in a named S3 directory."""
    _require_s3()

    if directory not in _S3_PREFIXES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown directory '{directory}'. Valid: {list(_S3_PREFIXES.keys())}",
        )

    prefix = _S3_PREFIXES[directory]
    objects = _list_objects_with_size(prefix)
    size_bytes = sum(o["size"] for o in objects)
    return {
        "directory": directory,
        "prefix": prefix,
        "count": len(objects),
        "size_bytes": size_bytes,
        "size_human": _fmt_bytes(size_bytes),
        "objects": [
            {
                "key": o["key"],
                "name": o["key"][len(prefix):],
                "size_bytes": o["size"],
                "size_human": _fmt_bytes(o["size"]),
            }
            for o in objects
        ],
    }


# ---------------------------------------------------------------------------
# S3 storage — delete entire directory
# ---------------------------------------------------------------------------

@router.delete("/storage/directories/{directory}")
def delete_directory(directory: str, confirm: str = ""):
    """
    Delete ALL objects in a named S3 directory.

    Requires `?confirm=<directory>` to prevent accidental deletion.

    Valid directories: videos, audio, transcripts, fragments, classified,
                       embeddings, storylines, assets, exports
    """
    _require_s3()

    if directory not in _S3_PREFIXES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown directory '{directory}'. Valid: {list(_S3_PREFIXES.keys())}",
        )

    if confirm != directory:
        raise HTTPException(
            status_code=400,
            detail=f"Pass ?confirm={directory} to confirm deletion of all objects in '{directory}'",
        )

    prefix = _S3_PREFIXES[directory]
    keys = _list_keys(prefix)
    deleted = _delete_keys(keys)

    return {
        "status": "deleted",
        "directory": directory,
        "prefix": prefix,
        "deleted_count": deleted,
    }


# ---------------------------------------------------------------------------
# S3 storage — delete single object by key
# ---------------------------------------------------------------------------

class DeleteObjectRequest(BaseModel):
    key: str


@router.delete("/storage/object")
def delete_object(body: DeleteObjectRequest):
    """Delete a single S3 object by its full key (e.g. `raw/youtube/ai4ia_channel/foo.mp4`)."""
    _require_s3()

    key = body.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="key must not be empty")

    try:
        _s3_client().delete_object(Bucket=_bucket(), Key=key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete object: {exc}")

    return {"status": "deleted", "key": key}


# ---------------------------------------------------------------------------
# S3 storage — delete video and all its derived artifacts
# ---------------------------------------------------------------------------

@router.delete("/storage/videos/{video_stem}")
def delete_video_artifacts(video_stem: str, confirm: str = ""):
    """
    Delete a source video and ALL its derived artifacts (audio, transcripts,
    fragments, classified fragments, embeddings) by video stem (filename without extension).

    Requires `?confirm=<video_stem>`.

    Example:
        DELETE /api/admin/storage/videos/ai4ia_xK2mP_some_title?confirm=ai4ia_xK2mP_some_title
    """
    _require_s3()

    if confirm != video_stem:
        raise HTTPException(
            status_code=400,
            detail=f"Pass ?confirm={video_stem} to confirm deletion",
        )

    stem = video_stem.strip()
    if not stem:
        raise HTTPException(status_code=400, detail="video_stem must not be empty")

    search_prefixes = [
        "raw/youtube/ai4ia_channel/",
        "raw/audio/",
        "raw/transcripts/",
        "raw/fragments/",
        "raw/fragments_classified/",
        "raw/embeddings/",
    ]

    all_keys: list[str] = []
    matched_by_prefix: dict[str, int] = {}

    for prefix in search_prefixes:
        keys = [k for k in _list_keys(prefix) if stem in k]
        if keys:
            matched_by_prefix[prefix] = len(keys)
            all_keys.extend(keys)

    if not all_keys:
        raise HTTPException(status_code=404, detail=f"No artifacts found for stem '{stem}'")

    deleted = _delete_keys(all_keys)

    return {
        "status": "deleted",
        "video_stem": stem,
        "deleted_count": deleted,
        "deleted_by_prefix": matched_by_prefix,
    }


# ---------------------------------------------------------------------------
# S3 storage — delete export artifacts for a job
# ---------------------------------------------------------------------------

@router.delete("/storage/exports/{job_id}")
def delete_export_artifacts(job_id: str, confirm: str = ""):
    """
    Delete all S3 export artifacts (mp4, srt, vtt, manifest) for a video generation job.

    Also removes any generated_assets associated with the job.

    Requires `?confirm=<job_id>`.
    """
    _require_s3()

    if confirm != job_id:
        raise HTTPException(
            status_code=400,
            detail=f"Pass ?confirm={job_id} to confirm deletion",
        )

    jid = job_id.strip()
    if not jid:
        raise HTTPException(status_code=400, detail="job_id must not be empty")

    keys = [k for k in _list_keys("exports/") if jid in k]
    keys += [k for k in _list_keys("generated_assets/") if jid in k]

    if not keys:
        raise HTTPException(status_code=404, detail=f"No export artifacts found for job '{jid}'")

    deleted = _delete_keys(keys)

    return {
        "status": "deleted",
        "job_id": jid,
        "deleted_count": deleted,
        "deleted_keys": keys,
    }


# ---------------------------------------------------------------------------
# Local file retention cleanup
# ---------------------------------------------------------------------------

@router.post("/cleanup/local-exports")
def cleanup_local_exports(
    older_than_days: int = Query(7, ge=1, le=365, description="Delete local job artifacts older than N days"),
    dry_run: bool = Query(True, description="Set to false to actually delete"),
):
    """
    Delete local export files (mp4, srt, vtt, manifest) and per-job asset dirs
    for jobs whose completed_at is older than `older_than_days` days.

    Only removes files that have already been uploaded to S3 (storage_backend=s3)
    or are from terminal jobs (completed/failed/cancelled) in local mode.

    Always defaults to dry_run=true — pass ?dry_run=false to actually delete.
    """
    from app.core.paths import DATA_DIR
    from app.services.job_store import list_jobs

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    terminal_statuses = {"completed", "failed", "cancelled"}

    try:
        all_jobs = list_jobs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load jobs: {exc}")

    deleted_paths: list[str] = []
    skipped_paths: list[str] = []

    for job in all_jobs:
        status = job.get("status", "")
        if status not in terminal_statuses:
            continue

        completed_at_raw = job.get("completed_at")
        if not completed_at_raw:
            continue

        try:
            completed_at = datetime.fromisoformat(completed_at_raw)
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if completed_at >= cutoff:
            continue

        job_id = job.get("job_id", "")

        # Collect local export files from job paths
        candidate_paths: list[Path] = []
        paths = job.get("paths") or {}
        for key in ("video_path", "export_manifest_path", "timeline_metadata_path", "srt_path", "vtt_path"):
            val = paths.get(key)
            if val:
                candidate_paths.append(Path(val))

        # Per-job generated_assets directory (local mode, stage 08)
        job_assets_dir = DATA_DIR / "jobs" / job_id
        if job_assets_dir.exists():
            candidate_paths.append(job_assets_dir)

        for path in candidate_paths:
            if not path.exists():
                continue
            if dry_run:
                skipped_paths.append(str(path))
            else:
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                    deleted_paths.append(str(path))
                except Exception:
                    skipped_paths.append(str(path))

    return {
        "dry_run": dry_run,
        "older_than_days": older_than_days,
        "cutoff": cutoff.isoformat(),
        "deleted_count": len(deleted_paths),
        "deleted_paths": deleted_paths,
        "skipped_count": len(skipped_paths),
        "skipped_paths": skipped_paths,
        "message": (
            f"Dry run — {len(skipped_paths)} path(s) would be deleted. Pass ?dry_run=false to apply."
            if dry_run else
            f"Deleted {len(deleted_paths)} path(s)."
        ),
    }
