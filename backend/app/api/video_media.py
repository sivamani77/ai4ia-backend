"""
Media streaming and download APIs for video jobs.

GET /api/video-jobs/{job_id}/video/stream
GET /api/video-jobs/{job_id}/video/download
GET /api/video-jobs/{job_id}/video/signed-url
GET /api/video-jobs/{job_id}/subtitles/{format}
GET /api/video-jobs/{job_id}/manifest
GET /api/video-jobs/{job_id}/timeline
GET /api/video-jobs/{job_id}/media
GET /api/video-jobs/{job_id}/thumbnail
GET /api/video-jobs/{job_id}/assets/{asset_id}/stream
GET /api/video-jobs/{job_id}/assets/{asset_id}/download
GET /api/video-jobs/{job_id}/assets/{asset_id}/metadata
GET /api/video-jobs/{job_id}/assets/{asset_id}/signed-url
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import re

from app.core.auth import CurrentUser, get_current_user
from app.core.paths import EXPORTS_DIR
from app.services.storage_service import (
    LocalStorageService,
    StorageObject,
    _safe_path,
    get_storage_service,
    load_registry,
    _OBJECT_STORAGE_BUCKET,
    _OBJECT_STORAGE_REGION,
    _OBJECT_STORAGE_ACCESS_KEY,
    _OBJECT_STORAGE_SECRET_KEY,
    _OBJECT_STORAGE_ENDPOINT,
    STORAGE_BACKEND,
)
from app.services.media_registry import build_media_registry
from app.services.video_job_runner import load_job

_PRESIGNED_EXPIRY = 3600  # 1 hour


def _make_presigned_url(object_key: str) -> str:
    """Generate an S3 presigned GET URL for the given object key."""
    import boto3
    from botocore.config import Config
    kwargs: dict[str, Any] = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
        kwargs["config"] = Config(signature_version="s3v4")
    else:
        # Sign for the regional host: opt-in regions (ap-south-2) reject the global one.
        kwargs["config"] = Config(signature_version="s3v4", s3={"addressing_style": "virtual"})
    s3 = boto3.client("s3", **kwargs)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": _OBJECT_STORAGE_BUCKET, "Key": object_key},
        ExpiresIn=_PRESIGNED_EXPIRY,
    )

router = APIRouter(prefix="/video-jobs", tags=["video-media"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_job(job_id: str) -> dict[str, Any]:
    try:
        return load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


def _assert_owns(job: dict[str, Any], current_user: CurrentUser) -> None:
    """Return 404 (not 403) to avoid leaking job existence to other users.

    Admins bypass the ownership check — this is how the admin per-user
    generations feed (app/api/admin_users.py) actually views/streams a
    user's rendered video via the same stream_url a regular user gets.
    """
    if current_user.persona == "admin":
        return
    # A job with no owner is treated as someone else's.
    if not job.get("user_id") or job["user_id"] != current_user.id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job['job_id']}")


def _require_owned_job(job_id: str, current_user: CurrentUser) -> dict[str, Any]:
    job = _require_job(job_id)
    _assert_owns(job, current_user)
    return job


def _get_registry(job_id: str) -> dict[str, Any]:
    """Load or lazily build the media registry."""
    try:
        storage = get_storage_service()
        if hasattr(storage, "_load_registry"):
            return storage._load_registry(job_id)
        return load_registry(job_id)
    except FileNotFoundError:
        pass
    # Try to build it on-demand for already-completed jobs
    job = _require_job(job_id)
    if job.get("status") in ("completed", "failed"):
        try:
            return build_media_registry(job_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=f"Media registry unavailable: {exc}")
    raise HTTPException(status_code=404, detail="Media not yet available — job still running")


def _resolve(job_id: str, object_id: str) -> StorageObject:
    _get_registry(job_id)  # ensures registry exists (lazy-builds for completed jobs)
    storage = get_storage_service()
    try:
        return storage.resolve(job_id, object_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail=f"Asset '{object_id}' not found")


def _file_response(obj: StorageObject, disposition: str = "inline") -> Response:
    storage = get_storage_service()
    local = storage.get_local_path(obj)
    if local and local.exists():
        return FileResponse(
            path=str(local),
            media_type=obj.content_type,
            filename=obj.filename if disposition == "attachment" else None,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'{disposition}; filename="{obj.filename}"',
            },
        )
    # S3 or missing local file — stream via storage abstraction
    if not storage.exists(obj):
        raise HTTPException(status_code=404, detail=f"File not found: {obj.filename}")
    headers = {"Content-Disposition": f'{disposition}; filename="{obj.filename}"'}
    if obj.size_bytes:
        headers["Content-Length"] = str(obj.size_bytes)
    return StreamingResponse(
        storage.open_stream(obj),
        media_type=obj.content_type,
        headers=headers,
    )


def parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Return the inclusive (start, end) of a single byte range, or raise 416."""
    unsatisfiable = HTTPException(
        status_code=416,
        detail="Range Not Satisfiable",
        headers={"Content-Range": f"bytes */{file_size}"},
    )
    try:
        unit, rng = range_header.split("=", 1)
        if unit.strip().lower() != "bytes":
            raise ValueError("unsupported unit")
        start_str, end_str = (s.strip() for s in rng.split("-", 1))
        if not start_str:
            # Suffix range: the last N bytes.
            suffix = int(end_str)
            if suffix <= 0:
                raise unsatisfiable
            return max(file_size - suffix, 0), file_size - 1
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")
    if start < 0 or start >= file_size or start > end:
        raise unsatisfiable
    return start, min(end, file_size - 1)


def _s3_range_response(obj: StorageObject, request: Request) -> Response:
    """Stream an S3 object with proper HTTP Range support."""
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
    s3 = boto3.client("s3", **kwargs)
    s3_key = obj.object_key

    try:
        head = s3.head_object(Bucket=_OBJECT_STORAGE_BUCKET, Key=s3_key)
        file_size = head["ContentLength"]
    except Exception:
        raise HTTPException(status_code=404, detail=f"File not found in S3: {obj.filename}")

    range_header = request.headers.get("range")
    if range_header:
        start, end = parse_byte_range(range_header, file_size)
        length = end - start + 1

        def _stream():
            resp = s3.get_object(
                Bucket=_OBJECT_STORAGE_BUCKET, Key=s3_key,
                Range=f"bytes={start}-{end}",
            )
            body = resp["Body"]
            while True:
                chunk = body.read(65536)
                if not chunk:
                    break
                yield chunk

        return StreamingResponse(
            _stream(), status_code=206, media_type=obj.content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            },
        )

    def _stream_full():
        resp = s3.get_object(Bucket=_OBJECT_STORAGE_BUCKET, Key=s3_key)
        body = resp["Body"]
        while True:
            chunk = body.read(65536)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(
        _stream_full(), status_code=200, media_type=obj.content_type,
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


def _range_response(obj: StorageObject, request: Request) -> Response:
    """
    Handle HTTP Range requests for binary files (video, audio).
    """
    storage = get_storage_service()
    local = storage.get_local_path(obj)
    if not local or not local.exists():
        # S3 path — use proper byte-range streaming
        if not storage.exists(obj):
            raise HTTPException(status_code=404, detail=f"File not found: {obj.filename}")
        if obj.object_key:
            return _s3_range_response(obj, request)
        # fallback: full stream (no range)
        headers: dict[str, str] = {"Accept-Ranges": "none"}
        if obj.size_bytes:
            headers["Content-Length"] = str(obj.size_bytes)
        return StreamingResponse(
            storage.open_stream(obj),
            media_type=obj.content_type,
            headers=headers,
        )

    file_size = local.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        # No range — stream full file via FileResponse (handles large files efficiently)
        return FileResponse(
            path=str(local),
            media_type=obj.content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    start, end = parse_byte_range(range_header, file_size)
    length = end - start + 1

    def _iter():
        with local.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        _iter(),
        status_code=206,
        media_type=obj.content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        },
    )


def _media_list_entry(job_id: str, oid: str, obj_dict: dict[str, Any]) -> dict[str, Any]:
    base = f"/api/video-jobs/{job_id}"
    kind = obj_dict.get("kind", "")
    entry: dict[str, Any] = {
        "asset_id": oid,
        "kind": kind,
        "filename": obj_dict.get("filename", ""),
        "content_type": obj_dict.get("content_type", ""),
        "size_bytes": obj_dict.get("size_bytes"),
    }
    # Passthrough extra metadata fields
    for k in ("section_id", "visual_title", "visual_asset_type", "visual_keywords",
              "voiceover_text", "card_type", "subtitle_format"):
        if k in obj_dict:
            entry[k] = obj_dict[k]

    if kind == "video":
        entry["stream_url"] = f"{base}/video/stream"
        entry["download_url"] = f"{base}/video/download"
    elif kind == "subtitle":
        fmt = obj_dict.get("subtitle_format") or oid
        entry["stream_url"] = f"{base}/subtitles/{fmt}"
    elif kind == "manifest":
        entry["stream_url"] = f"{base}/manifest"
    elif kind in ("visual_asset", "voiceover_audio", "card", "thumbnail"):
        entry["stream_url"] = f"{base}/assets/{oid}/stream"
        entry["download_url"] = f"{base}/assets/{oid}/download"
    return entry


# ---------------------------------------------------------------------------
# Video stream / download
# ---------------------------------------------------------------------------

@router.get("/{job_id}/video/stream", operation_id="video_jobs_stream_video")
async def stream_video(job_id: str, request: Request, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "video")
    return _range_response(obj, request)


@router.head("/{job_id}/video/stream", include_in_schema=False, operation_id="video_jobs_head_stream_video")
async def stream_video_head(job_id: str, request: Request, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "video")
    return _range_response(obj, request)


@router.get("/{job_id}/video/download", operation_id="video_jobs_download_video")
async def download_video(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "video")
    return _file_response(obj, disposition="attachment")


@router.head("/{job_id}/video/download", include_in_schema=False, operation_id="video_jobs_head_download_video")
async def download_video_head(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "video")
    return _file_response(obj, disposition="attachment")


@router.get("/{job_id}/video/signed-url")
async def video_signed_url(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "video")
    if STORAGE_BACKEND == "s3" and obj.object_key:
        url = _make_presigned_url(obj.object_key)
        return {"url": url, "expires_in": _PRESIGNED_EXPIRY, "storage_backend": "s3"}
    return {
        "url": f"/api/video-jobs/{job_id}/video/stream",
        "expires_in": None,
        "storage_backend": "local",
    }


# ---------------------------------------------------------------------------
# Progressive HLS streaming (stage 09 writes here live, while still running —
# see backend/app/pipeline/stage_09_export.py and the HLSStreamWriter class in
# scripts/09_export_video.py).
#
# Two serving modes, matching how the worker writes the files:
#   - local/EBS (STORAGE_BACKEND != "s3"): read straight off disk, same as
#     the worker wrote it — only valid when the API process and the Celery
#     worker share a filesystem.
#   - S3 (STORAGE_BACKEND == "s3"): the worker mirrors each new segment and
#     the playlist/subs to S3 as they're written (see
#     stage_09_export.py::_watch_hls_segments) specifically so this works
#     when the API and worker are separate hosts/containers with no shared
#     disk — the common case, and the reason local-only serving silently
#     returned nothing in that deployment.
#
# Read directly during the render either way — not via the media registry,
# which is only built once the job completes.
# ---------------------------------------------------------------------------

_HLS_SEGMENT_NAME_RE = re.compile(r"^(seg_\d{4}\.ts|subs\.vtt)$")


def _hls_dir(job_id: str) -> Path:
    return EXPORTS_DIR / "hls" / job_id


def _hls_s3_client():
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
    return boto3.client("s3", **kwargs), _OBJECT_STORAGE_BUCKET


def _hls_s3_get(job_id: str, filename: str) -> bytes:
    """Fetch an HLS file from S3. Raises HTTPException(404) if missing."""
    from botocore.exceptions import ClientError
    s3, bucket = _hls_s3_client()
    key = f"hls/{job_id}/{filename}"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/{job_id}/video/hls/master.m3u8", operation_id="video_jobs_hls_playlist")
async def hls_playlist(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    if STORAGE_BACKEND == "s3":
        body = _hls_s3_get(job_id, "master.m3u8")
        return Response(content=body, media_type="application/vnd.apple.mpegurl",
                         headers={"Cache-Control": "no-store"})
    playlist_path = _hls_dir(job_id) / "master.m3u8"
    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="Stream not started yet — no segments rendered.")
    return FileResponse(
        path=str(playlist_path),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{job_id}/video/hls/{filename}", operation_id="video_jobs_hls_segment")
async def hls_segment(job_id: str, filename: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    if not _HLS_SEGMENT_NAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Not found")
    media_type = "video/mp2t" if filename.endswith(".ts") else "text/vtt"
    if STORAGE_BACKEND == "s3":
        body = _hls_s3_get(job_id, filename)
        return Response(content=body, media_type=media_type, headers={"Cache-Control": "no-store"})
    file_path = _hls_dir(job_id) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path=str(file_path), media_type=media_type, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

def _vtt_to_srt(vtt_text: str) -> str:
    """Convert WebVTT text to SRT format."""
    lines = vtt_text.splitlines()
    srt_lines = []
    cue_idx = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            i += 1
            continue
        if "-->" in line:
            ts_line = line.replace(".", ",")
            srt_lines.append(str(cue_idx))
            srt_lines.append(ts_line)
            cue_idx += 1
            i += 1
            while i < len(lines) and lines[i].strip():
                srt_lines.append(lines[i])
                i += 1
            srt_lines.append("")
        else:
            i += 1
    return "\n".join(srt_lines)


@router.get("/{job_id}/subtitles/{fmt}")
async def get_subtitles(job_id: str, fmt: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    fmt = fmt.lower()
    if fmt not in ("srt", "vtt"):
        raise HTTPException(status_code=400, detail="Unsupported subtitle format. Use 'srt' or 'vtt'.")
    try:
        obj = _resolve(job_id, fmt)
        return _file_response(obj, disposition="inline")
    except HTTPException:
        # Fall back to live streaming HLS subtitles (subs.vtt) generated during Stage 09 render
        vtt_bytes: bytes | None = None
        if STORAGE_BACKEND == "s3":
            try:
                vtt_bytes = _hls_s3_get(job_id, "subs.vtt")
            except Exception:
                pass
        else:
            file_path = _hls_dir(job_id) / "subs.vtt"
            if file_path.exists():
                vtt_bytes = file_path.read_bytes()

        if not vtt_bytes:
            raise HTTPException(status_code=404, detail="Subtitle format not available yet.")

        vtt_text = vtt_bytes.decode("utf-8", errors="replace")
        if fmt == "vtt":
            return Response(content=vtt_text, media_type="text/vtt", headers={"Cache-Control": "no-cache"})
        else:
            srt_text = _vtt_to_srt(vtt_text)
            return Response(content=srt_text, media_type="application/x-subrip", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Export manifest
# ---------------------------------------------------------------------------

@router.get("/{job_id}/manifest")
async def get_manifest(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, "manifest")
    storage = get_storage_service()
    try:
        local = storage.get_local_path(obj)
        if local and local.exists():
            data = json.loads(local.read_text(encoding="utf-8"))
        else:
            raw = b"".join(storage.open_stream(obj))
            data = json.loads(raw.decode("utf-8"))
        return JSONResponse(content=data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read manifest: {exc}")


@router.get("/{job_id}/timeline")
async def get_timeline_metadata(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Render-timeline sidecar for this job's exported video: per-segment
    speaker, source-video year, topic, and placement (timeline_start/end)
    in the final render."""
    job = _require_owned_job(job_id, current_user)
    paths = job.get("paths") or {}
    raw_path = paths.get("timeline_metadata_path")
    if not raw_path:
        raise HTTPException(
            status_code=404,
            detail="Timeline metadata not available for this job (video not exported yet, "
                   "or rendered before timeline sidecars were introduced)",
        )

    local = Path(raw_path)
    try:
        if local.exists():
            data = json.loads(local.read_text(encoding="utf-8"))
        else:
            from app.pipeline.s3_helpers import use_s3, _s3, _bucket
            if not use_s3():
                raise HTTPException(status_code=404, detail="Timeline metadata file not found")
            key = f"exports/{local.name}"
            resp = _s3().get_object(Bucket=_bucket(), Key=key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
        return JSONResponse(content={"job_id": job_id, **data})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read timeline metadata: {exc}")


# ---------------------------------------------------------------------------
# Media registry listing
# ---------------------------------------------------------------------------

@router.get("/{job_id}/media")
async def list_media(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    reg = _get_registry(job_id)
    objects_list = [
        _media_list_entry(job_id, oid, obj)
        for oid, obj in reg.get("objects", {}).items()
        if oid != "thumbnail"  # expose thumbnail via /thumbnail endpoint
    ]
    return {
        "job_id": job_id,
        "storage_backend": reg.get("storage_backend", "local"),
        "objects": objects_list,
    }


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

@router.get("/{job_id}/thumbnail", operation_id="video_jobs_get_thumbnail")
async def get_thumbnail(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    reg = _get_registry(job_id)
    objects = reg.get("objects", {})

    # Priority: thumbnail → title_card → first visual_asset
    for candidate_id in ("thumbnail", "title_card"):
        if candidate_id in objects:
            try:
                obj = _resolve(job_id, candidate_id)
                return _file_response(obj, disposition="inline")
            except Exception:
                pass

    for oid, entry in objects.items():
        if entry.get("kind") == "visual_asset":
            try:
                obj = _resolve(job_id, oid)
                return _file_response(obj, disposition="inline")
            except Exception:
                continue

    raise HTTPException(status_code=404, detail="No thumbnail available")


@router.head("/{job_id}/thumbnail", include_in_schema=False, operation_id="video_jobs_head_thumbnail")
async def get_thumbnail_head(job_id: str, current_user: CurrentUser = Depends(get_current_user)):
    return await get_thumbnail(job_id, current_user)


# ---------------------------------------------------------------------------
# Generic asset stream / download / metadata / signed-url
# ---------------------------------------------------------------------------

@router.get("/{job_id}/assets/{asset_id}/stream", operation_id="video_jobs_stream_asset")
async def stream_asset(job_id: str, asset_id: str, request: Request, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, asset_id)
    # Use range support for audio/video; plain FileResponse for images/json
    if obj.content_type.startswith(("audio/", "video/")):
        return _range_response(obj, request)
    return _file_response(obj, disposition="inline")


@router.head("/{job_id}/assets/{asset_id}/stream", include_in_schema=False, operation_id="video_jobs_head_stream_asset")
async def stream_asset_head(job_id: str, asset_id: str, request: Request, current_user: CurrentUser = Depends(get_current_user)):
    return await stream_asset(job_id, asset_id, request, current_user)


@router.get("/{job_id}/assets/{asset_id}/download")
async def download_asset(job_id: str, asset_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, asset_id)
    return _file_response(obj, disposition="attachment")


@router.get("/{job_id}/assets/{asset_id}/metadata")
async def asset_metadata(job_id: str, asset_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, asset_id)
    base = f"/api/video-jobs/{job_id}"
    return {
        "asset_id": obj.object_id,
        "job_id": obj.job_id,
        "kind": obj.kind,
        "filename": obj.filename,
        "content_type": obj.content_type,
        "size_bytes": obj.size_bytes,
        "storage_backend": obj.storage_backend,
        "stream_url": f"{base}/assets/{asset_id}/stream",
        "download_url": f"{base}/assets/{asset_id}/download",
        **obj.metadata,
    }


@router.get("/{job_id}/assets/{asset_id}/signed-url")
async def asset_signed_url(job_id: str, asset_id: str, current_user: CurrentUser = Depends(get_current_user)):
    _require_owned_job(job_id, current_user)
    obj = _resolve(job_id, asset_id)
    if STORAGE_BACKEND == "s3" and obj.object_key:
        url = _make_presigned_url(obj.object_key)
        return {"url": url, "expires_in": _PRESIGNED_EXPIRY, "storage_backend": "s3"}
    return {
        "url": f"/api/video-jobs/{job_id}/assets/{asset_id}/stream",
        "expires_in": None,
        "storage_backend": "local",
    }
