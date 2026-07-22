from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def job_to_generation_entry(job: dict[str, Any]) -> dict[str, Any]:
    """Shape a raw job metadata dict (from job_store) into the admin-facing
    generation entry: prompt/status plus whatever rendered media exists.

    Shared by the admin export archive (all users, completed jobs) and the
    per-user generation feed (one user, any status) so both surfaces agree
    on what a "generation" looks like.
    """
    from app.api.video_media import _get_registry, _media_list_entry

    job_id = job.get("job_id") or job.get("id")
    prompt = job.get("prompt") or (job.get("payload") or {}).get("prompt")

    try:
        reg = _get_registry(job_id)
        objects = {
            oid: _media_list_entry(job_id, oid, obj)
            for oid, obj in reg.get("objects", {}).items()
        }
    except HTTPException:
        objects = {}

    return {
        "job_id": job_id,
        "user_id": job.get("user_id"),
        "status": job.get("status"),
        "prompt": prompt,
        "target_duration": job.get("target_duration"),
        "current_stage": job.get("current_stage"),
        "progress_percent": job.get("progress_percent"),
        "message": job.get("message"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "video": objects.get("video"),
        "srt": objects.get("srt"),
        "vtt": objects.get("vtt"),
        "manifest": objects.get("manifest"),
        "thumbnail": objects.get("thumbnail"),
        "other_assets": [
            v for k, v in objects.items()
            if k not in ("video", "srt", "vtt", "manifest", "thumbnail")
        ],
    }
