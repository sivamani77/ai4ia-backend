"""
Media registry builder.

Scans completed job paths and produces data/jobs/{job_id}/media_registry.json,
which is the bridge between local files now and object storage later.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.job_store import load_job
from app.services.storage_service import (
    STORAGE_BACKEND,
    _guess_content_type,
    _safe_path,
    _LOCAL_JOBS_DIR,
    save_registry,
)


def _ts(p: Path) -> str | None:
    try:
        return datetime.utcfromtimestamp(p.stat().st_mtime).isoformat()
    except Exception:
        return None


def _entry(
    object_id: str,
    kind: str,
    local_path: str,
    filename: str | None = None,
    content_type: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    p = Path(local_path)
    fn = filename or p.name
    ct = content_type or _guess_content_type(p)
    size = p.stat().st_size if p.exists() else None
    created = _ts(p) if p.exists() else None
    return {
        "object_id": object_id,
        "kind": kind,
        "local_path": local_path,
        "filename": fn,
        "content_type": ct,
        "size_bytes": size,
        "created_at": created,
        **extra,
    }


def build_media_registry(job_id: str) -> dict[str, Any]:
    """
    Build a media registry for job_id by scanning job paths from Supabase
    and the asset manifest. Writes media_registry.json and returns it.
    """
    job = load_job(job_id)
    paths = job.get("paths", {})
    objects: dict[str, Any] = {}

    # --- Core exports ---
    vp = paths.get("video_path")
    if vp and Path(vp).exists():
        safe = _safe_path(vp)
        if safe:
            objects["video"] = _entry(
                "video", "video", str(safe),
                content_type="video/mp4",
            )

    srtp = paths.get("srt_path")
    if srtp and Path(srtp).exists():
        safe = _safe_path(srtp)
        if safe:
            objects["srt"] = _entry(
                "srt", "subtitle", str(safe),
                content_type="application/x-subrip",
                subtitle_format="srt",
            )

    vttp = paths.get("vtt_path")
    if vttp and Path(vttp).exists():
        safe = _safe_path(vttp)
        if safe:
            objects["vtt"] = _entry(
                "vtt", "subtitle", str(safe),
                content_type="text/vtt",
                subtitle_format="vtt",
            )

    emp = paths.get("export_manifest_path")
    if emp and Path(emp).exists():
        safe = _safe_path(emp)
        if safe:
            objects["manifest"] = _entry(
                "manifest", "manifest", str(safe),
                filename="export_manifest.json",
                content_type="application/json",
            )

    # --- Asset manifest: visuals, voiceovers, cards ---
    amp = paths.get("asset_manifest_path")
    if amp and Path(amp).exists():
        try:
            mdata = json.loads(Path(amp).read_text(encoding="utf-8"))
        except Exception:
            mdata = {}

        tl = mdata.get("export_timeline") or []
        for item in tl:
            seg_type = item.get("type") or item.get("segment_type") or ""
            sid = item.get("section_id", "")

            if seg_type == "narration_broll":
                # Visual asset
                v_path = (
                    item.get("render_visual_path")
                    or item.get("visual_path")
                    or item.get("asset_path")
                    or ""
                )
                if v_path:
                    safe = _safe_path(v_path)
                    if safe:
                        oid = f"visual_{sid}" if sid else f"visual_{safe.stem}"
                        objects[oid] = _entry(
                            oid,
                            "visual_asset",
                            str(safe),
                            section_id=sid,
                            visual_title=item.get("visual_title", ""),
                            visual_asset_type=item.get("visual_asset_type", ""),
                            visual_keywords=item.get("visual_keywords", []),
                        )

                # Voiceover audio
                a_path = item.get("audio_path") or item.get("voiceover_path") or ""
                if a_path:
                    safe = _safe_path(a_path)
                    if safe:
                        oid = f"voiceover_{sid}" if sid else f"voiceover_{safe.stem}"
                        objects[oid] = _entry(
                            oid,
                            "voiceover_audio",
                            str(safe),
                            section_id=sid,
                            voiceover_text=item.get("voiceover_text") or item.get("subtitle_text") or "",
                        )

        # Cards
        cards_data = mdata.get("cards") or {}
        tc = cards_data.get("title_card")
        if tc:
            safe = _safe_path(tc)
            if safe:
                objects["title_card"] = _entry(
                    "title_card", "card", str(safe),
                    card_type="title_card",
                )

        cc = cards_data.get("closing_card")
        if cc:
            safe = _safe_path(cc)
            if safe:
                objects["closing_card"] = _entry(
                    "closing_card", "card", str(safe),
                    card_type="closing_card",
                )

        for sc in (cards_data.get("section_cards") or []):
            if not isinstance(sc, dict):
                continue
            sc_path = sc.get("path", "")
            sc_id = sc.get("section_id", "")
            if sc_path:
                safe = _safe_path(sc_path)
                if safe:
                    oid = f"section_card_{sc_id}" if sc_id else f"section_card_{safe.stem}"
                    objects[oid] = _entry(
                        oid, "card", str(safe),
                        card_type="section_card",
                        section_id=sc_id,
                    )

    # Thumbnail: prefer title_card, else first visual_asset
    if "thumbnail" not in objects:
        if "title_card" in objects:
            objects["thumbnail"] = {**objects["title_card"], "object_id": "thumbnail", "kind": "thumbnail"}
        else:
            for oid, obj in objects.items():
                if obj.get("kind") == "visual_asset":
                    objects["thumbnail"] = {**obj, "object_id": "thumbnail", "kind": "thumbnail"}
                    break

    registry = {
        "job_id": job_id,
        "storage_backend": STORAGE_BACKEND,
        "objects": objects,
    }

    save_registry(job_id, registry)

    # Persist registry path back to Supabase
    from app.services.job_store import save_job
    reg_path = str(_LOCAL_JOBS_DIR / job_id / "media_registry.json")
    paths["media_registry_path"] = reg_path
    job["paths"] = paths
    save_job(job_id, job)

    return registry
