"""Cached, indexed in-memory store of all classified fragments.

Loads once (lazy), merges LLM-extracted speaker rosters and derived
year/edition/session facets, and builds facet indexes. Every intelligence
endpoint reads from here instead of re-walking 73 JSON files per request.
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.paths import CLASSIFIED_FRAGMENTS_DIR, RAW_VIDEO_DIR, SPEAKERS_DIR
from app.services.enrichment import derive_facets

_lock = Lock()
_state: dict[str, Any] = {"loaded": False}


def _safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _infer_topic_from_filename(name: str) -> str:
    stem = Path(name).stem.replace(".classified", "").replace(".info", "")
    parts = stem.split("_")
    human = "_".join(parts[2:]) if len(parts) > 2 else stem
    return re.sub(r"\s+", " ", human.replace("_-_", " - ").replace("_", " ")).strip()


def _load_speaker_roster(video_id: str, source_video: str) -> dict | None:
    if not SPEAKERS_DIR.exists():
        return None
    for key in (video_id, source_video):
        path = SPEAKERS_DIR / f"{key}.json"
        if path.exists():
            return _safe_json(path)
    return None


def _build() -> dict[str, Any]:
    fragments: list[dict[str, Any]] = []
    by_id: dict[str, dict] = {}
    idx = {k: defaultdict(list) for k in ("theme", "chapter", "speaker", "year", "edition")}

    if not CLASSIFIED_FRAGMENTS_DIR.exists():
        return {"loaded": True, "fragments": [], "by_id": {}, "index": idx}

    for fpath in sorted(CLASSIFIED_FRAGMENTS_DIR.glob("*.classified.json")):
        payload = _safe_json(fpath)
        if not isinstance(payload, dict):
            continue

        source_video = payload.get("source_video") or fpath.stem.replace(".classified", "")
        video_id = payload.get("video_id") or source_video

        info = _safe_json(RAW_VIDEO_DIR / f"{source_video}.info.json")
        title = (info or {}).get("title") or _infer_topic_from_filename(fpath.name)
        facets = derive_facets(title, source_video)

        roster = _load_speaker_roster(video_id, source_video)
        if roster:
            primary_speaker = roster.get("primary_speaker")
            speaker_candidates = roster.get("speakers", [])
            attribution = "video-level"
        else:
            primary_speaker = _infer_topic_from_filename(fpath.name)
            speaker_candidates = []
            attribution = "filename-derived"

        for frag in payload.get("fragments", []):
            item = dict(frag)
            item.setdefault("source_video", source_video)
            item.setdefault("video_id", video_id)
            item["video_title"] = title
            item["speaker"] = primary_speaker
            item["speaker_candidates"] = speaker_candidates
            item["speaker_attribution"] = attribution
            item["year"] = facets["year"]
            item["edition"] = facets["edition"]
            item["session_type"] = facets["session_type"]
            item["stream_url"] = f"/api/archive/videos/{source_video}/stream"
            item["fragment_id"] = item.get("fragment_id") or item.get("id")
            item["start"] = float(item.get("start") or 0)
            item["end"] = float(item.get("end") or 0)
            if not item.get("duration"):
                item["duration"] = max(item["end"] - item["start"], 0)

            fragments.append(item)
            fid = item.get("fragment_id")
            if fid:
                by_id[fid] = item

            pt = _name(item.get("primary_theme"), "theme")
            pc = _name(item.get("primary_chapter"), "chapter")
            if pt:
                idx["theme"][pt.lower()].append(item)
            if pc:
                idx["chapter"][pc.lower()].append(item)
            if primary_speaker:
                idx["speaker"][primary_speaker.lower()].append(item)
            if facets["year"]:
                idx["year"][str(facets["year"])].append(item)
            if facets["edition"]:
                idx["edition"][facets["edition"].lower()].append(item)

    return {"loaded": True, "fragments": fragments, "by_id": by_id, "index": idx}


def _name(tag, key: str) -> str | None:
    if tag is None:
        return None
    if isinstance(tag, str):
        return tag
    if isinstance(tag, dict):
        return tag.get(key) or tag.get("name") or tag.get("label") or tag.get(f"{key}_id")
    return str(tag)


def _ensure() -> dict[str, Any]:
    if not _state.get("loaded"):
        with _lock:
            if not _state.get("loaded"):
                _state.update(_build())
    return _state


def reload() -> dict[str, int]:
    """Force a rebuild (Admin: call after a pipeline run)."""
    with _lock:
        _state.clear()
        _state.update(_build())
    return {"fragments": len(_state["fragments"])}


def all_fragments() -> list[dict]:
    return _ensure()["fragments"]


def get(fragment_id: str) -> dict | None:
    return _ensure()["by_id"].get(fragment_id)


def by_facet(facet: str, value: str) -> list[dict]:
    return _ensure()["index"].get(facet, {}).get(str(value).lower(), [])


def speakers_summary() -> list[dict]:
    counts: dict[str, dict] = {}
    for f in all_fragments():
        sp = f.get("speaker")
        if not sp:
            continue
        rec = counts.setdefault(sp, {"speaker": sp, "fragment_count": 0, "videos": set(),
                                     "attribution": f.get("speaker_attribution")})
        rec["fragment_count"] += 1
        rec["videos"].add(f.get("source_video"))
    out = []
    for rec in counts.values():
        rec["video_count"] = len(rec.pop("videos"))
        out.append(rec)
    return sorted(out, key=lambda r: r["fragment_count"], reverse=True)
