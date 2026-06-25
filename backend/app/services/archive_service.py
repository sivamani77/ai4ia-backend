import json
import re
import time
from pathlib import Path
from typing import Any

from app.core.paths import RAW_VIDEO_DIR, CLASSIFIED_FRAGMENTS_DIR
from app.core.taxonomy import THEMES, CHAPTERS, chapter_id_from_name
from app.db.repositories.fragment_repository import FragmentRepository

_CACHE_TTL = 60.0  # seconds

_themes_cache: tuple[dict, float] = ({}, 0.0)
_videos_cache: tuple[list, float] = ([], 0.0)


def safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def infer_speaker_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = stem.replace(".classified", "")
    parts = stem.split("_")

    if len(parts) > 2:
        human_part = "_".join(parts[2:])
    else:
        human_part = stem

    human_part = human_part.replace("_-_", " - ")
    human_part = human_part.replace("_", " ")
    human_part = re.sub(r"\s+", " ", human_part).strip()

    return human_part


def load_all_classified_fragments() -> list[dict[str, Any]]:
    return [dict(r) for r in FragmentRepository().list_all()]


def list_videos() -> list[dict[str, Any]]:
    global _videos_cache
    data, ts = _videos_cache
    if data and time.monotonic() - ts < _CACHE_TTL:
        return data

    videos = []

    if not RAW_VIDEO_DIR.exists():
        return videos

    for path in sorted(RAW_VIDEO_DIR.glob("*.mp4")):
        stem = path.stem
        info_path = RAW_VIDEO_DIR / f"{stem}.info.json"
        info = safe_read_json(info_path)

        title = stem
        duration = None
        webpage_url = None

        if isinstance(info, dict):
            title = info.get("title") or stem
            duration = info.get("duration")
            webpage_url = info.get("webpage_url")

        # Extract the YouTube ID (second _-delimited part: 0139_<youtube_id>_title)
        stem_parts = stem.split("_")
        youtube_id = stem_parts[1] if len(stem_parts) >= 2 else stem

        videos.append(
            {
                "video_id": youtube_id,
                "source_video": stem,
                "filename": path.name,
                "title": title,
                "speaker_or_topic": infer_speaker_from_filename(path.name),
                "path": str(path),
                "thumbnail": str(RAW_VIDEO_DIR / f"{stem}.jpg") if (RAW_VIDEO_DIR / f"{stem}.jpg").exists() else None,
                "duration": duration,
                "webpage_url": webpage_url,
                "stream_url": f"/api/archive/videos/{youtube_id}/stream",
            }
        )

    _videos_cache = (videos, time.monotonic())
    return videos


def get_video(video_id: str):
    for video in list_videos():
        if video["video_id"] == video_id or video.get("source_video") == video_id:
            return video
    return None


def get_fragment(fragment_id: str):
    row = FragmentRepository().get(fragment_id)
    return dict(row) if row else None


def extract_tag_name(tag, key_name: str) -> str | None:
    """
    Supports both:
    - plain string tags
    - dict tags like {"theme": "..."} or {"chapter": "..."}
    """
    if tag is None:
        return None

    if isinstance(tag, str):
        return tag

    if isinstance(tag, dict):
        return (
            tag.get(key_name)
            or tag.get("name")
            or tag.get("label")
            or tag.get("theme")
            or tag.get("chapter")
            or tag.get("theme_id")
            or tag.get("chapter_id")
        )

    return str(tag)


def get_themes_and_chapters() -> dict[str, Any]:
    """Nested theme -> chapters hierarchy with per-node fragment counts.

    Counts fetched via DB aggregate (not a full table scan). Cached 60s.
    """
    global _themes_cache
    cached, ts = _themes_cache
    if cached and time.monotonic() - ts < _CACHE_TTL:
        return cached

    # DB aggregate — only chapter name + count, no full rows loaded
    raw_counts = FragmentRepository().chapter_counts()
    chapter_counts: dict[str, int] = {}
    for chapter_name, count in raw_counts.items():
        cid = chapter_id_from_name(chapter_name or "")
        if cid:
            chapter_counts[cid] = chapter_counts.get(cid, 0) + count

    themes_out = []
    for tid, theme in THEMES.items():
        chapters_out = []
        theme_total = 0
        for cid in theme["chapters"]:
            count = chapter_counts.get(cid, 0)
            theme_total += count
            chapters_out.append({
                "chapter_id": cid,
                "name": CHAPTERS[cid]["name"],
                "fragment_count": count,
            })
        themes_out.append({
            "theme_id": tid,
            "name": theme["name"],
            "fragment_count": theme_total,
            "chapters": chapters_out,
        })

    result = {
        "themes": themes_out,
        "total_themes": len(themes_out),
        "total_chapters": len(CHAPTERS),
    }
    _themes_cache = (result, time.monotonic())
    return result


def search_fragments(
    query: str | None = None,
    theme: str | None = None,
    chapter: str | None = None,
    speaker: str | None = None,
    source_video: str | None = None,
    clip_usage: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    start_after: float | None = None,
    end_before: float | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    repo = FragmentRepository()
    rows = repo.search(
        query=query or None,
        theme=theme or None,
        chapter=chapter or None,
        speaker=speaker or None,
        source_video=source_video or None,
        min_duration=min_duration,
        max_duration=max_duration,
        start_after=start_after,
        end_before=end_before,
        limit=max(1, min(limit, 200)),
    )

    # clip_usage filter (stored as jsonb array — done in Python)
    clip_usage_l = (clip_usage or "").lower().strip()
    if clip_usage_l:
        rows = [
            r for r in rows
            if clip_usage_l in normalize_text(r.get("clip_usage_candidates")).lower()
        ]

    # Scoring (mirrors old logic so ranking is preserved)
    query_l = (query or "").lower().strip()
    theme_l = (theme or "").lower().strip()
    chapter_l = (chapter or "").lower().strip()
    speaker_l = (speaker or "").lower().strip()

    results = []
    for item in rows:
        item = dict(item)
        haystack = " ".join(filter(None, [
            item.get("text", ""), item.get("summary", ""),
            item.get("primary_theme", ""), item.get("primary_chapter", ""),
            item.get("speaker", ""),
            " ".join(str(k) for k in (item.get("keywords") or [])),
        ])).lower()
        score = 0
        if query_l and query_l in haystack:
            score += 5
        if theme_l and theme_l in haystack:
            score += 3
        if chapter_l and chapter_l in haystack:
            score += 3
        if speaker_l and speaker_l in haystack:
            score += 2
        if clip_usage_l and clip_usage_l in haystack:
            score += 2
        item["_search_score"] = score
        results.append(item)

    results.sort(
        key=lambda x: (
            x.get("_search_score", 0),
            x.get("quote_strength", 0) or 0,
            x.get("clarity_score", 0) or 0,
        ),
        reverse=True,
    )

    return results[: max(1, min(limit, 200))]
