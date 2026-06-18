import json
import re
from pathlib import Path
from typing import Any

from app.core.paths import RAW_VIDEO_DIR, CLASSIFIED_FRAGMENTS_DIR
from app.core.taxonomy import THEMES, CHAPTERS, chapter_id_from_name


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
    records = []

    if not CLASSIFIED_FRAGMENTS_DIR.exists():
        return records

    for file_path in sorted(CLASSIFIED_FRAGMENTS_DIR.glob("*.classified.json")):
        payload = safe_read_json(file_path)

        if not isinstance(payload, dict):
            continue

        source_video = payload.get("source_video") or file_path.stem.replace(".classified", "")
        video_title = payload.get("video_title") or infer_speaker_from_filename(file_path.name)
        speaker = payload.get("speaker") or infer_speaker_from_filename(file_path.name)

        fragments = payload.get("fragments", [])

        for fragment in fragments:
            item = dict(fragment)

            item.setdefault("source_video", source_video)
            item.setdefault("video_title", video_title)
            item.setdefault("speaker", speaker)

            item["classified_file"] = str(file_path)
            item["source_video_file"] = f"{source_video}.mp4"
            item["preview_url"] = f"/api/archive/videos/{source_video}/stream"

            item["fragment_id"] = item.get("fragment_id") or item.get("id")
            item["start"] = item.get("start") or item.get("start_time") or 0
            item["end"] = item.get("end") or item.get("end_time") or 0

            try:
                item["duration"] = item.get("duration") or max(float(item["end"]) - float(item["start"]), 0)
            except Exception:
                item["duration"] = item.get("duration") or 0

            item["primary_theme"] = item.get("primary_theme") or item.get("theme")
            item["primary_chapter"] = item.get("primary_chapter") or item.get("chapter")
            item["theme_tags"] = item.get("theme_tags", [])
            item["chapter_tags"] = item.get("chapter_tags", [])
            item["clip_usage_candidates"] = item.get("clip_usage_candidates", [])

            records.append(item)

    return records


def list_videos() -> list[dict[str, Any]]:
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

        videos.append(
            {
                "video_id": stem,
                "filename": path.name,
                "title": title,
                "speaker_or_topic": infer_speaker_from_filename(path.name),
                "path": str(path),
                "thumbnail": str(RAW_VIDEO_DIR / f"{stem}.jpg") if (RAW_VIDEO_DIR / f"{stem}.jpg").exists() else None,
                "duration": duration,
                "webpage_url": webpage_url,
                "stream_url": f"/api/archive/videos/{stem}/stream",
            }
        )

    return videos


def get_video(video_id: str):
    for video in list_videos():
        if video["video_id"] == video_id:
            return video
    return None


def get_fragment(fragment_id: str):
    for fragment in load_all_classified_fragments():
        if fragment.get("fragment_id") == fragment_id:
            return fragment
    return None


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

    Each theme carries its theme_id (T1-T7) and the list of chapters that belong
    to it (each with chapter_id Ch 1-Ch 25 and its own count). The frontend uses
    this to render the tree and know exactly which chapter sits under which theme.
    Counts come from each fragment's primary_chapter, rolled up to its theme.
    """
    chapter_counts: dict[str, int] = {}
    for item in load_all_classified_fragments():
        cid = chapter_id_from_name(extract_tag_name(item.get("primary_chapter"), "chapter") or "")
        if cid:
            chapter_counts[cid] = chapter_counts.get(cid, 0) + 1

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

    return {
        "themes": themes_out,
        "total_themes": len(themes_out),
        "total_chapters": len(CHAPTERS),
    }


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
    query_l = (query or "").lower().strip()
    theme_l = (theme or "").lower().strip()
    chapter_l = (chapter or "").lower().strip()
    speaker_l = (speaker or "").lower().strip()
    source_video_l = (source_video or "").lower().strip()
    clip_usage_l = (clip_usage or "").lower().strip()

    results = []

    for item in load_all_classified_fragments():
        haystack = " ".join(
            [
                normalize_text(item.get("fragment_id")),
                normalize_text(item.get("text")),
                normalize_text(item.get("summary")),
                normalize_text(item.get("keywords")),
                normalize_text(item.get("primary_theme")),
                normalize_text(item.get("primary_chapter")),
                normalize_text(item.get("theme_tags")),
                normalize_text(item.get("chapter_tags")),
                normalize_text(item.get("speaker")),
                normalize_text(item.get("video_title")),
                normalize_text(item.get("source_video")),
                normalize_text(item.get("clip_usage_candidates")),
            ]
        ).lower()

        if query_l and query_l not in haystack:
            continue

        if theme_l and theme_l not in haystack:
            continue

        if chapter_l and chapter_l not in haystack:
            continue

        if speaker_l and speaker_l not in haystack:
            continue

        if source_video_l and source_video_l not in normalize_text(item.get("source_video")).lower():
            continue

        if clip_usage_l:
            usage_text = normalize_text(item.get("clip_usage_candidates")).lower()
            if clip_usage_l not in usage_text:
                continue

        duration = float(item.get("duration") or 0)
        start = float(item.get("start") or 0)
        end = float(item.get("end") or 0)

        if min_duration is not None and duration < min_duration:
            continue

        if max_duration is not None and duration > max_duration:
            continue

        if start_after is not None and start < start_after:
            continue

        if end_before is not None and end > end_before:
            continue

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
