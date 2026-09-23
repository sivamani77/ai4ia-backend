import json
import re
import time
from pathlib import Path
from typing import Any

from app.core.paths import RAW_VIDEO_DIR, CLASSIFIED_FRAGMENTS_DIR
from app.core.settings import settings
from app.core.taxonomy import THEMES, CHAPTERS, chapter_id_from_name
from app.db.repositories.fragment_repository import FragmentRepository
from app.services.video_naming import parse_source_video

_S3_RAW_PREFIX = "raw/youtube/ai4ia_channel/"

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


def _list_videos_from_s3() -> list[dict[str, Any]]:
    """List source videos by scanning the raw/ prefix in S3."""
    from app.services.storage_service import S3StorageService, _OBJECT_STORAGE_BUCKET, _OBJECT_STORAGE_REGION, _OBJECT_STORAGE_ACCESS_KEY, _OBJECT_STORAGE_SECRET_KEY, _OBJECT_STORAGE_ENDPOINT
    import boto3

    kwargs: dict[str, Any] = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT

    s3 = boto3.client("s3", **kwargs)
    bucket = _OBJECT_STORAGE_BUCKET

    paginator = s3.get_paginator("list_objects_v2")
    mp4_keys: list[str] = []
    info_map: dict[str, Any] = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=_S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".mp4"):
                mp4_keys.append(key)
            elif key.endswith(".info.json"):
                stem = Path(key).stem.replace(".info", "")
                try:
                    resp = s3.get_object(Bucket=bucket, Key=key)
                    info_map[stem] = json.loads(resp["Body"].read().decode("utf-8"))
                except Exception:
                    pass

    videos = []
    for key in sorted(mp4_keys):
        stem = Path(key).stem
        info = info_map.get(stem, {})
        # YouTube ids are always exactly 11 characters — parse by that
        # fixed length, not split("_"), which mis-parses the 42 of 309
        # ids that contain an underscore themselves (e.g. "9_sJS37LjJ4"
        # truncates to "9"). See video_naming.py's docstring; this is
        # what scripts/reconcile_archive_stores.py caught: those 42
        # videos got the wrong primary key in the videos table.
        parsed_id, _ = parse_source_video(stem)
        youtube_id = parsed_id or stem
        title = info.get("title") or infer_speaker_from_filename(stem)
        videos.append({
            "video_id": youtube_id,
            "source_video": stem,
            "filename": Path(key).name,
            "title": title,
            "speaker_or_topic": infer_speaker_from_filename(stem),
            "path": None,
            "thumbnail": None,
            "duration": info.get("duration"),
            "webpage_url": info.get("webpage_url"),
            "description": info.get("description") or None,
            "s3_key": key,
            "stream_url": f"/api/archive/videos/{youtube_id}/stream",
        })
    return videos


def _video_record_to_dict(r: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the dict shape callers expect (matching
    _list_videos_from_s3()'s output) from a videos-table row."""
    source_video = r.get("source_video") or ""
    video_id = r.get("video_id")
    return {
        "video_id": video_id,
        "source_video": source_video,
        "filename": f"{source_video}.mp4" if source_video else None,
        "title": r.get("title"),
        "speaker_or_topic": infer_speaker_from_filename(source_video) if source_video else "",
        "path": None,
        "thumbnail": None,
        "duration": r.get("duration"),
        "webpage_url": r.get("webpage_url"),
        "s3_key": r.get("s3_raw_key"),
        "stream_url": f"/api/archive/videos/{video_id}/stream",
    }


def list_videos() -> list[dict[str, Any]]:
    global _videos_cache
    data, ts = _videos_cache
    if data and time.monotonic() - ts < _CACHE_TTL:
        return data

    videos: list[dict[str, Any]] = []

    if settings.ai4ia_storage_backend == "s3":
        from app.db.repositories.video_repository import VideoRepository
        repo = VideoRepository()
        try:
            records = [dict(r) for r in repo.list_all()]
        except Exception:
            records = []

        if records:
            videos = [_video_record_to_dict(r) for r in records]
        else:
            # videos table not populated yet — scan S3 live (original
            # behavior) and opportunistically sync it in, so next call
            # reads from Postgres and scripts/reconcile_archive_stores.py
            # has something to compare against.
            try:
                videos = _list_videos_from_s3()
                repo.upsert_many(videos)
            except Exception:
                pass
    else:
        if RAW_VIDEO_DIR.exists():
            for path in sorted(RAW_VIDEO_DIR.glob("*.mp4")):
                stem = path.stem
                info = safe_read_json(RAW_VIDEO_DIR / f"{stem}.info.json")
                title = (info or {}).get("title") or stem
                parsed_id, _ = parse_source_video(stem)
                youtube_id = parsed_id or stem
                videos.append({
                    "video_id": youtube_id,
                    "source_video": stem,
                    "filename": path.name,
                    "title": title,
                    "speaker_or_topic": infer_speaker_from_filename(path.name),
                    "path": str(path),
                    "thumbnail": str(RAW_VIDEO_DIR / f"{stem}.jpg") if (RAW_VIDEO_DIR / f"{stem}.jpg").exists() else None,
                    "duration": (info or {}).get("duration"),
                    "webpage_url": (info or {}).get("webpage_url"),
                    "stream_url": f"/api/archive/videos/{youtube_id}/stream",
                })

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
