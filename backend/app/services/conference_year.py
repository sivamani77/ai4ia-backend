"""The conference year of a talk, the one year every feature should use.

Order: the curated map (config/conference_years.json, checked against
ai4iaconference.com), then an "AI4IA <year>" title, then the caller's
fallback (an upload year or a stored value).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from app.core.paths import CONFIG_DIR
from app.services.video_naming import parse_source_video

YEARS_PATH = CONFIG_DIR / "conference_years.json"
_TITLE_YEAR = re.compile(r"AI4IA[\s_\-]*?(20\d{2})", re.IGNORECASE)


@lru_cache(maxsize=1)
def known_years() -> dict[str, int]:
    """video_id -> conference year; empty when the file is missing."""
    try:
        data = json.loads(YEARS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: int(v) for k, v in (data.get("years") or {}).items()}


def as_year(value) -> int | None:
    """A plausible year as an int, else None ("" and 0 from Chroma included)."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 2000 < year < 2100 else None


def conference_year(video_id: str | None = None, source_video: str | None = None,
                    fallback=None) -> int | None:
    parsed_id, title = parse_source_video(source_video or "")
    for vid in (parsed_id, video_id):
        if vid and vid in known_years():
            return known_years()[vid]
    match = _TITLE_YEAR.search(title or source_video or "")
    if match:
        return int(match.group(1))
    return as_year(fallback)
