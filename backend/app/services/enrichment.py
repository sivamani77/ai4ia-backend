"""Derive structured facets (year, edition, session_type) from titles.

Region is intentionally NOT derived: for this archive the title lead is usually
a speaker name, not a place, so a region facet would be misleading. Speaker is
the reliable cross-cutting dimension (see fragment_store / 04b_extract_speakers).

`edition` captures the conference edition (e.g. "AI4IA 2025") for a future
cross-edition relay timeline; today the archive is effectively one edition.
"""
import re
from typing import Any

_YEAR_RE = re.compile(r"AI4IA[\s_\-]*?(\d{4})", re.IGNORECASE)
_ANY_YEAR_RE = re.compile(r"(20\d{2})")

_SESSION_KEYWORDS = {
    "satellite": "satellite_event",
    "panel": "panel",
    "keynote": "keynote",
    "workshop": "workshop",
    "regional": "regional_session",
    "opening": "opening",
    "closing": "closing",
}


def derive_year(title: str, source_video: str) -> int | None:
    for text in (title, source_video):
        m = _YEAR_RE.search(text or "")
        if m:
            return int(m.group(1))
    for text in (title, source_video):
        m = _ANY_YEAR_RE.search(text or "")
        if m:
            return int(m.group(1))
    return None


def derive_edition(title: str, source_video: str) -> str | None:
    """Conference edition, e.g. 'AI4IA 2025'. None if no edition marker present."""
    year = derive_year(title, source_video)
    return f"AI4IA {year}" if year else None


def derive_session_type(title: str, source_video: str) -> str:
    blob = f"{title} {source_video}".lower()
    for keyword, label in _SESSION_KEYWORDS.items():
        if keyword in blob:
            return label
    return "talk"


def derive_facets(title: str, source_video: str) -> dict[str, Any]:
    return {
        "year": derive_year(title, source_video),
        "edition": derive_edition(title, source_video),
        "session_type": derive_session_type(title, source_video),
    }
