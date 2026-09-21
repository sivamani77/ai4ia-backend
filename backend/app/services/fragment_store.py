"""Fragment store — reads the fragments table from Postgres (DATABASE_URL).

Public API is identical to the previous filesystem/in-memory version so all
callers (embedding_index, insight_service, archive_service) require no changes.
"""
import time
from typing import Any
from urllib.parse import quote

from app.db.repositories.fragment_repository import FragmentRepository

_CACHE_TTL = 60.0  # seconds

_speakers_cache: tuple[list, float] = ([], 0.0)


def stream_url_for(source_video: str | None) -> str | None:
    """Build a fragment's archive stream URL from its source video.

    Fragment rows have no stream_url column — it's derived, the same way
    archive_service/archive.py build it for video listings and fragment
    previews. Callers that build a fragment/clip dict for a citation or
    evidence pool must compute it here rather than reading a "stream_url"
    key off the raw fragment row, which is never set.
    """
    if not source_video:
        return None
    return f"/api/archive/videos/{quote(source_video, safe='')}/stream"


def reload() -> dict[str, int]:
    """No-op — data lives in Supabase, no cache to rebuild."""
    repo = FragmentRepository()
    return {"fragments": repo.count()}


def all_fragments() -> list[dict]:
    return [dict(r) for r in FragmentRepository().list_all()]


def get(fragment_id: str) -> dict | None:
    row = FragmentRepository().get(fragment_id)
    return dict(row) if row else None


def get_many(fragment_ids: list[str]) -> dict[str, dict]:
    """Batched lookup — one Supabase round trip for all ids, keyed by fragment_id."""
    rows = FragmentRepository().get_many(fragment_ids)
    return {row["fragment_id"]: dict(row) for row in rows}


def by_facet(facet: str, value: str) -> list[dict]:
    kwargs: dict[str, Any] = {}
    if facet == "theme":
        kwargs["theme"] = value
    elif facet == "chapter":
        kwargs["chapter"] = value
    elif facet == "speaker":
        kwargs["speaker"] = value
    elif facet == "year":
        try:
            kwargs["year"] = int(value)
        except (TypeError, ValueError):
            return []
    elif facet == "edition":
        kwargs["edition"] = value
    else:
        return []
    return [dict(r) for r in FragmentRepository().search(**kwargs)]


def list_fragment_ids_by_source_video(source_video: str) -> list[str]:
    return FragmentRepository().list_fragment_ids_by_source_video(source_video)


def delete_many(fragment_ids: list[str]) -> list[str]:
    return FragmentRepository().delete_many(fragment_ids)


def speakers_summary() -> list[dict]:
    """Aggregate speakers via DB group-by. Result cached for 60s."""
    global _speakers_cache
    data, ts = _speakers_cache
    if data and time.monotonic() - ts < _CACHE_TTL:
        return data
    data = FragmentRepository().speakers_aggregate()
    _speakers_cache = (data, time.monotonic())
    return data
