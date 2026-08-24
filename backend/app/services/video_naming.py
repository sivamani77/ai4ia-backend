"""Parsing for the "0182_IeXbR5L4EWE_Title Here" source_video filename
convention used throughout the ingest pipeline and archive.

Historically re-implemented independently in three places
(archive_service.infer_speaker_from_filename, the original
scripts/migrate_fragments_to_postgres.py, and 04b_extract_speakers.py's
title handling) using split("_", 2) — which mis-parses the 42 of 309
videos (in the catalog as of this writing) whose YouTube id itself
contains an underscore, e.g. "9_sJS37LjJ4": the id gets truncated to "9"
and its other half stays glued onto the front of the title. This is the
one canonical implementation; parse by YouTube's fixed 11-character id
length instead of splitting on "_".
"""
from __future__ import annotations

import re

_SOURCE_VIDEO_PREFIX_RE = re.compile(r"^\d+_")


def parse_source_video(source_video: str) -> tuple[str | None, str]:
    """Returns (video_id, title). source_video looks like
    '0182_IeXbR5L4EWE_Ashwini Kotaru - ...': a numeric index, then the
    video_id, then the title."""
    if not source_video:
        return None, ""
    m = _SOURCE_VIDEO_PREFIX_RE.match(source_video)
    if not m:
        return None, source_video
    rest = source_video[m.end():]
    video_id = rest[:11] or None
    title = rest[11:].lstrip("_").strip()
    return video_id, title
