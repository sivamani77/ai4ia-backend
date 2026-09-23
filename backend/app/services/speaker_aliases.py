"""Canonical speaker names.

Thin wrapper over the speaker registry (config/speaker_registry.json, see
app.services.speaker_registry). Used at extraction time
(04b_extract_speakers.py, 05_classify_fragments.py) and in the Postgres
backfill (scripts/migrate_fragments_to_postgres.py). To add or correct a
name, edit the registry build, not this module.
"""
from __future__ import annotations

from app.services.speaker_registry import get_registry


def canonicalize(name: str | None) -> str | None:
    """Canonical spelling for a known speaker, `name` unchanged if unknown,
    None for extractor placeholders such as "I" or "null"."""
    return get_registry().canonicalize(name)
