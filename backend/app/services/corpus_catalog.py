"""Chapter numbers and year range of the active corpus, for the resolve step."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger("ai4ia.services.corpus_catalog")

_TTL_SECONDS = 3600
_cache: tuple[float, "Catalog | None"] | None = None


@dataclass(frozen=True)
class Catalog:
    chapter_numbers: frozenset[int]
    first_year: int
    last_year: int


def build_catalog(records: list[dict]) -> Catalog | None:
    chapters = {int(m.group(1)) for r in records
                if (m := re.match(r"\s*Ch\s*(\d+)", str(r.get("primary_chapter") or "")))}
    years = set()
    for r in records:
        try:
            year = int(r.get("video_year") or 0)
        except (TypeError, ValueError):
            continue
        if 1900 < year < 2100:
            years.add(year)
    if not chapters or not years:
        return None
    return Catalog(frozenset(chapters), min(years), max(years))


def get_catalog() -> Catalog | None:
    """Cached for an hour. None if the corpus cannot be read; checks that need it are skipped."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < _TTL_SECONDS:
        return _cache[1]
    try:
        from app.pipeline.storyline_gen.retrieval import _load_corpus_from_chroma
        catalog = build_catalog(_load_corpus_from_chroma())
    except Exception:
        logger.warning("Corpus catalog unavailable; chapter and year checks skipped", exc_info=True)
        catalog = None
    _cache = (now, catalog)
    return catalog
