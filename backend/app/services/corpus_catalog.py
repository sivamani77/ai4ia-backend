"""Chapter numbers, year range and talk titles of the active corpus, for the resolve step."""
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
    titles: frozenset[str] = frozenset()  # normalized talk titles, without the speaker prefix

    def names_talk(self, prompt: str) -> bool:
        """The prompt is a talk title, or quotes one of four words or more."""
        text = _normalize(prompt)
        return bool(text) and any(text == t or (len(t.split()) >= 4 and t in text) for t in self.titles)


def _normalize(text: str) -> str:
    from app.pipeline.storyline_gen.text_utils import normalize_name

    return " ".join(normalize_name(text).split())


def talk_title(source_video: str) -> str:
    """The talk's own title: '0117_<id>_Speaker - Title' gives 'Title'."""
    from app.services.video_naming import parse_source_video

    _, title = parse_source_video(source_video)
    return title.split(" - ", 1)[-1].strip()


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
    titles = {_normalize(talk_title(str(r.get("source_video") or ""))) for r in records}
    return Catalog(frozenset(chapters), min(years), max(years), frozenset(t for t in titles if t))


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
