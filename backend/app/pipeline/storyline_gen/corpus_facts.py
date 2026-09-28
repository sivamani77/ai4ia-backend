"""Facts about the archive, for prompts that ask about the archive itself.

Computed from the same records stage 07 retrieves from, so every number is
live and describes exactly the corpus the video's clips are drawn from.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

from app.pipeline.storyline_gen.text_utils import clean_text, normalize_name

# Values the speaker extractor emits when it could not identify anyone.
_NOT_A_SPEAKER = {"null", "none", "unknown", "n/a", "na", "speaker", "i"}
_HONORIFIC = re.compile(r"^(dr|prof|professor|mr|mrs|ms|miss|sir|hon|justice)\.?\s+", re.I)


def _speaker_key(name: str) -> str | None:
    """Identity key for counting speakers. Only deterministic fixes: junk
    values dropped, honorifics stripped, known aliases resolved. Near-miss
    spellings are left alone, since merging them could join two people."""
    from app.pipeline.storyline_gen.prompt_intent import SPEAKER_ALIAS_MAP

    key = normalize_name(name)
    if key in _NOT_A_SPEAKER or len(key) <= 1:
        return None
    key = _HONORIFIC.sub("", key).strip()
    return normalize_name(SPEAKER_ALIAS_MAP.get(key, key)) or None


# Two keys this similar are probably one person spelled two ways
# ("dario pryshko" / "dario prishko"). Used only to bound the count and to
# withhold ambiguous names, never to decide an identity.
_VARIANT_SIMILARITY = 0.88


def _variant_groups(keys: list[str]) -> list[set[str]]:
    """Cluster keys that are near-miss spellings of each other."""
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if SequenceMatcher(None, a, b).ratio() >= _VARIANT_SIMILARITY:
                parent[find(a)] = find(b)
    groups: dict[str, set[str]] = defaultdict(set)
    for k in keys:
        groups[find(k)].add(k)
    return list(groups.values())


def _year(value: Any) -> int | None:
    # Chroma stores a missing year as "", not None.
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1900 < year < 2100 else None


def build_corpus_facts(records: list[dict[str, Any]], top_n: int = 5,
                       registry_speaker_count: int | None = None, year: int | None = None) -> dict[str, Any]:
    """Counts and rankings over the active corpus. Omits anything it cannot
    measure rather than guessing. `registry_speaker_count` (speakers with a
    recorded talk, panels included) replaces the fragment-based estimate.
    `year`: the records are one conference year's, and the facts say so."""
    from app.pipeline.storyline_gen.prompt_intent import SPEAKER_ALIAS_MAP

    videos: set[str] = set()
    speakers: Counter = Counter()
    spellings: dict[str, Counter] = defaultdict(Counter)
    themes: Counter = Counter()
    chapters: Counter = Counter()
    video_years: dict[str, int] = {}

    for record in records:
        source_video = clean_text(str(record.get("source_video") or ""))
        if not source_video:
            continue
        videos.add(source_video)

        raw_speaker = clean_text(str(record.get("speaker") or ""))
        speaker = _speaker_key(raw_speaker)
        if speaker:
            speakers[speaker] += 1
            spellings[speaker][raw_speaker] += 1
        theme = clean_text(str(record.get("primary_theme") or ""))
        if theme:
            themes[theme] += 1
        chapter = clean_text(str(record.get("primary_chapter") or ""))
        if chapter:
            chapters[chapter] += 1

        video_year = _year(record.get("video_year"))
        if video_year is not None:
            video_years[source_video] = video_year

    keys = sorted(speakers)
    groups = _variant_groups(keys)
    ambiguous = {k for g in groups if len(g) > 1 for k in g}
    canonical = {normalize_name(v) for v in SPEAKER_ALIAS_MAP.values()}

    def nameable(key: str) -> bool:
        # A full name, spelled one way, or confirmed by the alias map.
        if len(key.split()) < 2:
            return False
        return key in canonical or key not in ambiguous

    def display(key: str) -> str:
        for alias in SPEAKER_ALIAS_MAP.values():
            if normalize_name(alias) == key:
                return alias
        return spellings[key].most_common(1)[0][0]

    named = [(k, c) for k, c in speakers.most_common() if nameable(k)][:top_n]

    facts: dict[str, Any] = {
        "video_count": len(videos),
        "fragment_count": sum(1 for r in records if r.get("source_video")),
        # Upper bound: distinct after deterministic cleanup. Lower bound: also
        # merging near-miss spellings. The true count lies between them.
        "speaker_count": len(speakers),
        "speaker_count_lower_bound": len(groups),
        "theme_count": len(themes),
        "chapter_count": len(chapters),
        "themes": [{"name": n, "fragments": c} for n, c in themes.most_common()],
        "chapters": [{"name": n, "fragments": c} for n, c in chapters.most_common()],
        "top_speakers": [{"name": display(k), "fragments": c} for k, c in named],
    }
    if registry_speaker_count:
        facts["registry_speaker_count"] = registry_speaker_count
    if year:
        facts["year"] = year
    elif video_years:
        facts["year_first"] = min(video_years.values())
        facts["year_last"] = max(video_years.values())
        facts["edition_years"] = sorted(set(video_years.values()))
    return facts


AI4IA_FULL_NAME = "Artificial Intelligence for Information Accessibility"


def _over(count: int) -> str:
    """"over 60" for 65, "over 50" for exactly 60; a small count is stated as it is."""
    return f"over {(count - 1) // 10 * 10}" if count > 10 else str(count)


def format_facts_for_prompt(facts: dict[str, Any]) -> str:
    """Plain-text facts block for the storyline LLM. Every figure here is one
    the narration may state; anything absent must not be invented."""
    lines = [f"- AI4IA stands for: {AI4IA_FULL_NAME}"]
    if facts.get("year"):
        lines.append(f"- Scope: AI4IA {facts['year']} only. Every figure below is for {facts['year']}; "
                     f"describe that edition, not the whole archive.")
    lines += [
        f"- Talks in the archive: {facts['video_count']}",
        f"- Speakers: {_over(facts.get('registry_speaker_count') or facts['speaker_count_lower_bound'])}",
        f"- Themes: {facts['theme_count']}",
        f"- Chapters: {facts['chapter_count']}",
    ]
    if "year_first" in facts:
        # Range only: the narration states the span, never a count of years.
        lines.append(f"- Years covered: {facts['year_first']}–{facts['year_last']} (state the range; never count the years)")
    if facts["themes"]:
        lines.append("- Themes, most discussed first:")
        lines += [f"    {t['name']}" for t in facts["themes"]]
    if facts["chapters"]:
        lines.append("- Most discussed chapters:")
        lines += [f"    {c['name']}" for c in facts["chapters"][:8]]
    if facts["top_speakers"]:
        lines.append("- Most represented speakers:")
        lines += [f"    {s['name']}" for s in facts["top_speakers"]]
    return "\n".join(lines)


__all__ = ["build_corpus_facts", "format_facts_for_prompt"]
