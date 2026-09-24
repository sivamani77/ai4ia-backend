"""Make an archive overview answer the figure the prompt asked for.

The narration is told to state the asked figure; this checks it did, and
otherwise opens the first narration with the figure from archive_facts.
"""
from __future__ import annotations

import re
from typing import Any

# Which figure a question asks for.
_QUESTIONS = {
    "speakers": re.compile(r"\bhow many\b.{0,30}\b(speakers?|people|presenters?|panell?ists?|voices)\b"
                           r"|\bnumber of (speakers|presenters)\b", re.I),
    "talks": re.compile(r"\bhow many\b.{0,30}\b(talks?|videos?|sessions?|presentations?|recordings?)\b"
                        r"|\bnumber of (talks|videos|sessions)\b", re.I),
    "themes": re.compile(r"\bhow many\b.{0,30}\bthemes?\b|\bnumber of themes\b", re.I),
    "chapters": re.compile(r"\bhow many\b.{0,30}\bchapters?\b|\bnumber of chapters\b", re.I),
    "years": re.compile(r"\b(how many years|which years|what years|since when|how long)\b", re.I),
}

# The facts-block line each figure comes from (corpus_facts.format_facts_for_prompt).
_FACT_LINES = {
    "speakers": re.compile(r"^- Speakers: (over \d+)", re.M),
    "talks": re.compile(r"^- Talks in the archive: (\d+)", re.M),
    "themes": re.compile(r"^- Themes: (\d+)", re.M),
    "chapters": re.compile(r"^- Chapters: (\d+)", re.M),
    "years": re.compile(r"^- Years covered: (\d{4})\D+(\d{4})", re.M),
}

_SENTENCES = {
    "speakers": "The AI4IA archive brings together {0} speakers.",
    "talks": "The AI4IA archive holds {0} talks.",
    "themes": "Its talks are organized into {0} themes.",
    "chapters": "Its talks are organized into {0} chapters.",
    "years": "The talks were recorded between {0} and {1}.",
}

_NUMBER_WORDS = {
    "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "25": "twenty-five",
}


def asked_figures(prompt: str) -> list[str]:
    """Figures the prompt asks for, in a fixed order."""
    return [key for key, pattern in _QUESTIONS.items() if pattern.search(prompt or "")]


def _stated(text: str, values: tuple[str, ...]) -> bool:
    """True when the text contains the figure, as digits or as a number word."""
    lowered = text.lower()
    for value in values:
        number = value.split()[-1]
        if not re.search(rf"\b{number}\b", lowered) and _NUMBER_WORDS.get(number, "\0") not in lowered:
            return False
    return True


def ensure_asked_figures(timeline: list[dict[str, Any]], prompt: str, facts: str) -> list[str]:
    """Open the first narration with any asked figure it does not state.
    Returns the figures added."""
    first = next((item for item in timeline
                  if (item.get("type") or item.get("segment_type")) == "narration_broll"), None)
    if first is None or not facts:
        return []
    text = first.get("voiceover_text") or ""
    added: list[str] = []
    for key in asked_figures(prompt):
        match = _FACT_LINES[key].search(facts)
        if match and not _stated(text, match.groups()):
            added.append(_SENTENCES[key].format(*match.groups()))
    if added:
        first["voiceover_text"] = " ".join(added + [text]).strip()
        first["subtitle_text"] = first["voiceover_text"]
    return added
