"""Pure speaker-name helpers shared by the registry builder
(scripts/build_speaker_registry.py), the runtime registry, and ingest.

No settings or database imports, so standalone scripts can use it.
"""
from __future__ import annotations

import re
import unicodedata

HONORIFICS = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "adv", "advocate",
              "justice", "judge", "hon", "sir", "lt", "gen", "phd", "jr"}

NON_PERSON_WORDS = {"zimbabwe", "eswatini", "uganda", "kenya", "nigeria", "ghana", "india",
                    "pakistan", "nepal", "uruguay", "america", "latin", "africa", "asia",
                    "europe", "caribbean", "event", "events", "conference", "forum", "webinar",
                    "launch", "panel", "opening", "closing", "satellite", "regional", "session",
                    "keynote", "ai4ia", "ai4ai", "presentation", "unesco", "annual", "live"}

_NAME_PARTICLES = {"de", "da", "do", "dos", "van", "von", "der", "al", "el", "le", "la", "del"}

# Values the extractor emits when it could not identify anyone.
PLACEHOLDER_NAMES = {"", "i", "null", "none", "unknown", "n/a", "na", "speaker"}


def fold(text: str) -> str:
    """Strip accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def norm_key(name: str | None) -> str:
    """Lookup key: accents, case, punctuation and titles removed."""
    tokens = re.sub(r"[^a-z0-9]+", " ", fold(name or "").lower()).split()
    return " ".join(t for t in tokens if t not in HONORIFICS)


def is_placeholder(name: str | None) -> bool:
    return (name or "").strip().lower() in PLACEHOLDER_NAMES


def clean_person_name(raw: str) -> str | None:
    """Strip affiliation, titles and nicknames; None if it does not look like a name."""
    name = re.split(r"\s+\||\s+\(|,", raw, maxsplit=1)[0]
    name = re.sub(r"[“\"][^”\"]*[”\"]", " ", name)
    name = re.sub(r"^\s*((prof|dr|mr|mrs|ms|professor|adv)\.?\s+)+", "", name, flags=re.I)
    name = " ".join(name.replace("_", " ").split()).strip(" .:;-")
    tokens = name.split()
    if not 2 <= len(tokens) <= 5:
        return None
    if any(fold(t).lower() in NON_PERSON_WORDS for t in tokens):
        return None
    if not all(t[:1].isupper() or t.lower() in _NAME_PARTICLES for t in tokens):
        return None
    return name


_ROLE_HEADER = re.compile(
    r"^(?:the\s+)?(keynote\s+presenters?|presenters?|moderators?|facilitators?|panelists?|speakers?)\b([^:]{0,40}):?\s*(.*)$",
    re.I,
)
_ROLE_NAMES = {"keynote": "Keynote", "presenter": "Presenter", "moderator": "Moderator",
               "facilitator": "Facilitator", "panelist": "Speaker", "speaker": "Speaker"}


def description_roster(description: str) -> list[tuple[str, str]]:
    """(name, role) pairs listed under role headers in a video description,
    e.g. 'Moderators: Dr. A and B.' or 'Presenters:' followed by one per line."""
    roster: list[tuple[str, str]] = []
    role = None
    for line in (description or "").splitlines():
        text = line.strip()
        if not text:
            role = None
            continue
        header = _ROLE_HEADER.match(text)
        if header and (":" in text or text.lower().rstrip("s") in ("moderator", "presenter", "facilitator")):
            word = header.group(1).lower().split()[0].rstrip("s")
            role = _ROLE_NAMES.get(word, "Speaker")
            for part in re.split(r",|;|\s+and\s+", header.group(3).strip()):
                name = clean_person_name(part)
                if name:
                    roster.append((name, role))
            continue
        if text.endswith(":"):
            role = None
            continue
        if role:
            name = clean_person_name(text)
            if name:
                roster.append((name, role))
    return roster


def title_names(title: str) -> list[str]:
    """Person names at the front of a talk title, e.g. '(6K) C Howell - Topic'."""
    text = re.sub(r"^\(\s*[0-9A-Z]+\s*\)\s*[:：]?\s*", "", (title or "").strip())
    keynote = re.match(r"^keynote\s*\d*\s*[:：]\s*", text, re.I)
    if keynote:
        text = text[keynote.end():]
    text = text.replace("_", " ")
    lead = re.split(r"\s+[-–]\s*|\s*[-–]\s+|\s{3,}", text, maxsplit=1)[0]
    if lead == text and not keynote:
        return []
    names = []
    for part in re.split(r"\s+&\s+|\s+and\s+", lead):
        part = re.sub(r"^((prof|dr|mr|mrs|ms|professor)\.?\s+)+", "", part.strip(), flags=re.I)
        tokens = part.split()
        if keynote and tokens and tokens[0].lower().rstrip(".") == "adv":
            names.append(" ".join(tokens))
            continue
        name = clean_person_name(part)
        if name:
            names.append(name)
    return names
