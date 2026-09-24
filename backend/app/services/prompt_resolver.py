"""L2a resolve: check the people, chapters and years a prompt names against the archive.

See docs/intent-layer-design.md §6. Declines only when the prompt clearly
refers to something the archive does not have; anything else passes on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

from app.core.error_messages import DECLINE_MESSAGES
from app.services.corpus_catalog import Catalog
from app.services.speaker_names import norm_key

CLOSE = 0.80          # full-name similarity that counts as a plausible match
CLEAR_GAP = 0.08      # how far ahead of the next person a "did you mean" must be
TOKEN_CLOSE = 0.85    # one-word misspelling of a first name or surname
MAX_LISTED = 5

_NAME = r"[A-Za-zÀ-ÿ.'\-]+(?:\s+[A-Za-zÀ-ÿ.'\-]+){0,3}?"
_MENTIONS = [
    re.compile(rf"\b(?:what|how)\s+(?:did|does|do|has|have|would)\s+(?P<name>{_NAME})\s+"
               r"(?:say|said|think|argue|talk|speak|discuss|mention|mean|believe|feel|propose|suggest|see)\b", re.I),
    re.compile(rf"(?P<name>{_NAME})'s\s+(?:views?|talks?|position|opinions?|thoughts|perspective|speech"
               r"|presentation|remarks|ideas|comments)\b", re.I),
    re.compile(rf"\baccording\s+to\s+(?P<name>{_NAME})(?=\s*(?:[,.;?!]|$|\s+(?:on|about|regarding)\b))", re.I),
    re.compile(rf"\bspeaker\s+(?P<name>{_NAME})(?=\s*(?:[,.;?!]|$|\s+(?:on|about|say|said|talk)\b))", re.I),
]
# Words that name groups or places, never one person.
_NOT_A_PERSON = frozenset("""
    speaker speakers panelist panelists panel panels expert experts people participants they he she we
    you i it others everyone anyone someone researchers scholars governments government women men youth
    students experts regulators the a an this that these those conference archive ai ai4ia unesco ifap
    wgia africa europe asia america caribbean jamaica india china world
""".split())
_NAME_PARTICLES = frozenset({"de", "da", "del", "van", "von", "der", "la", "le", "bin", "al", "e"})
_CHAPTER = re.compile(r"\b(?:chapter|ch\.?)\s*(\d{1,3})\b", re.I)
_ARCHIVE_NOUN = r"(?:conference|edition|event|events|session|sessions|talk|talks|video|videos|speakers?)"
_YEARS = [
    re.compile(rf"\b((?:19|20)\d{{2}})\s+{_ARCHIVE_NOUN}\b", re.I),
    re.compile(rf"\b{_ARCHIVE_NOUN}\s+(?:from|in|during|of)\s+((?:19|20)\d{{2}})\b", re.I),
]


@dataclass(frozen=True)
class Decline:
    reason_code: str
    message: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Resolution:
    speaker: str | None = None
    slots: dict = field(default_factory=dict)
    dropped_slots: list[dict] = field(default_factory=list)
    decline: Decline | None = None


def _registry():
    from app.services.speaker_registry import get_registry
    return get_registry()


@lru_cache(maxsize=1)
def _people() -> dict[str, dict]:
    """canonical -> {names: normalized names and aliases, tokens: their words, confirmed}."""
    people = {}
    for canonical, entry in _registry().speakers.items():
        names = {norm_key(n) for n in (canonical, *(entry.get("aliases") or [])) if norm_key(n)}
        people[canonical] = {
            "names": names,
            "tokens": {t for n in names for t in n.split() if len(t) > 1},
            "confirmed": entry.get("status", "confirmed") == "confirmed",
        }
    return people


@lru_cache(maxsize=1)
def _all_name_tokens() -> frozenset[str]:
    return frozenset(t for p in _people().values() for t in p["tokens"] if len(t) > 2)


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def _decline(code: str, suggestions: list[str] | None = None, **fmt) -> Decline:
    return Decline(code, DECLINE_MESSAGES[code].format(**fmt), suggestions or [])


def _is_filler(word: str) -> bool:
    from app.pipeline.storyline_gen.prompt_intent import STOPWORDS
    w = word.lower().strip(".'")
    return w in _NOT_A_PERSON or w in STOPWORDS


def _mentioned_name(prompt: str) -> str | None:
    """A person the prompt asks about, if its wording clearly names one."""
    for pattern in _MENTIONS:
        match = pattern.search(prompt)
        if not match:
            continue
        words = match.group("name").split()
        while words and _is_filler(words[0]):
            words.pop(0)
        while words and _is_filler(words[-1]):
            words.pop()
        if not words:
            continue
        name = " ".join(words)
        # A name is capitalised ("Cordle Grene"); other text only counts if close to a real speaker.
        looks_like_name = all(w[0].isupper() or w.lower() in _NAME_PARTICLES for w in words)
        if looks_like_name or _best_scores(norm_key(name))[0][1] >= CLOSE:
            return name
    return None


def _best_scores(key: str) -> list[tuple[str, float]]:
    scored = [(c, max(SequenceMatcher(None, key, n).ratio() for n in p["names"]))
              for c, p in _people().items()]
    return sorted(scored, key=lambda x: -x[1])


def _list_decline(name: str, matches: list[str]) -> Decline:
    """Several people match: suggest only confirmed names (review names are never shown)."""
    matches = sorted(matches)
    confirmed = [m for m in matches if _people()[m]["confirmed"]][:MAX_LISTED]
    if len(matches) > 3:
        return _decline("speaker_too_many", confirmed, name=name, options=_join(confirmed) if confirmed else "")
    if len(confirmed) < len(matches):
        return _decline("speaker_ambiguous_partial", confirmed, name=name, options=_join(confirmed))
    return _decline("speaker_ambiguous", confirmed, name=name, options=_join(confirmed))


def _match_mention(name: str) -> tuple[str | None, Decline | None]:
    """(canonical, None) when one person matches, (None, decline) when the name
    clearly means a speaker we do not have, else (None, None): not a person
    after all (e.g. "Nigeria", "European Union"), so the prompt goes on as a topic."""
    shown = name[:60]
    key = norm_key(name)
    tokens = key.split()
    if len(tokens) == 1:
        token = tokens[0]
        exact = [c for c, p in _people().items() if token in p["tokens"]]
        if len(exact) == 1:
            return exact[0], None
        if exact:
            return None, _list_decline(shown, exact)
        close = [c for c, p in _people().items()
                 if len(token) >= 4 and any(SequenceMatcher(None, token, t).ratio() >= TOKEN_CLOSE for t in p["tokens"])]
        if len(close) == 1 and _people()[close[0]]["confirmed"]:
            return None, _decline("speaker_did_you_mean", close, name=shown, options=close[0])
        if len(close) > 1:
            return None, _list_decline(shown, close)
        return None, None

    ranked = _best_scores(key)
    (top, top_score), second_score = ranked[0], ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score >= CLOSE and top_score - second_score >= CLEAR_GAP:
        if _people()[top]["confirmed"]:
            return None, _decline("speaker_did_you_mean", [top], name=shown, options=top)
        return None, _decline("speaker_not_found", name=shown)
    close = [c for c, s in ranked if s >= CLOSE and top_score - s < CLEAR_GAP]
    if len(close) >= 2:
        return None, _list_decline(shown, close)
    # Only a person if it shares a first name or surname with a speaker ("John Smith").
    if any(t in _all_name_tokens() for t in tokens):
        return None, _decline("speaker_not_found", name=shown)
    return None, None


def resolve(prompt: str, catalog: Catalog | None = None) -> Resolution:
    """Resolve the prompt's speaker, chapter and year against the archive."""
    from app.pipeline.storyline_gen.prompt_intent import detect_requested_speaker, load_known_speakers

    result = Resolution()
    registry = _registry()

    speaker = detect_requested_speaker(prompt, load_known_speakers())
    if not speaker:
        mention = _mentioned_name(prompt)
        if mention:
            speaker, decline = _match_mention(mention)
            if decline:
                result.decline = decline
                result.dropped_slots.append({"slot": "speaker", "value": mention, "reason": decline.reason_code})
                return result
    if speaker:
        canonical = registry.canonical(speaker) or speaker
        if canonical in registry.speakers and not registry.has_recording(canonical):
            result.decline = _decline("speaker_no_recording", name=canonical)
            return result
        result.speaker = canonical
        result.slots["speaker"] = canonical

    if catalog is not None:
        for match in _CHAPTER.finditer(prompt):
            number = int(match.group(1))
            if number not in catalog.chapter_numbers:
                result.decline = _decline("chapter_not_found", n=number, first=min(catalog.chapter_numbers),
                                          last=max(catalog.chapter_numbers))
                result.dropped_slots.append({"slot": "chapter", "value": number, "reason": "chapter_not_found"})
                return result
            result.slots["chapter"] = number
        years = sorted({int(m.group(1)) for p in _YEARS for m in p.finditer(prompt)})
        outside = [y for y in years if not catalog.first_year <= y <= catalog.last_year]
        if outside:
            result.decline = _decline("year_out_of_range", first=catalog.first_year, last=catalog.last_year)
            result.dropped_slots += [{"slot": "year", "value": y, "reason": "year_out_of_range"} for y in outside]
            return result
        if years:
            result.slots["years"] = years
    return result
