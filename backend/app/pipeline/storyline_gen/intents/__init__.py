"""Intent registry: the first module whose matches() is true decides the instructions."""
from __future__ import annotations

import re
from types import ModuleType

from app.pipeline.storyline_gen.intents import (
    chronological,
    counting,
    overview,
    speaker,
    speaker_compare,
    speaker_panel,
    speaker_topic,
    theme_chapter,
    title,
    topic,
)
from app.pipeline.storyline_gen.intents.base import system_prompt, template_hash

REGISTRY: list[ModuleType] = [
    title, speaker_compare, speaker_panel, speaker_topic, speaker, counting, overview, theme_chapter, chronological,
    topic,
]

# Changes whenever any intent's template changes; part of the storyline cache key.
REGISTRY_VERSION = ",".join(f"{m.NAME}:{m.VERSION}:{template_hash(m.INSTRUCTIONS)}" for m in REGISTRY)


def select(profile: dict, director_plan: dict) -> ModuleType:
    return next(m for m in REGISTRY if m.matches(profile or {}, director_plan or {}))


def prepare_director_plan(intent: ModuleType, director_plan: dict) -> dict:
    """The director plan as this intent sends it to the LLM."""
    prepare = getattr(intent, "prepare_director_plan", None)
    return prepare(director_plan) if prepare else director_plan


def extend_payload(intent: ModuleType, payload: dict, context: dict | None) -> dict:
    """Add the intent's own evidence (facts block, panel roster, facet) to the LLM payload."""
    extend = getattr(intent, "extend_payload", None)
    if extend:
        extend(payload, context or {})
    return payload


def by_name(name: str | None) -> ModuleType | None:
    return next((m for m in REGISTRY if m.NAME == name), None)


def for_storyline(data: dict) -> ModuleType:
    """The intent a saved storyline was planned with. Stage 07 records it even
    when the LLM storyline failed; older files fall back to selecting again."""
    storyline = data.get("storyline") or {}
    recorded = storyline.get("format") or (storyline.get("intent") or {}).get("name")
    profile = data.get("prompt_profile") or data.get("prompt_intent") or {}
    return by_name(recorded) or select(profile, data.get("director_plan") or storyline.get("director_plan") or {})


# The prompt asks to hear speakers: clips, quotes, or what speakers said.
_CLIPS_REQUESTED = re.compile(
    r"\b(clips?|soundbites?|sound bites?|quotes?|quotations?|excerpts?|footage|own words|hear from)\b"
    r"|\bwhat (?:did|do|does) (?:the )?(?:speakers?|panell?ists?|experts?|participants?|presenters?)\b"
    r"|\b(?:speakers?|panell?ists?|experts?|presenters?) (?:said|say|says|argued|argue|discussed)\b",
    re.I,
)


def clips_requested(prompt: str) -> bool:
    return bool(_CLIPS_REQUESTED.search(prompt or ""))


def clips_required(intent: ModuleType, prompt: str) -> bool:
    """Speaker, panel and single-talk requests need clips; others only when the prompt asks."""
    return intent.CLIPS == "required" or clips_requested(prompt)


def describe(intent: ModuleType) -> dict[str, str]:
    """What a job records about the intent it used."""
    return {"name": intent.NAME, "version": intent.VERSION, "template_hash": template_hash(intent.INSTRUCTIONS)}


__all__ = [
    "REGISTRY", "REGISTRY_VERSION", "by_name", "clips_required", "clips_requested", "describe", "extend_payload",
    "for_storyline", "prepare_director_plan", "select", "system_prompt",
]
