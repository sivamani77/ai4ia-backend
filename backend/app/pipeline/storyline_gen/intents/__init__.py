"""Intent registry: the first module whose matches() is true decides the instructions."""
from __future__ import annotations

from types import ModuleType

from app.pipeline.storyline_gen.intents import chronological, speaker, speaker_topic, title, topic
from app.pipeline.storyline_gen.intents.base import system_prompt, template_hash

REGISTRY: list[ModuleType] = [title, speaker_topic, speaker, chronological, topic]

# Changes whenever any intent's template changes; part of the storyline cache key.
REGISTRY_VERSION = ",".join(f"{m.NAME}:{m.VERSION}:{template_hash(m.INSTRUCTIONS)}" for m in REGISTRY)


def select(profile: dict, director_plan: dict) -> ModuleType:
    return next(m for m in REGISTRY if m.matches(profile or {}, director_plan or {}))


def prepare_director_plan(intent: ModuleType, director_plan: dict) -> dict:
    """The director plan as this intent sends it to the LLM."""
    prepare = getattr(intent, "prepare_director_plan", None)
    return prepare(director_plan) if prepare else director_plan


def describe(intent: ModuleType) -> dict[str, str]:
    """What a job records about the intent it used."""
    return {"name": intent.NAME, "version": intent.VERSION, "template_hash": template_hash(intent.INSTRUCTIONS)}


__all__ = ["REGISTRY", "REGISTRY_VERSION", "describe", "prepare_director_plan", "select", "system_prompt"]
