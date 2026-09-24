"""Fallback: any prompt no other intent claims."""
from app.pipeline.storyline_gen.intents import chronological
from app.pipeline.storyline_gen.intents.base import format_instructions
from app.pipeline.validate_gen.narrative_templates import GENERIC_NARRATIVE_SECTIONS

NAME = "topic"
VERSION = "2"
CLIPS = "optional"
# None: 07b picks the generic or the deepfake arc from the prompt.
SECTIONS = None
INSTRUCTIONS = format_instructions(GENERIC_NARRATIVE_SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return True


def prepare_director_plan(director_plan: dict) -> dict:
    """Year-spread evidence keeps its relay structure, as before phase 8."""
    if director_plan.get("video_mode") == "chronological_relay":
        return chronological.prepare_director_plan(director_plan)
    return director_plan
