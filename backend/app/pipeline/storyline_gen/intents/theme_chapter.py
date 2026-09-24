"""One theme or chapter of the archive."""
from app.pipeline.storyline_gen.intents import chronological

NAME = "theme_chapter"
VERSION = "1"
# The temporal rules apply unchanged when the director plans the evidence by year.
INSTRUCTIONS = """
THEME / CHAPTER RULES:
- The request is for one part of the archive (see requested_facet). Keep every section inside it.
- Name the theme or chapter as requested_facet gives it. Do not drift into neighbouring topics to fill time.
""" + chronological.INSTRUCTIONS


def matches(profile: dict, director_plan: dict) -> bool:
    return (profile.get("intent_type") in ("theme_video", "chapter_video")
            and bool(profile.get("theme_hint") or profile.get("chapter_hint")))


def prepare_director_plan(director_plan: dict) -> dict:
    if director_plan.get("video_mode") == "chronological_relay":
        return chronological.prepare_director_plan(director_plan)
    return director_plan


def extend_payload(payload: dict, context: dict) -> None:
    payload["requested_facet"] = context.get("requested_facet") or {}
