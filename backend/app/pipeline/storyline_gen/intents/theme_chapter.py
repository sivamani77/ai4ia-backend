"""One theme or chapter of the archive."""
from app.pipeline.storyline_gen.intents import chronological
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "theme_chapter"
VERSION = "2"
CLIPS = "optional"
SECTIONS = [
    section("S1_OPENING", "The Theme",
            "Introduce the theme or chapter as requested_facet names it, and where it sits among AI4IA's themes.",
            "concept_card", ["theme", "chapter", "AI4IA"],
            "This film explores one part of the AI4IA archive."),
    section("S2_QUESTIONS", "The Questions It Raises",
            "The questions this theme raises.",
            "mindmap", ["questions", "issues", "stakes"],
            "The theme raises questions that matter.", core=True),
    section("S3_PERSPECTIVES", "Perspectives",
            "Perspectives from the archive's talks on this theme; speaker clips go here when used.",
            "diagram", ["perspectives", "talks", "evidence"],
            "Speakers bring different perspectives to the theme.", core=True, repeat=True),
    section("S4_TENSIONS", "Where Views Differ",
            "Where the talks disagree or pull in different directions.",
            "mindmap", ["tensions", "debate", "trade-offs"],
            "Not every speaker sees the theme the same way."),
    section("S5_SYNTHESIS", "What It Adds Up To",
            "What the theme adds up to.",
            "closing_synthesis", ["synthesis", "theme", "future"],
            "Taken together, the talks give the theme a clear shape."),
]
# The temporal rules apply unchanged when the director plans the evidence by year.
INSTRUCTIONS = """
THEME / CHAPTER RULES:
- The request is for one part of the archive (see requested_facet). Keep every section inside it.
- Name the theme or chapter as requested_facet gives it. Do not drift into neighbouring topics to fill time.
- When the director plans the evidence by year, keep this format and apply the temporal rules inside its sections.
""" + chronological.TEMPORAL_RULES + "\n" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return (profile.get("intent_type") in ("theme_video", "chapter_video")
            and bool(profile.get("theme_hint") or profile.get("chapter_hint")))


def prepare_director_plan(director_plan: dict) -> dict:
    if director_plan.get("video_mode") == "chronological_relay":
        return chronological.prepare_director_plan(director_plan)
    return director_plan


def extend_payload(payload: dict, context: dict) -> None:
    payload["requested_facet"] = context.get("requested_facet") or {}
