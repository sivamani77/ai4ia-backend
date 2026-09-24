"""The prompt names one talk by its title."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "title"
VERSION = "2"
CLIPS = "required"
SECTIONS = [
    section("S1_OPENING", "The Talk",
            "Introduce the talk: who gave it, at which event and in which year, only from the archive.",
            "concept_card", ["talk", "speaker", "AI4IA"],
            "This film follows one talk from the AI4IA archive."),
    section("S2_QUESTION", "The Question",
            "The problem the talk takes on.",
            "mindmap", ["question", "problem", "stakes"],
            "The talk takes on a pressing question."),
    section("S3_ARGUMENT", "The Argument",
            "The talk's main argument, in the speaker's own words.",
            "concept_card", ["argument", "ideas", "voice"],
            "Its argument is best heard in the speaker's own words.", core=True, repeat=True),
    section("S4_EXAMPLES", "The Examples",
            "The examples and evidence the talk uses.",
            "diagram", ["examples", "evidence", "cases"],
            "The talk grounds its argument in examples.", core=True, repeat=True),
    section("S5_TAKEAWAY", "The Takeaway",
            "The talk's takeaway.",
            "closing_synthesis", ["takeaway", "insight", "talk"],
            "The talk leaves a clear takeaway."),
]
INSTRUCTIONS = """
SINGLE TALK RULES:
- Every clip comes from the requested talk. Never use another talk's clips.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_source_video"))
