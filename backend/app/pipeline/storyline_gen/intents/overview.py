"""A question about the archive itself: what AI4IA is, how many talks or speakers."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "overview"
VERSION = "5"
CLIPS = "optional"
SECTIONS = [
    section("S1_OPENING", "What AI4IA Is",
            "Answer the question asked first, with its figure from archive_facts, and say what AI4IA "
            "(Artificial Intelligence for Information Accessibility) is.",
            "concept_card", ["AI4IA", "archive", "conference"],
            "AI4IA, Artificial Intelligence for Information Accessibility, keeps an archive of its conference talks."),
    section("S2_SCALE", "The Archive",
            "The archive by its figures: talks, speakers and years, exactly as archive_facts gives them.",
            "diagram", ["talks", "speakers", "years"],
            "The archive brings together talks from many speakers.", core=True),
    section("S3_THEMES", "What the Talks Cover",
            "The themes the talks cover, as archive_facts lists them.",
            "mindmap", ["themes", "chapters", "agenda"],
            "The talks are organized into themes.", core=True, repeat=True),
    section("S4_VOICES", "The Voices",
            "The range of voices at the conference; speaker clips go here when used.",
            "concept_card", ["voices", "speakers", "perspectives"],
            "The conference gathers a wide range of voices."),
    section("S5_SYNTHESIS", "Why It Matters",
            "Why the archive matters.",
            "closing_synthesis", ["access", "knowledge", "AI4IA"],
            "The archive keeps these conversations open to everyone."),
]
INSTRUCTIONS = """
ARCHIVE OVERVIEW RULES:
- The request is about the AI4IA archive itself. Answer it from archive_facts.
- Answer the question asked first. When the prompt asks for a figure (how many speakers, talks, themes, chapters or years), the first section's narration states that figure.
- State figures (talks, speakers, themes, chapters, years) exactly as archive_facts gives them: "309 talks", never "more than 309"; "over 300 speakers" as written. Never compute a new figure, such as a number of years from the year range.
- Expand the name AI4IA only as archive_facts gives it.
- Clips are voices that illustrate the conference. Never present a clip as the source of a figure, and never attribute a statistic to a speaker.
- Name a speaker only if the name appears in archive_facts or in the clip's source_video text.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return profile.get("intent_type") == "archive_overview"


def extend_payload(payload: dict, context: dict) -> None:
    payload["archive_facts"] = context.get("archive_facts") or ""
