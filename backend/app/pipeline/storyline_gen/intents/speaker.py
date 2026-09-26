"""One speaker, across their talks."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "speaker"
VERSION = "2"
CLIPS = "required"
SECTIONS = [
    section("S1_OPENING", "Who They Are",
            "Introduce the speaker only as the archive shows them: name, AI4IA talk titles and years. "
            "Never invent a biography or job title.",
            "concept_card", ["speaker", "AI4IA", "archive"],
            "This film follows one voice from the AI4IA archive."),
    section("S2_FOCUS", "The Questions They Bring",
            "The questions and concerns this speaker brings to the AI4IA conversation.",
            "mindmap", ["questions", "concerns", "focus"],
            "Every speaker arrives with questions of their own."),
    section("S3_IDEAS", "In Their Own Words",
            "Their key ideas, carried by their own speaker clips.",
            "concept_card", ["ideas", "argument", "voice"],
            "Their ideas are best heard in their own words.", core=True, repeat=True),
    section("S4_EXAMPLES", "Cases and Evidence",
            "The cases, examples and evidence they use, with their clips.",
            "diagram", ["examples", "evidence", "cases"],
            "They ground their ideas in concrete cases.", core=True, repeat=True),
    section("S5_SYNTHESIS", "What They Add",
            "What this speaker adds to the AI4IA conversation.",
            "closing_synthesis", ["contribution", "insight", "AI4IA"],
            "Together, these ideas add a distinct voice to the AI4IA conversation."),
]
INSTRUCTIONS = """
SPEAKER RULES:
- Every speaker clip is the requested speaker's own. Never use another speaker's clip as filler.
- Narration never words a line as the speaker's quote unless it plays over their clip.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_speaker"))
