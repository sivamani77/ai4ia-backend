"""A speaker who appears only on panels, where clips cannot be attributed to one person."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "speaker_panel"
VERSION = "2"
CLIPS = "required"
SECTIONS = [
    section("S1_OPENING", "The Panel",
            "Introduce the panel: the event, its title and its members, only as panel_context lists them.",
            "concept_card", ["panel", "discussion", "AI4IA"],
            "This film follows a panel discussion from the AI4IA archive."),
    section("S2_DISCUSSION", "The Discussion",
            "What the panel set out to discuss.",
            "mindmap", ["discussion", "questions", "panel"],
            "The panel came together around a shared set of questions."),
    section("S3_EXCHANGE", "Voices From the Panel",
            "The panel's own voices. Clips are framed as the panel, never as one person.",
            "concept_card", ["voices", "exchange", "panel"],
            "The discussion is best heard in the panel's own voices.", core=True, repeat=True),
    section("S4_THEIR_PART", "Their Part in the Panel",
            "What the evidence shows of the requested speaker's part, or say plainly that the clips "
            "cannot be attributed to one panellist.",
            "diagram", ["participation", "panel", "evidence"],
            "The recordings capture the panel as a whole, not each panellist on their own.", core=True),
    section("S5_SYNTHESIS", "What the Panel Adds",
            "What the panel adds to the AI4IA conversation.",
            "closing_synthesis", ["panel", "insight", "AI4IA"],
            "Together, the panel adds several voices to the AI4IA conversation."),
]
INSTRUCTIONS = """
PANEL SPEAKER RULES:
- The requested speaker appears only in panel discussions (see panel_context). Clips come from those panels.
- The clip audio may be any panellist. Never say or imply that a clip's words are the requested speaker's.
- Frame the clips as the panel: "In a panel with ..., speakers discussed ...". Name the panel members only as panel_context lists them.
- Do not quote the requested speaker unless a clip's own text makes clear who is speaking.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return bool(profile.get("requested_speaker")) and bool(profile.get("speaker_panel_only"))


def extend_payload(payload: dict, context: dict) -> None:
    payload["panel_context"] = context.get("panel_context") or []
