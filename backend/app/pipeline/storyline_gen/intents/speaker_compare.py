"""Two speakers compared, each in their own clips (docs/design/intent-layer-design.md §7.7)."""
from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "speaker_compare"
VERSION = "1"
CLIPS = "required"
SECTIONS = [
    section("S1_OPENING", "Two Voices",
            "Introduce the two speakers and the question they are compared on, only from archive evidence.",
            "concept_card", ["comparison", "speakers", "question"],
            "Two AI4IA voices, one question."),
    section("S2_FIRST", "The First Voice",
            "Side A's view, carried by side A's own clips.",
            "concept_card", ["position", "view", "first speaker"],
            "The first speaker sets out their view.", core=True, repeat=True),
    section("S3_SECOND", "The Second Voice",
            "Side B's view, carried by side B's own clips.",
            "concept_card", ["position", "view", "second speaker"],
            "The second speaker sets out theirs.", core=True, repeat=True),
    section("S4_CONTRAST", "Where They Meet and Differ",
            "Where the two views agree and where they part, only from what both sides' clips say.",
            "diagram", ["agreement", "difference", "contrast"],
            "Set side by side, the two views share some ground and part on some points."),
    section("S5_SYNTHESIS", "What the Comparison Shows",
            "What the two views together add to the question.",
            "closing_synthesis", ["takeaway", "comparison", "insight"],
            "Together, the two voices sharpen the question."),
]
INSTRUCTIONS = """
COMPARISON RULES:
- The film compares the two speakers in `comparison.sides`: side A first, then side B. Each candidate's
  `compare_side` says whose it is.
- A speaker clip is only ever presented as its own side's speaker. Never attribute one side's words to the other.
- Keep the two sides balanced: similar time and a similar number of clips for each.
- "Where they meet and differ" may only contrast points that clips from both sides actually make. Do not invent
  agreement or disagreement.
- If `comparison.gap` names a speaker, the archive has no talk where that speaker discusses the topic: say so once,
  in general terms, and never describe that speaker's views.
- A side marked `panel_only` appears only in panels: frame those clips as the panel, never as that person.
""" + format_instructions(SECTIONS, CLIPS)


def matches(profile: dict, director_plan: dict) -> bool:
    return len(profile.get("compare_speakers") or []) == 2


def extend_payload(payload: dict, context: dict) -> None:
    payload["comparison"] = context.get("comparison") or {}
