"""How a topic changed over the years, told in time order (director mode chronological_relay)."""
import re

from app.pipeline.storyline_gen.intents.base import format_instructions, section

NAME = "chronological"
VERSION = "3"
CLIPS = "optional"
# Only the middle period repeats, so a long video stays in time order.
SECTIONS = [
    section("S1_OPENING", "The Question Over Time",
            "The question, and the span of years the talks cover.",
            "concept_card", ["timeline", "question", "years"],
            "The conversation on this question has unfolded over several years."),
    section("S2_EARLY", "The Early Talks",
            "What the earliest talks said.",
            "diagram", ["early", "origins", "first talks"],
            "The earliest talks set out the first questions.", core=True),
    section("S3_SHIFT", "How It Shifted",
            "How the conversation shifted in the years between.",
            "mindmap", ["shift", "change", "debate"],
            "Over the following years, the conversation shifted.", repeat=True),
    section("S4_RECENT", "The Recent Talks",
            "What the most recent talks say.",
            "diagram", ["recent", "today", "latest"],
            "The most recent talks show where the conversation stands.", core=True),
    section("S5_SYNTHESIS", "What Changed",
            "What changed over the years, and what persisted.",
            "closing_synthesis", ["change", "continuity", "future"],
            "Some questions changed over the years; others persisted."),
]
TEMPORAL_RULES = """
TEMPORAL ARC RULES (chronological_relay mode):
- Each section corresponds to a distinct time period. The year_slot for each section is provided in the director_plan.
- Every section narration MUST open with a temporal anchor sentence: "In [year]..." or "By [year], ..." or "As early as [year]...".
- Do NOT invent years — use only the year_slot value provided for that section.
- The narration must reflect the intellectual/historical progression: context → problem → risk → solution → synthesis.
- Voice-over bridges between sections must reference the time gap: "Three years later...", "By the following decade...", etc.
- If year_slot is null for a section, omit the temporal anchor and write a thematic bridge instead.
"""
INSTRUCTIONS = TEMPORAL_RULES + "\n" + format_instructions(SECTIONS, CLIPS)


# The prompt asks how something changed over time.
_TIME_WORDING = re.compile(
    r"\b(evolv\w*|over (?:the )?(?:years|time)|across (?:the )?years|through the years|history|historical"
    r"|timeline|trends?|(?:chang\w*|shift\w*) over|(?:since|from|between) (?:19|20)\d\d)\b",
    re.I,
)


def asks_about_time(prompt: str) -> bool:
    return bool(_TIME_WORDING.search(prompt or ""))


def matches(profile: dict, director_plan: dict) -> bool:
    """Evidence spread over years is not enough: the prompt must ask about change over time,
    by its wording or as read by the intent router."""
    return (director_plan.get("video_mode") == "chronological_relay"
            and (asks_about_time(profile.get("raw_prompt") or "")
                 or (profile.get("route") or {}).get("intent") == "chronological"))


def prepare_director_plan(director_plan: dict) -> dict:
    """The relay prompt always gets a structure list, even when the director left it empty."""
    return {**director_plan, "recommended_structure": director_plan.get("recommended_structure") or []}
