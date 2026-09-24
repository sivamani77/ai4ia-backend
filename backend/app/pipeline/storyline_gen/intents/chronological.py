"""Evidence spread across years, told in time order (director mode chronological_relay)."""
NAME = "chronological"
VERSION = "1"
INSTRUCTIONS = """
TEMPORAL ARC RULES (chronological_relay mode):
- Each section corresponds to a distinct time period. The year_slot for each section is provided in the director_plan.
- Every section narration MUST open with a temporal anchor sentence: "In [year]..." or "By [year], ..." or "As early as [year]...".
- Do NOT invent years — use only the year_slot value provided for that section.
- The narration must reflect the intellectual/historical progression: context → problem → risk → solution → synthesis.
- Voice-over bridges between sections must reference the time gap: "Three years later...", "By the following decade...", etc.
- If year_slot is null for a section, omit the temporal anchor and write a thematic bridge instead.
"""


def matches(profile: dict, director_plan: dict) -> bool:
    return director_plan.get("video_mode") == "chronological_relay"


def prepare_director_plan(director_plan: dict) -> dict:
    """The relay prompt always gets a structure list, even when the director left it empty."""
    return {**director_plan, "recommended_structure": director_plan.get("recommended_structure") or []}
