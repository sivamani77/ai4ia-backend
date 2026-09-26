"""The 07b repair prompt must carry fragment ids the repair can copy back."""
from __future__ import annotations

import json
import re

from app.pipeline.validate_gen.quality_gates import _build_repair_prompt
from app.pipeline.validate_gen.timeline_enrichment import enrich_final_timeline

# A real id from the corpus; 84% of ids are longer than 50 characters.
LONG_ID = "0243_srXR5cfLK4I_2022 - 3rd Annual AI4IA Conference - Live Opening Session_0021"


def _prompt_ids(user: str) -> list[str]:
    return re.findall(r'"fragment_id": "([^"]*)"', user)


def test_repair_prompt_keeps_the_full_fragment_id():
    _, user = _build_repair_prompt(
        current_timeline=[{"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": LONG_ID}],
        quality_gates={"has_speaker_clips": False, "offending_items": {}},
        narrative_sections=[],
        user_prompt="What did Cordel Green say about media regulation?",
        iteration=1,
    )
    assert LONG_ID in _prompt_ids(user)


def test_an_id_copied_from_the_prompt_survives_enrichment():
    original = [{"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": LONG_ID,
                 "speaker": "Cordel Green", "source_video": "0243_srXR5cfLK4I_2022"}]
    _, user = _build_repair_prompt(original, {"offending_items": {}}, [], "prompt", 1)
    repaired = [{"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": _prompt_ids(user)[0]}]

    result = enrich_final_timeline(repaired, candidate_map={}, original_timeline=original)

    assert [item["fragment_id"] for item in result] == [LONG_ID]
    assert json.dumps(result)  # still serialisable for the storyline file
