"""Speaker names reach the video in registry spelling, not the prompt's or the LLM's."""
from __future__ import annotations

import pytest

from app.pipeline.validate_gen.timeline_enrichment import canonicalize_speaker_names, enrich_final_timeline
from app.services.speaker_registry import get_registry

REGISTRY = get_registry()


@pytest.mark.parametrize("text,expected", [
    ("warned Cordell Green today", "warned Cordel Green today"),
    ("CORDELL GREEN said", "Cordel Green said"),
    ("Cordell Greene spoke", "Cordel Green spoke"),          # longer alias wins
    ("Cordel Green spoke", "Cordel Green spoke"),            # canonical untouched
    ("Cordell Greenfield", "Cordell Greenfield"),            # whole words only
    ("green policy in media", "green policy in media"),      # ordinary words untouched
    ("", ""),
])
def test_fix_names_in_text(text, expected):
    assert REGISTRY.fix_names_in_text(text) == expected


def test_clip_label_comes_from_the_clip_not_the_llm():
    llm_item = {"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": "f1",
                "speaker": "Cordell Green", "start": 10.0, "end": 40.0}
    candidate = {"fragment_id": "f1", "speaker": "Cordel Green", "source_video": "v", "start": 10.0, "end": 40.0}

    result = enrich_final_timeline([llm_item], candidate_map={"f1": candidate}, original_timeline=[])

    assert result[0]["speaker"] == "Cordel Green"


def test_canonicalize_speaker_names_fixes_labels_and_text():
    timeline = [
        {"type": "speaker_clip", "speaker": "Kodel Green", "subtitle_text": "Thank you."},
        {"type": "narration_broll", "voiceover_text": "Cordell Green argued for oversight.",
         "subtitle_text": "Cordell Green argued for oversight.", "visual_title": "Cordell Green on regulation"},
        {"type": "speaker_clip", "speaker": "Multiple speakers"},
        {"type": "title_card", "title": "Cordell Green on Media Regulation"},
    ]
    result = canonicalize_speaker_names(timeline)

    assert result[0]["speaker"] == "Cordel Green"
    assert result[1]["voiceover_text"] == "Cordel Green argued for oversight."
    assert result[1]["subtitle_text"] == "Cordel Green argued for oversight."
    assert result[1]["visual_title"] == "Cordel Green on regulation"
    assert result[2]["speaker"] == "Multiple speakers"
    assert result[3]["title"] == "Cordel Green on Media Regulation"
