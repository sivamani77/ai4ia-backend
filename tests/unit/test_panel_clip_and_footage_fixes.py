"""Unmeasured audio is not shown as unusable; no muted speaker footage is inserted; AI4IA's full name."""
from __future__ import annotations

from app.pipeline.export_gen.timeline_selection import ensure_requested_speaker
from app.pipeline.storyline_gen.corpus_facts import build_corpus_facts, format_facts_for_prompt
from app.pipeline.validate_gen.fragment_validation import _build_validation_prompt

SECTIONS = [{"section_id": "S2_CONTEXT", "title": "Context", "purpose": "background"}]


def _prompt(candidate):
    _, user, _ = _build_validation_prompt(candidate, [], SECTIONS, "What did Natali Helberger say?", 60)
    return user


def test_missing_audio_measurements_are_not_shown_as_zeros():
    user = _prompt({"fragment_id": "f1", "start": 0, "end": 30, "text": "A coherent European approach."})
    assert "speech_ratio=0.00" not in user and "is_usable=False" not in user
    assert user.count("not measured (no known audio problem)") == 2


def test_real_measurements_are_still_shown():
    user = _prompt({"fragment_id": "f1", "start": 0, "end": 30, "text": "t",
                    "speech_quality": {"speech_ratio": 0.78, "is_usable_speech": True},
                    "moment_scores": {"quote_completeness_score": 0.41, "final_moment_score": 0.22}})
    assert "speech_ratio=0.78, is_usable=True" in user
    assert "completeness=0.41, final=0.22" in user


def test_no_speaker_footage_is_inserted_for_a_requested_speaker():
    manifest = {"prompt_intent": {"requested_speaker": "Cordel Green"},
                "selected_candidates": [{"fragment_id": "f1", "speaker_visual_allowed": True,
                                         "final_video_allowed": True}]}
    timeline = [{"type": "card", "segment_type": "title_card"}]
    issues = ensure_requested_speaker(manifest, timeline)
    assert timeline == [{"type": "card", "segment_type": "title_card"}]
    assert "No speaker clip" in issues[0]


def test_facts_give_the_full_name():
    facts = format_facts_for_prompt(build_corpus_facts([{"source_video": "v", "speaker": "A B"}]))
    assert "- AI4IA stands for: Artificial Intelligence for Information Accessibility" in facts
