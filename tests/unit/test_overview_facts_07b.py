"""Overview facts reach 07b, so it keeps the figures stage 07 wrote from them."""
from __future__ import annotations

import pytest

from app.pipeline.validate_gen import narrative_assembly
from app.pipeline.validate_gen.archive_answer import asked_figures, ensure_asked_figures
from app.pipeline.validate_gen.timeline_enrichment import _narration_grounding_issues

FACTS = "- Talks in the archive: 309\n- Speakers: over 300"
APPROVED = [{"fragment_id": "f1", "decision": "speaker_clip", "approved": True, "text": "Welcome to AI4IA."}]
SECTIONS = [{"section_id": "S1_HOOK", "title": "Hook", "purpose": "open", "card_text": "AI4IA"}]


def _assembly_user_text(monkeypatch, storyline_data):
    sent = {}

    def fake(payload, timeout=300):
        sent["user"] = payload["messages"][-1]["content"]
        return "{}"

    monkeypatch.setattr(narrative_assembly, "post_with_retries", fake)
    narrative_assembly.assemble_narrative_with_llm(APPROVED, SECTIONS, "What is AI4IA?", 60, storyline_data)
    return sent["user"]


def test_overview_facts_are_in_the_assembly_prompt(monkeypatch):
    text = _assembly_user_text(monkeypatch, {"storyline": {"archive_facts": FACTS}})
    assert "ARCHIVE FACTS" in text and "Talks in the archive: 309" in text


def test_other_requests_get_the_same_prompt_as_before(monkeypatch):
    with_empty = _assembly_user_text(monkeypatch, {"storyline": {}})
    assert "ARCHIVE FACTS" not in with_empty
    assert with_empty == _assembly_user_text(monkeypatch, {})


def test_facts_count_as_a_grounding_source():
    narration = "Around 45 percent of the talks cover governance."
    assert _narration_grounding_issues(narration, "", set())
    assert not _narration_grounding_issues(narration, "45 percent", set())


FULL_FACTS = ("- AI4IA stands for: Artificial Intelligence for Information Accessibility\n"
              "- Talks in the archive: 309\n- Speakers: over 300\n- Themes: 7\n- Chapters: 25\n"
              "- Years covered: 2021–2026 (state the range; never count the years)")


def _timeline(text):
    return [{"type": "title_card"}, {"type": "speaker_clip", "quote_text": "Hi."},
            {"type": "narration_broll", "voiceover_text": text, "subtitle_text": text},
            {"type": "narration_broll", "voiceover_text": "Later narration."}]


@pytest.mark.parametrize("prompt,expected", [
    ("How many speakers are there?", ["speakers"]),
    ("How many talks and how many themes does AI4IA have?", ["talks", "themes"]),
    ("Which years does the archive cover?", ["years"]),
    ("What is AI4IA?", []),
])
def test_asked_figures(prompt, expected):
    assert asked_figures(prompt) == expected


def test_missing_figure_opens_the_first_narration():
    timeline = _timeline("AI4IA holds 309 talks across seven themes.")
    added = ensure_asked_figures(timeline, "How many speakers are in AI4IA?", FULL_FACTS)
    assert added == ["The AI4IA archive brings together over 300 speakers."]
    first = timeline[2]
    assert first["voiceover_text"].startswith("The AI4IA archive brings together over 300 speakers. AI4IA holds")
    assert first["subtitle_text"] == first["voiceover_text"]
    assert timeline[3]["voiceover_text"] == "Later narration."


@pytest.mark.parametrize("prompt,text", [
    ("How many speakers are there?", "Over 300 speakers have taken the stage."),
    ("How many themes?", "The talks fall into seven themes."),
    ("Which years does it cover?", "Talks recorded from 2021 to 2026."),
    ("What is AI4IA?", "AI4IA is a conference."),
])
def test_stated_or_unasked_figures_are_left_alone(prompt, text):
    timeline = _timeline(text)
    assert ensure_asked_figures(timeline, prompt, FULL_FACTS) == []
    assert timeline[2]["voiceover_text"] == text


def test_no_facts_no_change():
    timeline = _timeline("Narration.")
    assert ensure_asked_figures(timeline, "How many speakers?", "") == []
