"""Overview facts reach 07b, so it keeps the figures stage 07 wrote from them."""
from __future__ import annotations

from app.pipeline.validate_gen import narrative_assembly
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
