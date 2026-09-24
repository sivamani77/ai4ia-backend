"""The intents refactor must send the LLM exactly what it sent before (phase 4 gate)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.storyline_gen import intents, storyline_llm

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INPUTS = json.loads((FIXTURES / "storyline_prompt_inputs.json").read_text())
SNAPSHOTS = json.loads((FIXTURES / "storyline_prompt_snapshots.json").read_text())


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake(user_content, timeout, temperature=0.4, top_p=None, max_tokens=8192, system=None):
        calls["system"], calls["user"] = system, user_content
        return '{"title": "t", "export_timeline": []}'

    monkeypatch.setattr(storyline_llm, "call_strong_llm", fake)
    return calls


@pytest.mark.parametrize("case", sorted(INPUTS))
def test_prompt_is_unchanged(case, captured):
    c = INPUTS[case]
    storyline_llm.generate_llm_storyline(c["user_prompt"], c["duration"], c["candidates"], c["style"],
                                         c["director_plan"], c["prompt_intent"], c["evidence_package"])
    assert captured["system"] == SNAPSHOTS[case]["system"]
    assert captured["user"] == SNAPSHOTS[case]["user"]


@pytest.mark.parametrize("case,expected", [
    ("chronological_topic", "chronological"), ("chronological_chapter", "theme_chapter"),
    ("low_evidence_topic", "topic"), ("speaker_topic", "speaker_topic"),
    ("speaker_only", "speaker"), ("title", "title"),
])
def test_intent_selection(case, expected):
    c = INPUTS[case]
    assert intents.select(c["prompt_intent"], c["director_plan"]).NAME == expected


def test_storyline_records_the_intent(captured):
    c = INPUTS["chronological_topic"]
    result = storyline_llm.generate_llm_storyline(c["user_prompt"], c["duration"], c["candidates"], c["style"],
                                                  c["director_plan"], c["prompt_intent"], c["evidence_package"])
    assert result["intent"]["name"] == "chronological"
    assert result["intent"]["version"] == "1"
    assert len(result["intent"]["template_hash"]) == 12


def test_registry_version_changes_with_any_template(monkeypatch):
    from app.pipeline.storyline_gen.intents import base, topic

    before = ",".join(f"{m.NAME}:{m.VERSION}:{base.template_hash(m.INSTRUCTIONS)}" for m in intents.REGISTRY)
    assert before == intents.REGISTRY_VERSION
    monkeypatch.setattr(topic, "INSTRUCTIONS", "new rule")
    after = ",".join(f"{m.NAME}:{m.VERSION}:{base.template_hash(m.INSTRUCTIONS)}" for m in intents.REGISTRY)
    assert after != before


@pytest.mark.parametrize("plan,expected", [
    ({"video_mode": "chronological_relay"}, []),
    ({"video_mode": "chronological_relay", "recommended_structure": None}, []),
    ({"video_mode": "chronological_relay", "recommended_structure": ["S1"]}, ["S1"]),
])
def test_chronological_always_sends_a_structure_list(plan, expected):
    intent = intents.select({}, plan)
    assert intents.prepare_director_plan(intent, plan)["recommended_structure"] == expected
    assert "recommended_structure" not in intents.prepare_director_plan(intents.select({}, {}), {})


def test_storyline_cache_key_includes_the_intents_version():
    from app.services import storyline_cache

    params = storyline_cache._relevant_params({"target_duration": 60})
    assert params["intents_version"] == intents.REGISTRY_VERSION


class TestNewIntents:
    def test_overview_sends_the_archive_facts(self, captured):
        c = INPUTS["low_evidence_topic"]
        profile = {**c["prompt_intent"], "intent_type": "archive_overview"}
        result = storyline_llm.generate_llm_storyline(
            "What is AI4IA and how many speakers are there?", c["duration"], c["candidates"], c["style"],
            c["director_plan"], profile, c["evidence_package"], intent_context={"archive_facts": "- Talks in the archive: 309"})
        assert result["intent"]["name"] == "overview"
        assert "ARCHIVE OVERVIEW RULES" in captured["system"]
        assert "Talks in the archive: 309" in captured["user"]

    def test_panel_speaker_sends_the_roster(self, captured):
        c = INPUTS["speaker_topic"]
        profile = {**c["prompt_intent"], "speaker_panel_only": True}
        context = {"panel_context": [{"video": "Europe panel", "panel": ["Natali Helberger", "Stefan Baack"]}]}
        result = storyline_llm.generate_llm_storyline(c["user_prompt"], c["duration"], c["candidates"], c["style"],
                                                      c["director_plan"], profile, c["evidence_package"],
                                                      intent_context=context)
        assert result["intent"]["name"] == "speaker_panel"
        assert "Never say or imply that a clip's words are the requested speaker's" in captured["system"]
        assert "Stefan Baack" in captured["user"]

    def test_intents_without_context_send_the_same_payload(self, captured):
        """Only overview, speaker_panel and theme_chapter add keys."""
        c = INPUTS["speaker_topic"]
        storyline_llm.generate_llm_storyline(c["user_prompt"], c["duration"], c["candidates"], c["style"],
                                             c["director_plan"], c["prompt_intent"], c["evidence_package"],
                                             intent_context={"archive_facts": "x", "panel_context": ["y"]})
        assert captured["user"] == SNAPSHOTS["speaker_topic"]["user"]


def test_speaker_panel_only_uses_the_registry():
    from app.pipeline.storyline_gen.intents.context import speaker_panel_only

    assert speaker_panel_only("Natali Helberger") is True
    assert speaker_panel_only("Cordel Green") is False
    assert speaker_panel_only(None) is False


def test_build_intent_context():
    from app.pipeline.storyline_gen.intents.context import build_intent_context

    records = [{"video_id": "xmkczLmiOf0", "video_title": "Europe Satellite Event", "source_video": "s"}]
    context = build_intent_context({"requested_speaker": "Natali Helberger", "speaker_panel_only": True,
                                    "chapter_hint": "Ch 5: Open Data"}, records)
    assert {"video": "Europe Satellite Event", "panel": context["panel_context"][0]["panel"]} in context["panel_context"]
    assert "Natali Helberger" in context["panel_context"][0]["panel"]
    assert context["requested_facet"] == {"chapter_hint": "Ch 5: Open Data"}
    assert "archive_facts" not in context


def test_panel_ranking_puts_named_turns_first_and_spreads_the_rest():
    from app.pipeline.storyline_gen.retrieval import rank_panel_fragments

    frags = [{"fragment_id": f"f{i}", "source_video": "v", "start": i * 10, "text": "general remarks",
              "quote_strength": 0.5} for i in range(9)]
    frags[3]["text"] = "Over to you, Dariia Opryshko."   # named by the moderator
    order = [f["fragment_id"] for f in rank_panel_fragments(frags, "Dariia Opryshko", [])]
    assert order[0] == "f3"
    assert set(order[1:3]) == {"f4", "f5"}              # the next two turns
    rest = order[3:]
    assert {rest[0], rest[1], rest[2]} & {"f0", "f1", "f2"} and {rest[0], rest[1], rest[2]} & {"f6", "f7", "f8"}


def test_retrieval_without_a_requested_speaker_skips_the_panel_fallback():
    """A chapter prompt has no speaker; the panel ranking must not run (it crashed on None)."""
    from app.pipeline.storyline_gen.retrieval import hybrid_retrieve_candidates

    records = [{"fragment_id": "f1", "source_video": "v", "text": "healthcare AI", "primary_chapter": "Ch 13: Health",
                "speaker": "", "speaker_attribution": "Multiple speakers", "start": 0}]
    profile = {"requested_speaker": None, "topic_terms": ["healthcare"], "chapter_hint": "Ch 13: Health"}
    assert hybrid_retrieve_candidates(profile, records, records, top_k=10)
