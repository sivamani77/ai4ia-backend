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
    ("chronological_topic", "chronological"), ("chronological_chapter", "chronological"),
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
