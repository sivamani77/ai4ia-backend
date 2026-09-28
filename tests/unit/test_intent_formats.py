"""Phase 8: documentary format and clip policy per intent (docs/design/intent-layer-design.md §7.4)."""
from __future__ import annotations

import pytest

from app.pipeline.storyline_gen import intents
from app.pipeline.storyline_gen.intents.base import agenda_themes
from app.pipeline.validate_gen import narrative_assembly
from app.pipeline.validate_gen.fallback_timeline import build_fallback_final_timeline
from app.pipeline.validate_gen.fragment_validation import _build_validation_prompt, reject_off_agenda
from app.pipeline.validate_gen.quality_gates import (
    _add_visual_asset_plan,
    _build_repair_prompt,
    _enforce_section_rhythm,
    _fill_missing_sections,
    check_quality_gates,
    frame_duration_fill_as_perspectives_section,
    limit_speaker_clips,
)
from app.pipeline.validate_gen.topic_detection import detect_narrative_sections

FORMATTED = [m for m in intents.REGISTRY if m.SECTIONS]
SPEAKER_SECTIONS = intents.by_name("speaker").SECTIONS


def ids(sections):
    return [s["section_id"] for s in sections]


def narration(sid):
    return {"type": "narration_broll", "section_id": sid, "voiceover_text": "Narration.", "visual_asset_prompt": "p"}


def clip(sid, fid="f"):
    return {"type": "speaker_clip", "section_id": sid, "fragment_id": fid, "speaker": "Cordel Green",
            "quote_complete": True}


class TestFormats:
    def test_every_intent_declares_a_clip_policy(self):
        policy = {m.NAME: m.CLIPS for m in intents.REGISTRY}
        assert policy == {
            "title": "required", "speaker_compare": "required", "speaker_panel": "required",
            "speaker_topic": "required", "speaker": "required", "counting": "optional",
            "overview": "optional", "theme_chapter": "optional", "chronological": "optional", "topic": "optional",
        }

    @pytest.mark.parametrize("intent", FORMATTED, ids=lambda m: m.NAME)
    def test_sections_are_complete(self, intent):
        sections = intent.SECTIONS
        assert len(set(ids(sections))) == len(sections) >= 4
        for s in sections:
            assert s["title"] and s["purpose"] and s["default_voiceover"] and s["visual"]["visual_title"]
        assert sum(1 for s in sections[1:-1] if s["core"]) == 2

    @pytest.mark.parametrize("intent", intents.REGISTRY, ids=lambda m: m.NAME)
    def test_instructions_carry_the_format_placement_and_agenda(self, intent):
        text = intent.INSTRUCTIONS
        assert "DOCUMENTARY FORMAT" in text
        assert "The first and last sections are narration only" in text
        assert all(theme in text for theme in agenda_themes())

    def test_intents_differ_in_format(self):
        titles = {m.NAME: tuple(s["title"] for s in m.SECTIONS) for m in FORMATTED}
        assert len(set(titles.values())) == len(titles)


class TestSectionCount:
    @pytest.mark.parametrize("duration,count", [(60, 4), (120, 6), (300, 8), (600, 12), (900, 15)])
    @pytest.mark.parametrize("intent", FORMATTED, ids=lambda m: m.NAME)
    def test_first_and_last_once_at_every_length(self, intent, duration, count):
        sections = detect_narrative_sections(duration, format_sections=intent.SECTIONS)
        assert len(sections) == count
        assert sections[0] is intent.SECTIONS[0] and sections[-1] is intent.SECTIONS[-1]
        assert len(set(ids(sections))) == count

    def test_short_video_keeps_the_core_sections(self):
        sections = detect_narrative_sections(60, format_sections=SPEAKER_SECTIONS)
        assert ids(sections) == ["S1_OPENING", "S3_IDEAS", "S4_EXAMPLES", "S5_SYNTHESIS"]

    def test_chronological_stays_in_time_order(self):
        sections = detect_narrative_sections(600, format_sections=intents.by_name("chronological").SECTIONS)
        order = [sid.split("_")[1] for sid in ids(sections)]
        assert order == ["OPENING", "EARLY"] + ["SHIFT"] * 8 + ["RECENT", "SYNTHESIS"]

    def test_topic_uses_the_topic_arc_whatever_the_subject(self):
        # The hand-written deepfake arc and its canned narration were removed.
        assert ids(detect_narrative_sections(120)) == ["S1_HOOK", "S2_CONTEXT", "S3_EVIDENCE", "S4_CHALLENGE",
                                                       "S5_RESPONSE", "S6_SYNTHESIS"]


class TestClipPolicy:
    @pytest.mark.parametrize("prompt", [
        "What did speakers say about deepfakes?",
        "What did the panelists say on AI in health?",
        "AI governance, with clips from speakers",
        "Show me quotes on data access",
        "Experts argued about bias; let me hear from them",
    ])
    def test_prompt_asks_for_clips(self, prompt):
        assert intents.clips_requested(prompt)
        assert intents.clips_required(intents.by_name("topic"), prompt)

    @pytest.mark.parametrize("prompt", [
        "Deepfakes and elections", "How many speakers are there?", "What is AI4IA?", "AI in health",
    ])
    def test_prompt_does_not_ask_for_clips(self, prompt):
        assert not intents.clips_requested(prompt)
        assert not intents.clips_required(intents.by_name("topic"), prompt)
        assert intents.clips_required(intents.by_name("speaker"), prompt)

    def test_for_storyline_uses_the_recorded_format(self):
        data = {"storyline": {"format": "overview", "intent": {"name": "topic"}}}
        assert intents.for_storyline(data).NAME == "overview"
        assert intents.for_storyline({"storyline": {"intent": {"name": "speaker"}}}).NAME == "speaker"

    def test_for_storyline_selects_again_for_older_files(self):
        data = {"storyline": {}, "prompt_profile": {"requested_speaker": "Cordel Green"}}
        assert intents.for_storyline(data).NAME == "speaker"


class TestGates:
    SECTIONS = SPEAKER_SECTIONS

    def timeline(self, with_clip):
        items = [narration(s["section_id"]) for s in self.SECTIONS]
        return items + ([clip("S3_IDEAS")] if with_clip else [])

    def test_optional_request_passes_without_clips(self):
        _, gates = check_quality_gates(self.timeline(False), self.SECTIONS, 0, clips_required=False)
        assert gates["has_speaker_clips"] and gates["final_timeline_approved"]

    def test_required_request_fails_without_clips(self):
        _, gates = check_quality_gates(self.timeline(False), self.SECTIONS, 1, clips_required=True)
        assert not gates["has_speaker_clips"] and not gates["final_timeline_approved"]


class TestPlacement:
    SECTIONS = [{"section_id": s} for s in ("S1", "S2", "S3", "S4")]

    def test_clips_never_open_or_close_the_film(self):
        timeline = [{"type": "title_card"}, clip("S1", "open"), narration("S1"), narration("S2"), clip("S2", "mid"),
                    narration("S3"), narration("S4"), clip("S4", "close"), {"type": "closing_card"}]
        result = _enforce_section_rhythm(timeline, self.SECTIONS)
        order = [(x["type"], x.get("section_id"), x.get("fragment_id")) for x in result]
        assert order == [
            ("title_card", None, None),
            ("narration_broll", "S1", None),
            ("narration_broll", "S2", None), ("speaker_clip", "S2", "open"), ("speaker_clip", "S2", "mid"),
            ("narration_broll", "S3", None), ("speaker_clip", "S3", "close"),
            ("narration_broll", "S4", None),
            ("closing_card", None, None),
        ]

    def test_cap_keeps_the_first_clips(self):
        timeline = [narration("S1"), clip("S2", "a"), clip("S2", "b"), clip("S3", "c")]
        assert [x.get("fragment_id") for x in limit_speaker_clips(timeline, 2)] == [None, "a", "b"]

    def test_additional_perspectives_come_before_the_closing_section(self):
        timeline = [narration("S1"), narration("S2"), narration("S4"),
                    {**clip("S4", "extra"), "added_by_duration_fill": True}, {"type": "closing_card"}]
        assert frame_duration_fill_as_perspectives_section(timeline, closing_section_id="S4") == 1
        order = [x.get("label") or x.get("fragment_id") or x.get("section_id") or x["type"] for x in timeline]
        assert order == ["S1", "S2", "additional_perspectives_divider", "extra", "S4", "closing_card"]

    def test_missing_sections_get_a_clip_only_where_allowed(self):
        verdicts = [{"fragment_id": "sp", "decision": "speaker_clip", "approved": True, "source_video": "v"}]
        result = _fill_missing_sections([narration("S2")], self.SECTIONS, verdicts, [], clip_sections={"S3"})
        by_section = {x["section_id"]: x["type"] for x in result}
        assert by_section["S1"] == "narration_broll" and by_section["S3"] == "speaker_clip"

    def test_fallback_puts_clips_in_middle_sections_only(self):
        verdicts = [{"fragment_id": f"c{i}", "decision": "speaker_clip", "approved": True} for i in range(4)]
        tl = build_fallback_final_timeline(verdicts, self.SECTIONS, {"storyline": {}})["final_timeline"]
        assert {x["section_id"] for x in tl if x["type"] == "speaker_clip"} == {"S2", "S3"}
        capped = build_fallback_final_timeline(verdicts, self.SECTIONS, {"storyline": {}}, max_clips=1)
        assert sum(1 for x in capped["final_timeline"] if x["type"] == "speaker_clip") == 1


class TestAgenda:
    def test_off_agenda_speaker_clip_is_rejected(self):
        result = reject_off_agenda({"decision": "speaker_clip", "approved": True, "agenda_relevant": False,
                                    "reason": "Thanks the organisers."})
        assert result["decision"] == "reject" and not result["approved"]
        assert result["reason"].startswith("Not on the AI4IA agenda.")

    @pytest.mark.parametrize("verdict", [
        {"decision": "speaker_clip", "approved": True, "agenda_relevant": True},
        {"decision": "speaker_clip", "approved": True},  # no verdict: the fallback path
        {"decision": "narration_broll", "approved": True, "agenda_relevant": False},
    ])
    def test_other_verdicts_are_kept(self, verdict):
        assert reject_off_agenda(dict(verdict)) == verdict

    def test_validator_lists_the_format_and_the_agenda(self):
        system, user, _ = _build_validation_prompt(
            {"fragment_id": "f", "text": "t"}, [], SPEAKER_SECTIONS, "What did Cordel Green say?", 120)
        assert '"section_id": "S1_OPENING | S2_FOCUS | S3_IDEAS | S4_EXAMPLES | S5_SYNTHESIS"' in user
        assert '"agenda_relevant"' in user
        assert all(theme in system for theme in agenda_themes())
        assert "S3_HUMAN_RISK" not in user

    def test_repair_prompt_states_placement_and_agenda(self):
        _, user = _build_repair_prompt([], {"offending_items": {}}, SPEAKER_SECTIONS, "p", 1, clips_required=False)
        assert "first and last sections are narration only" in user
        assert "optional for this request: use at most 2" in user
        assert agenda_themes()[0] in user


class TestAssemblyPrompt:
    APPROVED = [{"fragment_id": "f1", "decision": "speaker_clip", "approved": True, "text": "x", "video_year": 2024}]

    def user_text(self, monkeypatch, storyline_data, **kwargs):
        sent = {}

        def fake(payload, timeout=300):
            sent["user"] = payload["messages"][-1]["content"]
            return "{}"

        monkeypatch.setattr(narrative_assembly, "post_with_retries", fake)
        narrative_assembly.assemble_narrative_with_llm(self.APPROVED, SPEAKER_SECTIONS, "p", 120, storyline_data,
                                                       **kwargs)
        return sent["user"]

    def test_optional_clips(self, monkeypatch):
        text = self.user_text(monkeypatch, {"storyline": {}}, clips_required=False)
        assert "SPEAKER CLIPS: optional for this request. Use at most 2" in text
        assert '"year": 2024' in text

    def test_required_clips_with_panel_and_facet(self, monkeypatch):
        storyline = {"panel_context": [{"video": "Panel A", "panel": ["A", "B"]}],
                     "requested_facet": {"theme_hint": "AI in Health"}}
        text = self.user_text(monkeypatch, {"storyline": storyline}, clips_required=True, speaker_target=2)
        assert "needs real speaker clips: use at least 2" in text
        assert "PANEL CONTEXT" in text and "Panel A" in text
        assert "REQUESTED FACET" in text and "AI in Health" in text

    def test_system_prompt_forbids_clips_at_the_ends(self):
        assert "Never open or close the film with a clip." in narrative_assembly.NARRATIVE_ASSEMBLY_SYSTEM
        assert "Start with the strongest hook" not in narrative_assembly.NARRATIVE_ASSEMBLY_SYSTEM


def test_section_visual_plan_is_used():
    sections = detect_narrative_sections(120, format_sections=SPEAKER_SECTIONS)
    result = _add_visual_asset_plan([narration("S3_IDEAS_2")], sections)
    assert result[0]["visual_title"] == "In Their Own Words"
    assert result[0]["visual_asset_type"] == "concept_card"


class TestSectionIdAlignment:
    SECTIONS = detect_narrative_sections(120, format_sections=intents.by_name("title").SECTIONS)

    @pytest.mark.parametrize("given,expected", [
        ("S1_HOOK", "S1_OPENING"),          # copied from an example: found in three phase 8 renders
        ("S6_SYNTHESIS", "S5_TAKEAWAY"),
        ("S3_EVIDENCE", "S3_ARGUMENT"),
        ("S2_QUESTION", "S2_QUESTION"),
        ("S_PERSPECTIVES", "S_PERSPECTIVES"),
        ("S9_UNKNOWN", "S9_UNKNOWN"),
    ])
    def test_unknown_ids_map_onto_the_format(self, given, expected):
        from app.pipeline.validate_gen.quality_gates import align_section_ids

        assert align_section_ids([narration(given)], self.SECTIONS)[0]["section_id"] == expected

    def test_prompt_examples_use_this_formats_ids(self, monkeypatch):
        sent = {}

        def fake(payload, timeout=300):
            sent["user"] = payload["messages"][-1]["content"]
            return "{}"

        monkeypatch.setattr(narrative_assembly, "post_with_retries", fake)
        narrative_assembly.assemble_narrative_with_llm(TestAssemblyPrompt.APPROVED, self.SECTIONS, "p", 120,
                                                       {"storyline": {}})
        _, repair = _build_repair_prompt([], {"offending_items": {}}, self.SECTIONS, "p", 1)
        for text in (sent["user"], repair):
            assert "S1_HOOK" not in text and "S2_CONTEXT" not in text
        assert '"section_id": "S1_OPENING"' in sent["user"] and '"section_id": "S2_QUESTION"' in repair


class TestRouting:
    RELAY = {"video_mode": "chronological_relay"}

    @pytest.mark.parametrize("prompt,expected", [
        ("How has AI governance discussion evolved over the years?", "chronological"),
        ("What changed since 2021 in AI ethics?", "chronological"),
        ("Deepfakes and elections", "topic"),
        ("What did speakers say about deepfakes?", "topic"),
    ])
    def test_chronological_needs_time_wording(self, prompt, expected):
        profile = {"raw_prompt": prompt, "intent_type": "topic_video"}
        assert intents.select(profile, self.RELAY).NAME == expected

    @pytest.mark.parametrize("topic_terms,expected", [([], "speaker"), (["media", "regulation"], "speaker_topic")])
    def test_speaker_topic_needs_a_topic(self, topic_terms, expected):
        profile = {"requested_speaker": "Cordel Green", "intent_type": "speaker_topic_summary",
                   "topic_terms": topic_terms}
        assert intents.select(profile, {}).NAME == expected

    # Case matrix rows 10 and 31 (docs/design/intent-layer-design.md §13), from the profile the prompt produces.
    @pytest.mark.parametrize("prompt,speaker,expected", [
        ("What did Cordell Green say?", "Cordel Green", "speaker"),       # an alias is not a topic
        ("What did Cordell Green say about regulation?", "Cordel Green", "speaker_topic"),
        ("Chapter 5", None, "theme_chapter"),                            # not an archive overview
        ("What is AI4IA?", None, "overview"),
    ])
    def test_profile_routes_to(self, prompt, speaker, expected):
        from app.pipeline.storyline_gen.prompt_intent import detect_prompt_intent

        records = [{"primary_chapter": "Ch 5: Open Data & Information Accessibility"}]
        profile = detect_prompt_intent(prompt, [], records, requested_speaker=speaker)
        assert intents.select(profile, {}).NAME == expected
