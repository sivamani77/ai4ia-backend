"""Stage 07b keeps real speaker clips through validation and repair."""
from __future__ import annotations

import pytest

from app.pipeline.validate_gen.quality_gates import (
    _build_repair_prompt,
    check_quality_gates,
    repair_can_help,
    repair_is_better,
    repair_shelf,
    speaker_clip_target,
    unchecked_speaker_candidates,
)

SECTIONS = [{"section_id": "S1_HOOK", "title": "Hook", "purpose": "open"}]
LONG_ID = "0147_B-B0ilBRyLo_AI4IA Conference 2023 - Opening Session_0031"


def clip(fid, kind="speaker_clip", **extra):
    return {"type": kind, "section_id": "S1_HOOK", "fragment_id": fid, "speaker": "Cordel Green",
            "voiceover_text": "Narration.", "visual_asset_prompt": "p", **extra}


def verdict(fid, decision="speaker_clip", approved=True):
    return {"fragment_id": fid, "decision": decision, "approved": approved, "speaker": "Cordel Green"}


class TestSpeakerClipTarget:
    @pytest.mark.parametrize("approved,expected", [(0, 0), (1, 1), (2, 2), (5, 2)])
    def test_lowered_to_what_was_approved(self, approved, expected):
        verdicts = [verdict(f"f{i}") for i in range(approved)]
        assert speaker_clip_target(verdicts) == expected

    def test_counts_only_approved_speaker_clips(self):
        verdicts = [verdict("a", approved=False), verdict("b", decision="narration_broll"), verdict("c")]
        assert speaker_clip_target(verdicts) == 1

    def test_one_clip_passes_when_only_one_was_approved(self):
        timeline = [clip("a")]
        assert check_quality_gates(timeline, SECTIONS, speaker_target=1)[1]["final_timeline_approved"]
        assert not check_quality_gates(timeline, SECTIONS)[1]["final_timeline_approved"]

    def test_no_speaker_clip_never_passes(self):
        timeline = [clip("a", kind="narration_broll")]
        assert not check_quality_gates(timeline, SECTIONS, speaker_target=0)[1]["final_timeline_approved"]


class TestRepairShelf:
    def test_unused_approved_clips_speaker_clips_first(self):
        verdicts = [verdict("used"), verdict("broll", decision="narration_broll"),
                    verdict("rejected", approved=False), verdict("spare")]
        shelf = repair_shelf([clip("used")], verdicts)
        assert [c["fragment_id"] for c in shelf] == ["spare", "broll"]


class TestRepairCanHelp:
    SPEAKER_FAIL = {"has_speaker_clips": True, "final_timeline_approved": False, "all_sections_covered": True}

    def test_not_when_no_speaker_clip_is_available(self):
        assert not repair_can_help(self.SPEAKER_FAIL, [verdict("b", decision="narration_broll")])

    def test_when_a_speaker_clip_is_on_the_shelf(self):
        assert repair_can_help(self.SPEAKER_FAIL, [verdict("spare")])

    def test_when_another_gate_fails(self):
        gates = {**self.SPEAKER_FAIL, "all_sections_covered": False}
        assert repair_can_help(gates, [])


class TestRepairIsBetter:
    PASS = {"has_speaker_clips": True, "final_timeline_approved": True}
    FAIL = {"has_speaker_clips": True, "final_timeline_approved": False}

    def test_rejects_losing_a_speaker_clip(self):
        # The Cordel Green case: the only clip turned into narration.
        before = ([clip("a")], self.FAIL)
        after = ([clip("a", kind="narration_broll")], {**self.FAIL, "has_speaker_clips": False})
        assert not repair_is_better(before, after)

    def test_rejects_passing_fewer_gates(self):
        assert not repair_is_better(([clip("a")], self.PASS), ([clip("a")], self.FAIL))

    def test_accepts_adding_a_speaker_clip(self):
        assert repair_is_better(([clip("a")], self.FAIL), ([clip("a"), clip("b")], self.PASS))


class TestUncheckedSpeakerCandidates:
    def test_rank_order_skipping_checked_and_ineligible(self):
        shortlist = [
            {"fragment_id": "checked", "speaker_clip_allowed": True},
            {"fragment_id": "visual_only", "speaker_clip_allowed": False, "segment_type": "archive_visual"},
            {"fragment_id": "s1", "speaker_clip_allowed": True},
            {"fragment_id": "s1", "speaker_clip_allowed": True},
            {"fragment_id": "s2", "speaker_soundbite_allowed": True},
            {"fragment_id": "s3", "speaker_clip_allowed": True},
        ]
        picked = unchecked_speaker_candidates(shortlist, {"checked"}, limit=2)
        assert [c["fragment_id"] for c in picked] == ["s1", "s2"]


class TestRepairPrompt:
    def test_states_the_real_rule_and_offers_the_shelf(self):
        _, user = _build_repair_prompt(
            current_timeline=[clip("a", quote_complete=True)],
            quality_gates={"final_timeline_approved": False, "offending_items": {}},
            narrative_sections=SECTIONS,
            user_prompt="What did Cordel Green say about media regulation?",
            iteration=1,
            shelf=[verdict(LONG_ID)],
            speaker_target=2,
        )
        assert "has 1 speaker clip(s) and needs at least 2" in user
        assert f'"fragment_id": "{LONG_ID}"' in user
        assert "Never demote a" in user and "quote_complete=true" in user
