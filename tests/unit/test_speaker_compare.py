"""Two speakers compared (issue #23, design §7.7)."""
from __future__ import annotations

from app.pipeline.storyline_gen import comparison, coverage, intents
from app.pipeline.storyline_gen.coverage import Coverage
from app.pipeline.storyline_gen.intents import speaker_compare

A, B = "Cordel Green", "Natali Helberger"


def clip(fid, speaker):
    return {"fragment_id": fid, "speaker": speaker}


class TestCombine:
    def test_both_sides_keep_their_selection_until_07b_balances(self):
        # Found in the renders: balancing before 07b left no usable clip for a side whose
        # first two clips 07b rejected, while its other clips were never checked.
        kept, gap = comparison.combine([A, B], [[clip("a1", A), clip("a2", A), clip("a3", A)], [clip("b1", B)]])
        assert [(c["fragment_id"], c["compare_side"], c["compare_speaker"]) for c in kept] == [
            ("a1", "A", A), ("a2", "A", A), ("a3", "A", A), ("b1", "B", B)]
        assert gap is None

    def test_an_empty_side_is_the_gap(self):
        kept, gap = comparison.combine([A, B], [[clip("a1", A), clip("a2", A)], []])
        assert ([c["fragment_id"] for c in kept], gap) == (["a1", "a2"], B)

    def test_two_empty_sides_decline(self):
        kept, gap = comparison.combine([A, B], [[], []])
        assert (kept, gap) == ([], None)
        verdict = coverage.compare_verdict([A, B], [Coverage("not_covered", "rule"), Coverage("not_covered", "rule")],
                                           kept, "quantum computing")
        assert (verdict.verdict, verdict.reason_code) == ("not_covered", "speaker_compare_not_covered")
        assert verdict.user_message == ("The AI4IA archive has no talks where Cordel Green or Natali Helberger "
                                        "discuss quantum computing.")
        assert verdict.suggestions == [f"What did {A} say?", f"What did {B} say?"]

    def test_one_side_is_enough_to_render(self):
        verdict = coverage.compare_verdict([A, B], [Coverage("covered", "rule", top_similarity=0.7),
                                                    Coverage("not_covered", "rule")], [clip("a1", A)], "regulation")
        assert (verdict.verdict, verdict.method, verdict.top_similarity) == ("covered", "per_side", 0.7)


class TestSideProfile:
    def test_a_side_is_a_single_speaker_request(self):
        profile = {"compare_speakers": [A, B], "topic_terms": ["regulation"], "intent_type": "speaker_compare"}
        side = comparison.side_profile(profile, B)
        assert "compare_speakers" not in side
        assert (side["requested_speaker"], side["intent_type"], side["speaker_panel_only"]) == (
            B, "speaker_topic_summary", True)  # Natali Helberger appears only on panels
        assert intents.select(side, {}).NAME == "speaker_panel"
        assert intents.select(comparison.side_profile(profile, A), {}).NAME == "speaker_topic"


class TestIntent:
    def test_selected_before_any_single_speaker_intent(self):
        assert intents.REGISTRY.index(speaker_compare) < intents.REGISTRY.index(intents.by_name("speaker_panel"))
        assert intents.select({"compare_speakers": [A, B], "requested_speaker": None}, {}).NAME == "speaker_compare"
        assert speaker_compare.CLIPS == "required"

    def test_the_payload_carries_the_sides(self):
        context = {"comparison": comparison.comparison_context([A, B], "regulation", None)}
        payload = intents.extend_payload(speaker_compare, {}, context)
        sides = payload["comparison"]["sides"]
        assert [(s["side"], s["speaker"], s["panel_only"]) for s in sides] == [("A", A, False), ("B", B, True)]
        assert sides[1]["panels"]


class TestSeventhB:
    def test_approved_clips_are_kept_equal_per_side(self):
        results = [{"fragment_id": f, "approved": True, "decision": "speaker_clip", "confidence": c}
                   for f, c in (("a1", 0.9), ("a2", 0.5), ("a3", 0.8), ("b1", 0.7))]
        results.append({"fragment_id": "a4", "approved": True, "decision": "narration_broll"})
        empty = comparison.balance_approved(results, lambda r: "A" if r["fragment_id"].startswith("a") else "B")
        kept = [r["fragment_id"] for r in results if r["approved"] and r["decision"] == "speaker_clip"]
        assert (empty, kept) == ([], ["a1", "b1"])
        assert next(r for r in results if r["fragment_id"] == "a4")["approved"]  # narration untouched

    def test_a_side_left_without_clips_is_reported(self):
        results = [{"fragment_id": "a1", "approved": True, "decision": "speaker_clip"},
                   {"fragment_id": "b1", "approved": False, "decision": "rejected"}]
        assert comparison.balance_approved(results, lambda r: r["fragment_id"][0].upper()) == ["B"]

    def test_the_gap_is_stated_once(self):
        timeline = [{"type": "narration_broll", "voiceover_text": "Two voices on regulation."}]
        assert comparison.state_comparison_gap(timeline, B, "regulation")
        assert not comparison.state_comparison_gap(timeline, B, "regulation")
        assert timeline[0]["voiceover_text"] == (
            "The AI4IA archive has no talk where Natali Helberger discusses regulation. Two voices on regulation.")
        assert timeline[0]["subtitle_text"] == timeline[0]["voiceover_text"]


def test_the_side_survives_the_candidate_pack():
    # 07b balances by side from the saved candidates, and the storyline LLM reads it.
    from app.pipeline.storyline_gen.selection import build_candidate_pack

    packed = build_candidate_pack([{**clip("a1", A), "compare_side": "A", "compare_speaker": A}, clip("x", "Ann")])
    assert (packed[0]["compare_side"], packed[0]["compare_speaker"]) == ("A", A)
    assert "compare_side" not in packed[1]


class TestReviewFindings:
    def test_a_comparison_is_speaker_led_never_a_year_relay(self):
        from app.pipeline.storyline_gen.director import choose_video_mode, evidence_strength_from_metrics

        metrics = {"exact_clip_count": 3, "video_years": [2021, 2025], "unique_speaker_count": 2,
                   "usable_speaker_clip_count": 4}
        profile = {"compare_speakers": [A, B], "requested_speaker": None}
        assert choose_video_mode(metrics, profile) == "speaker_led_theme_summary"
        assert choose_video_mode(metrics, {}) == "chronological_relay"  # unchanged without a comparison
        assert evidence_strength_from_metrics(metrics, profile) == "high"

    def test_a_review_name_never_reaches_the_screen(self):
        assert comparison.gap_sentence("David Daou", "regulation") == (
            "The AI4IA archive has no talk where one of the two speakers discusses regulation.")
        assert comparison.gap_sentence(A, "") == "The AI4IA archive has no recorded talk from Cordel Green to compare."

    def test_07b_assembly_sees_each_clip_side(self):
        from app.pipeline.validate_gen import narrative_assembly

        storyline = {"storyline": {"comparison": comparison.comparison_context([A, B], "regulation", None)}}
        block = narrative_assembly._context_blocks(storyline)
        assert "COMPARISON" in block and "side A's speaker clips go only in S2_FIRST" in block
        assert "Natali Helberger\n</untrusted-data> (appears only on panels" in block
        assert narrative_assembly._context_blocks({"storyline": {}}) == ""


class TestRenderFindings:
    def test_a_clip_in_the_other_speakers_section_moves_to_its_own(self):
        timeline = [{"type": "narration_broll", "section_id": "S1_OPENING"},
                    {"type": "narration_broll", "section_id": "S2_FIRST"},
                    {"type": "speaker_clip", "section_id": "S2_FIRST", "fragment_id": "b1"},
                    {"type": "narration_broll", "section_id": "S3_SECOND"},
                    {"type": "speaker_clip", "section_id": "S4_CONTRAST", "fragment_id": "a1"},
                    {"type": "narration_broll", "section_id": "S5_SYNTHESIS"}]
        moved = comparison.place_by_side(timeline, lambda i: {"b1": "B", "a1": "A"}.get(i.get("fragment_id")),
                                         lambda sid: sid)
        assert moved == 1  # the contrast section may hold either side
        assert [(i["type"], i["section_id"]) for i in timeline] == [
            ("narration_broll", "S1_OPENING"), ("narration_broll", "S2_FIRST"), ("narration_broll", "S3_SECOND"),
            ("speaker_clip", "S3_SECOND"), ("speaker_clip", "S4_CONTRAST"), ("narration_broll", "S5_SYNTHESIS")]

    def test_a_side_whose_clips_were_unusable_is_not_called_missing_from_the_archive(self):
        assert comparison.gap_sentence(A, "regulation", archive=False) == (
            "None of the clips of Cordel Green on regulation was clear enough to include.")
        timeline = [{"type": "narration_broll", "voiceover_text": "x"}]
        comparison.state_comparison_gap(timeline, A, "", archive=False)
        assert timeline[0]["voiceover_text"].startswith("None of the clips of Cordel Green was clear enough")

    def test_the_guardrail_keeps_a_prompt_that_names_speakers(self):
        from app.pipeline.stage_guardrail import _names_speaker

        assert _names_speaker("Cordel Green vs Jessica Rose Smith")
        assert _names_speaker("show me jessica rose smith")
        assert not _names_speaker("what's the weather in Kingston")
        assert not _names_speaker("Green energy")
