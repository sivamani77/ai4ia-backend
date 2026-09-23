"""Speaker prompts: initials detection and the panel-clip fallback."""
from __future__ import annotations

import pytest

from app.core.settings import settings
from app.pipeline.storyline_gen.prompt_intent import candidate_speaker_matches, detect_requested_speaker
from app.pipeline.storyline_gen.retrieval import hybrid_retrieve_candidates


def solo(fid, speaker, quote=0.5):
    return {"fragment_id": fid, "source_video": f"0001_AAAAAAAAAAA_{speaker} - Talk", "speaker": speaker,
            "speaker_attribution": "video-level", "speaker_roster": speaker, "quote_strength": quote,
            "text": "", "primary_theme": "", "primary_chapter": ""}


def panel(fid, roster, quote=0.9):
    return {"fragment_id": fid, "source_video": "0002_BBBBBBBBBBB_Europe Satellite Event Panel 1", "speaker": "",
            "speaker_attribution": "Multiple speakers", "speaker_roster": "; ".join(roster),
            "quote_strength": quote, "text": "", "primary_theme": "", "primary_chapter": ""}


class TestInitials:
    def test_initials_resolve_through_registry(self):
        assert detect_requested_speaker("a video with the talk by C Howell", []) == "Chuck Howell"
        assert detect_requested_speaker("what did C. Howell say", ["Chuck Howell"]) == "Chuck Howell"

    @pytest.mark.parametrize("prompt", [
        "Make A Video about AI in Africa",
        "an overview of AI governance",
        "c howell on sts",  # lowercase initials are not matched
    ])
    def test_ordinary_text_is_not_a_speaker(self, prompt):
        assert detect_requested_speaker(prompt, []) is None

    def test_similar_short_name_is_not_confused(self):
        known = ["A Singh", "Ashita Singh"]
        assert detect_requested_speaker("Ashita Singh on AI in India", known) == "Ashita Singh"
        assert detect_requested_speaker("keynote by Mr A Singh", known) == "A Singh"

    def test_full_names_still_win(self):
        assert detect_requested_speaker("Kodel Green on media regulation", []) == "Cordel Green"


class TestPanelMatch:
    def test_roster_counts_only_when_allowed(self):
        record = panel("p1", ["Dariia Opryshko", "Natali Helberger"])
        assert candidate_speaker_matches(record, "Dariia Opryshko")[0] is False
        assert candidate_speaker_matches(record, "Dariia Opryshko", allow_panel=True)[0] is True
        assert candidate_speaker_matches(dict(record, speaker_match_panel=True), "Dariia Opryshko")[0] is True

    def test_other_panelist_does_not_match(self):
        record = panel("p1", ["Dariia Opryshko", "Natali Helberger"])
        assert candidate_speaker_matches(record, "Stefan Baack", allow_panel=True)[0] is False


class TestPanelFallback:
    def run(self, records, speaker):
        return hybrid_retrieve_candidates({"requested_speaker": speaker}, records, [], top_k=50)

    def test_panel_only_speaker_gets_panel_clips(self):
        records = [panel("p1", ["Dariia Opryshko", "X Y"]), panel("p2", ["Dariia Opryshko"]), solo("s1", "Other Person")]
        result = self.run(records, "Dariia Opryshko")
        assert {r["fragment_id"] for r in result} == {"p1", "p2"}
        assert all(r["speaker_match_panel"] and r["speaker_match"] for r in result)

    def test_few_solo_clips_come_first_then_panels(self, monkeypatch):
        monkeypatch.setattr(settings, "speaker_panel_fallback_min_clips", 3)
        records = [solo("s1", "Cordel Green", 0.1), solo("s2", "Cordel Green", 0.2), panel("p1", ["Cordel Green"], 0.99)]
        result = [r["fragment_id"] for r in self.run(records, "Cordel Green")]
        assert result == ["s2", "s1", "p1"]

    def test_enough_solo_clips_means_no_panels(self, monkeypatch):
        monkeypatch.setattr(settings, "speaker_panel_fallback_min_clips", 2)
        records = [solo("s1", "Cordel Green"), solo("s2", "Cordel Green"), panel("p1", ["Cordel Green"])]
        assert [r["fragment_id"] for r in self.run(records, "Cordel Green")] == ["s1", "s2"]
