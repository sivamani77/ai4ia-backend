"""Counting and ranking, computed by code (issue #23, design §7.8)."""
from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.storyline_gen import counting, coverage, intents, retrieval
from app.pipeline.storyline_gen.intents import counting as counting_intent
from app.services.speaker_registry import get_registry


def fragment(fid, video_id, source_video, text):
    return {"fragment_id": fid, "video_id": video_id, "source_video": source_video, "text": text}


def single_and_panel_videos():
    registry = get_registry()
    single = next(vid for vid in registry.video_ids() if registry.video(vid).kind == "single"
                  and all((registry.speakers.get(n) or {}).get("status", "confirmed") == "confirmed"
                          for n in registry.video(vid).names))
    panel = next(vid for vid in registry.video_ids() if registry.video(vid).kind == "panel")
    return single, panel, registry.video(single).names[0]


class TestCountTopic:
    def setup_method(self):
        self.single, self.panel, self.speaker = single_and_panel_videos()
        s, p = f"0001_{self.single}_Speaker - Trust online", f"0002_{self.panel}_Panel 1 - Media"
        self.records = [
            fragment("s1", self.single, s, "deepfakes erode trust"),
            fragment("s2", self.single, s, "a deepfake video spread"),
            fragment("s3", self.single, s, "unrelated words"),        # no term
            fragment("p1", self.panel, p, "deepfakes in elections"),
            fragment("p2", self.panel, p, "synthetic media and deepfakes"),
            fragment("o1", "other", "0003_other_X - Y", "deepfakes once"),  # one passage only
        ]
        self.scores = {"s1": 0.80, "s2": 0.70, "s3": 0.90, "p1": 0.75, "p2": 0.69, "o1": 0.95}

    def test_a_talk_needs_two_passages_that_score_and_name_the_topic(self):
        facts = counting.count_topic(self.records, self.scores, ["deepfake"], floor=0.68)
        assert (facts["talk_count"], facts["speaker_count"], facts["panel_count"]) == (2, 1, 1)
        assert facts["top_speakers"] == [(self.speaker, 2)]
        assert [t for t, _ in facts["top_talks"]] == ["Trust online", "Media"]
        assert facts["talks"] == {self.records[0]["source_video"], self.records[3]["source_video"]}

    def test_a_passage_below_the_floor_does_not_count(self):
        self.scores["s2"] = 0.60
        facts = counting.count_topic(self.records, self.scores, ["deepfake"], floor=0.68)
        assert (facts["talk_count"], facts["speaker_count"]) == (1, 0)

    def test_a_title_naming_the_topic_counts_as_the_term(self):
        # The coverage test's term check reads the talk title too: a talk called "Deepfakes" discusses deepfakes.
        # Its passages still have to reach the similarity floor.
        for r in self.records[:3]:
            r["source_video"] = r["source_video"].replace("Trust online", "Deepfakes and trust")
        self.scores.update(s2=0.60)
        facts = counting.count_topic(self.records, self.scores, ["deepfake"], floor=0.68)
        assert facts["top_speakers"] == [(self.speaker, 2)]  # s1 and s3 (by title), not s2 (below the floor)

    def test_the_registry_id_comes_from_the_file_name(self):
        # A stored video_id is cut at an underscore ("DMaOd1" for "DMaOd1_0lko"); the file name is whole.
        for r in self.records[:3]:
            r["video_id"] = "cut"
        facts = counting.count_topic(self.records, self.scores, ["deepfake"], floor=0.68)
        assert facts["speaker_count"] == 1

    def test_excluded_terms_leave_the_count(self):
        facts = counting.count_topic(self.records, self.scores, ["deepfake"], floor=0.68, exclude_terms=["election"])
        assert facts["panel_count"] == 0  # the panel keeps only one passage without "elections"

    def test_expanded_terms_count_too(self):
        facts = counting.count_topic(self.records, self.scores, ["synthetic media"], floor=0.68)
        assert facts["talk_count"] == 0  # only one passage says it
        self.records[3]["text"] = "synthetic media in elections"
        assert counting.count_topic(self.records, self.scores, ["synthetic media"], floor=0.68)["talk_count"] == 1

    def test_zero_is_answered_by_a_decline(self):
        facts = counting.count_topic(self.records, self.scores, ["quantum"], floor=0.68)
        verdict = coverage.count_verdict(facts, "quantum computing", 2024, [{"primary_theme": "AI Ethics"}])
        assert (verdict.verdict, verdict.reason_code) == ("not_covered", "count_none")
        assert verdict.user_message == ("No talks in the AI4IA archive from 2024 discuss quantum computing. "
                                        "Related themes it does cover: AI Ethics.")


class TestAnswer:
    FACTS = counting.format_count_facts(
        {"talk_count": 12, "speaker_count": 9, "panel_count": 2, "top_speakers": [("Cordel Green", 7), ("Ann Lee", 5)],
         "unnamed_speakers": 1, "top_speaker_unnamed": False, "top_talks": [("Harmful deepfakes", 6), ("Media", 3)]},
        "deepfakes", 2024)

    def test_the_facts_block(self):
        assert self.FACTS.splitlines()[:3] == [
            "- Topic: deepfakes", "- Year: 2024",
            "- Talks that discuss it: 12 (a talk counts when at least 2 of its passages discuss it)"]
        assert "- Speakers who discuss it most (passages): Cordel Green (7); Ann Lee (5)" in self.FACTS
        assert '- Talks that discuss it most (passages): "Harmful deepfakes" (6); "Media" (3)' in self.FACTS

    def test_the_answer_is_added_when_the_narration_leaves_it_out(self):
        timeline = [{"type": "narration_broll", "voiceover_text": "Deepfakes worry many speakers."}]
        added = counting.ensure_count_answer(timeline, self.FACTS, "talks", "count")
        assert added == "12 talks in AI4IA 2024 discuss deepfakes."
        assert timeline[0]["voiceover_text"].startswith(added) and timeline[0]["subtitle_text"]

    @pytest.mark.parametrize("text,stated", [
        ("In 2024, 12 talks took on deepfakes.", True),
        ("Twelve talks took on deepfakes.", True),
        ("12 of the talks took on deepfakes.", True),
        ("Deepfakes came up often.", False),            # "often" holds no 12
        ("Three talks, 12 panels.", False),             # 12 is not next to "talks"
    ])
    def test_the_figure_must_sit_next_to_its_noun(self, text, stated):
        timeline = [{"type": "narration_broll", "voiceover_text": text}]
        assert (counting.ensure_count_answer(timeline, self.FACTS, "talks", "count") is None) == stated

    def test_one_and_two_in_words(self):
        facts = counting.format_count_facts({"talk_count": 1, "speaker_count": 1, "panel_count": 0,
                                             "top_speakers": [], "unnamed_speakers": 0, "top_talks": []}, "space law")
        assert counting.answer_sentence(facts, "talks", "count") == "1 talk in the AI4IA archive discusses space law."
        timeline = [{"type": "narration_broll", "voiceover_text": "One talk in the archive covers it."}]
        assert counting.ensure_count_answer(timeline, facts, "talks", "count") is None

    def test_speakers_and_ranking(self):
        assert counting.answer_sentence(self.FACTS, "speakers", "count") == (
            "9 speakers in AI4IA 2024 discuss deepfakes in their own talks.")
        assert counting.answer_sentence(self.FACTS, "speakers", "rank") == (
            "Of the speakers in AI4IA 2024, Cordel Green discusses deepfakes the most.")
        assert counting.answer_sentence(self.FACTS, "talks", "rank") == (
            'The talk in AI4IA 2024 that discusses deepfakes most is "Harmful deepfakes".')
        timeline = [{"type": "narration_broll", "voiceover_text": "Cordel Green leads this conversation."}]
        assert counting.ensure_count_answer(timeline, self.FACTS, "speakers", "rank") is None

    def test_no_single_winner_for_a_tie_or_a_hidden_top_speaker(self):
        tie = self.FACTS.replace("Ann Lee (5)", "Ann Lee (7)")
        assert counting.answer_sentence(tie, "speakers", "rank") == (
            "Speakers who discuss deepfakes most in AI4IA 2024 include Cordel Green and Ann Lee.")
        hidden = self.FACTS + "\n- The speaker who discusses it most is not named here"
        assert counting.answer_sentence(hidden, "speakers", "rank").startswith("Speakers who discuss deepfakes most")

    def test_a_ranking_with_nobody_named_falls_back_to_the_count(self):
        facts = counting.format_count_facts({"talk_count": 3, "speaker_count": 0, "panel_count": 3,
                                             "top_speakers": [], "unnamed_speakers": 0, "top_talks": []}, "ethics")
        assert counting.answer_sentence(facts, "speakers", "rank") == "3 talks in the AI4IA archive discuss ethics."

    def test_a_topic_ending_in_a_year_is_not_the_year(self):
        facts = counting.format_count_facts({"talk_count": 3, "speaker_count": 0, "panel_count": 0,
                                             "top_speakers": [], "unnamed_speakers": 0, "top_talks": []}, "AI in 2030")
        assert counting.answer_sentence(facts, "talks", "count") == "3 talks in the AI4IA archive discuss AI in 2030."


class TestScoring:
    def test_every_record_is_scored_in_local_mode(self, monkeypatch):
        monkeypatch.setattr(retrieval, "_chroma_mode", lambda: False)
        monkeypatch.setattr(retrieval, "_FILTERED_RECORDS", [{"fragment_id": "a"}, {"fragment_id": "b"}])
        monkeypatch.setattr(retrieval, "_FILTERED_MATRIX", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        monkeypatch.setattr(retrieval, "embed_query", lambda q: np.array([0.6, 0.8], dtype=np.float32))
        assert retrieval.score_every_record("q") == {"a": 0.6, "b": 0.8}


    def test_a_failed_fetch_fails_the_count_instead_of_scoring_zero(self, monkeypatch):
        monkeypatch.setattr(retrieval, "_chroma_mode", lambda: True)
        monkeypatch.setattr(retrieval, "_FILTERED_RECORDS", [{"fragment_id": "a"}, {"fragment_id": "b"}])
        monkeypatch.setattr(retrieval, "_backfill_semantic_scores",
                            lambda scored, q: scored[0].update(semantic_score=0.7, semantic_score_backfilled=True))
        with pytest.raises(RuntimeError, match="Could not score 1 of 2"):
            retrieval.score_every_record("q")


class TestIntent:
    def test_counting_is_chosen_for_a_counting_profile(self):
        profile = {"intent_type": "counting", "topic_terms": ["deepfake"]}
        assert intents.select(profile, {}).NAME == "counting"
        assert counting_intent.CLIPS == "optional"
        payload = intents.extend_payload(counting_intent, {}, {"count_facts": "- Talks that discuss x: 3 ("})
        assert payload["count_facts"].startswith("- Talks")


class TestWording:
    def test_the_topic_is_kept_as_written_without_filler(self):
        from tests.unit.test_intent_router import check

        route = check("How many talks were about AI ethics?", intent="counting", topic="AI ethics",
                      aggregate="count", count_unit="talks")
        assert (route.topic_terms, route.topic_text) == (["ethic"], "ethics")

    def test_terms_match_by_stem(self):
        from app.pipeline.storyline_gen.coverage import has_term

        terms = counting.count_terms(["regulating", "synthetic media"])
        assert terms == ["regulat", "synthetic media"]
        assert has_term({"text": "the regulation of platforms"}, terms[0])
        assert not has_term({"text": "the regulator"}, "regulating")  # what the literal form missed
