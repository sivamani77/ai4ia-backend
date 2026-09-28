"""Year-scoped overview and topic films (issue #23, design §7.6)."""
from __future__ import annotations

import numpy as np

from app.pipeline.storyline_gen import retrieval
from app.pipeline.storyline_gen.corpus_facts import build_corpus_facts, format_facts_for_prompt
from app.pipeline.validate_gen.archive_answer import ensure_asked_figures

RECORDS = [
    {"fragment_id": "a", "source_video": "0001_v1_Ann Lee - Open data", "video_id": "v1", "video_year": 2023,
     "primary_theme": "T1", "primary_chapter": "Ch 5: Open Data"},
    {"fragment_id": "b", "source_video": "0001_v1_Ann Lee - Open data", "video_id": "v1", "video_year": 2023,
     "primary_theme": "T1", "primary_chapter": "Ch 5: Open Data"},
    {"fragment_id": "c", "source_video": "0002_v2_Tom Reed - Health", "video_id": "v2", "video_year": 2024,
     "primary_theme": "T2", "primary_chapter": "Ch 13: Health"},
]


def test_restrict_to_year_narrows_records_and_matrix(monkeypatch):
    monkeypatch.setattr(retrieval, "_FILTERED_RECORDS", list(RECORDS))
    monkeypatch.setattr(retrieval, "_FILTERED_MATRIX", np.eye(3, dtype=np.float32))
    kept = retrieval.restrict_to_year(2023)
    assert [r["fragment_id"] for r in kept] == ["a", "b"]
    assert retrieval.get_filtered_records_or_load() == kept
    assert retrieval._FILTERED_MATRIX.tolist() == [[1, 0, 0], [0, 1, 0]]


def test_chroma_search_over_fetches_for_a_small_allowed_set(monkeypatch):
    asked = {}

    class Collection:
        def count(self):
            return 5000

        def query(self, **kwargs):
            asked.update(kwargs)
            return {"ids": [["a"]], "distances": [[0.2]], "metadatas": [[{"video_year": 2023}]]}

    monkeypatch.setattr(retrieval, "_get_chroma_collection", lambda: Collection())
    monkeypatch.setattr(retrieval, "_query_vector", lambda prompt: np.zeros(2, dtype=np.float32))
    retrieval._semantic_rank_via_chroma("p", 90, allowed_ids={f"f{i}" for i in range(1000)})
    assert asked["n_results"] == 90 * 5 * 5  # a fifth of the collection is allowed
    retrieval._semantic_rank_via_chroma("p", 90)
    assert asked["n_results"] == 90


def test_year_facts_say_their_scope_and_drop_the_year_range():
    facts = build_corpus_facts(RECORDS[:2], registry_speaker_count=12, year=2023)
    block = format_facts_for_prompt(facts)
    assert "- Scope: AI4IA 2023 only." in block
    assert "- Talks in the archive: 1" in block and "- Speakers: over 10" in block
    assert "Years covered" not in block
    assert "Years covered" in format_facts_for_prompt(build_corpus_facts(RECORDS))


def test_registry_speaker_count_for_some_videos():
    from app.services.speaker_registry import get_registry

    registry = get_registry()
    video_id = next(iter(registry.video_ids()))
    count = registry.recorded_speaker_count({video_id})
    assert count == len(registry.video(video_id).names)
    assert registry.recorded_speaker_count(set()) == 0
    assert registry.recorded_speaker_count() >= count


def test_an_asked_figure_for_a_year_is_stated_for_that_year():
    facts = format_facts_for_prompt(build_corpus_facts(RECORDS[:2], registry_speaker_count=12, year=2023))
    timeline = [{"type": "narration_broll", "voiceover_text": "The 2023 edition looked at open data."}]
    assert ensure_asked_figures(timeline, "How many talks were at AI4IA 2023?", facts) == ["AI4IA 2023 had 1 talk."]
    # A span of years makes no sense for one edition.
    timeline = [{"type": "narration_broll", "voiceover_text": "x"}]
    assert ensure_asked_figures(timeline, "Which years does AI4IA 2023 cover?", facts) == []


def test_a_topic_not_covered_in_a_year_says_the_year():
    from app.pipeline.storyline_gen import coverage

    profile = {"raw_prompt": "quantum farming in 2024", "topic_terms": ["quantum", "farming"], "year": 2024}
    code, message, _ = coverage.decline("topic", profile, [], ["quantum farming"], [])
    assert code == "topic_not_covered"
    assert message.startswith("The AI4IA archive has no 2024 talks on ")


def test_over_is_never_the_exact_count():
    from app.pipeline.storyline_gen.corpus_facts import _over

    assert [_over(n) for n in (307, 65, 60, 12, 7)] == ["over 300", "over 60", "over 50", "over 10", "7"]


class TestReviewFindings:
    def test_since_a_year_is_not_a_one_year_film(self):
        from app.pipeline.storyline_gen.route import profile_for_route

        route = {"method": "router", "intent": "chronological", "speakers": [], "topic_terms": ["regulation"],
                 "year": 2023}
        assert profile_for_route("How has AI regulation evolved since 2023?", [], RECORDS, route)["year"] is None

    def test_a_chapter_with_no_talks_that_year_declines(self):
        from app.pipeline.storyline_gen import coverage

        profile = {"year": 2023, "chapter_hint": "Ch 13: Health", "topic_terms": []}
        year_records = RECORDS[:2]  # 2023 has only chapter 5
        result = coverage.assess("theme_chapter", profile, [], "Chapter 13 in 2023", None, floor=0.68, low=0.6,
                                 records=year_records)
        assert (result.verdict, result.method, result.reason_code) == ("not_covered", "year_scope", "topic_not_covered")
        assert result.user_message.startswith("The AI4IA archive has no 2023 talks on Ch 13: Health.")
        profile["chapter_hint"] = "Ch 5: Open Data"
        assert coverage.assess("theme_chapter", profile, [], "p", None, floor=0.68, low=0.6,
                               records=year_records).method != "year_scope"

    def test_a_year_left_empty_declines_any_film(self):
        from app.pipeline.storyline_gen import coverage

        result = coverage.assess("overview", {"year": 2022}, [], "AI4IA 2022", None, floor=0.68, low=0.6, records=[])
        assert result.verdict == "not_covered"

    def test_one_talk_is_singular(self):
        facts = format_facts_for_prompt(build_corpus_facts(RECORDS[:2], registry_speaker_count=12, year=2023))
        timeline = [{"type": "narration_broll", "voiceover_text": "x"}]
        assert ensure_asked_figures(timeline, "How many talks were at AI4IA 2023?", facts) == ["AI4IA 2023 had 1 talk."]

    def test_the_semantic_cache_never_matches_another_year(self):
        from app.services.storyline_cache import _years

        assert _years("Overview of AI4IA 2024") != _years("Overview of AI4IA 2023")
        assert _years("AI and health") == _years("health and AI") == set()

    def test_the_route_version_changed_with_this_pr(self):
        from app.services import intent_router

        assert intent_router.VALIDATOR_VERSION == "2"
