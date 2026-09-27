"""Intent router (issue #23): the LLM reads the prompt's meaning; code checks every fact it returns."""
from __future__ import annotations

import pytest

from app.core.settings import settings
from app.services import intent_router
from app.services.corpus_catalog import Catalog
from app.services.intent_router import Route, validate

_REAL_ASK = intent_router._ask  # conftest replaces it for every test
HAIKU = "0100_abc_Shanti Williams - When I asked AI to compose a Haiku"
CATALOG = Catalog(
    frozenset(range(1, 26)), 2021, 2025,
    frozenset({"when i asked ai to compose a haiku", "opening session"}),
    ("AI in Health, Science & Environment", "AI Ethics, Governance & Policy"),
    {5: "Ch 5: Open Data & Information Accessibility", 13: "Ch 13: AI in Healthcare & Medicine",
     21: "Ch 21: Indigenous Knowledge, Language & Localized AI"},
    {"when i asked ai to compose a haiku": (HAIKU,), "opening session": ("0001_a_Opening Session", "0002_b_Opening Session")},
)


def check(prompt, **raw):
    return validate(prompt, {"speakers": [], "topic": "", "exclude": [], "language": "en", **raw}, CATALOG)


class TestSpeakers:
    def test_answer_words_are_not_a_topic(self):
        # The uat trigger: "contributed" made it speaker_topic, and coverage declined it.
        route = check("what did cordel green contributed in AI4IA?", intent="speaker",
                      speakers=["cordel green"], topic="contributed")
        assert (route.intent, route.speakers, route.topic_terms) == ("speaker", ["Cordel Green"], [])

    def test_a_name_not_in_the_prompt_is_dropped(self):
        route = check("AI in education", intent="speaker", speakers=["Cordel Green"], topic="education")
        assert (route.intent, route.speakers) == ("topic", [])
        assert route.dropped == [{"slot": "speaker", "value": "Cordel Green", "reason": "not_in_prompt"}]

    def test_a_misspelled_name_gets_did_you_mean(self):
        route = check("wat did cordell grene say abt regulations", intent="speaker",
                      speakers=["cordell grene"], topic="regulations")
        assert route.decline.reason_code == "speaker_did_you_mean"
        assert route.decline.suggestions == ["Cordel Green"]

    def test_an_unknown_person_declines_with_a_topic_suggestion(self):
        route = check("What did Bartholomew Quenneville say about AI in music?", intent="speaker",
                      speakers=["Bartholomew Quenneville"], topic="AI in music")
        assert route.decline.reason_code == "speaker_not_found"
        assert "Check the spelling" in route.decline.message
        assert route.decline.suggestions == ["What did speakers say about music?"]

    def test_a_one_word_non_person_goes_on_as_a_topic(self):
        route = check("What did Nigeria say about AI policy?", intent="speaker", speakers=["Nigeria"], topic="AI policy")
        assert (route.decline, route.intent, route.topic_terms) == (None, "topic", ["policy"])

    def test_one_speaker_on_a_topic_route_is_a_speaker_film(self):
        route = check("Cordel Green regulation AI Caribbean media", intent="topic", speakers=["Cordel Green"],
                      topic="regulation Caribbean media")
        assert (route.intent, route.speakers) == ("speaker", ["Cordel Green"])
        assert route.topic_terms == ["regulation", "caribbean", "media"]

    def test_a_string_instead_of_a_list_is_accepted(self):
        route = check("Show me Jessica Rose Smith", intent="speaker", speakers="Jessica Rose Smith")
        assert route.speakers == ["Jessica Rose Smith"]


class TestComparison:
    def test_two_speakers(self):
        route = check("Compare Cordel Green and Natali Helberger on regulation", intent="comparison",
                      speakers=["Cordel Green", "Natali Helberger"], topic="regulation")
        assert (route.intent, route.speakers, route.topic_terms) == (
            "comparison", ["Cordel Green", "Natali Helberger"], ["regulation"])

    def test_a_third_speaker_is_dropped_and_recorded(self):
        route = check("Compare Cordel Green, Natali Helberger and David Daou", intent="comparison",
                      speakers=["Cordel Green", "Natali Helberger", "David Daou"])
        assert route.speakers == ["Cordel Green", "Natali Helberger"]
        assert {"slot": "speaker", "value": "David Daou", "reason": "comparison_takes_two"} in route.dropped

    def test_two_years_and_no_speakers_is_chronological(self):
        route = check("How did AI4IA 2021 compare to AI4IA 2025?", intent="comparison", year=2021)
        assert (route.intent, route.year) == ("chronological", None)


class TestExclusions:
    def test_an_excluded_speaker_is_never_the_subject(self):
        route = check("AI regulation but not Cordel Green", intent="topic", speakers=["Cordel Green"],
                      topic="AI regulation", exclude=["Cordel Green"])
        assert (route.intent, route.speakers, route.excluded_speakers) == ("topic", [], ["Cordel Green"])
        assert route.topic_terms == ["regulation"]

    def test_an_excluded_subject_leaves_the_topic(self):
        route = check("Make a video about Cordel Green but not about regulation", intent="speaker",
                      speakers=["Cordel Green"], topic="regulation", exclude=["regulation"])
        assert (route.intent, route.topic_terms, route.exclude_terms) == ("speaker", [], ["regulation"])

    def test_an_exclusion_not_in_the_prompt_is_ignored(self):
        route = check("AI regulation", intent="topic", topic="regulation", exclude=["deepfakes"])
        assert route.exclude_terms == []


class TestYearsAndChapters:
    def test_a_year_must_be_written_in_the_prompt(self):
        route = check("What happened at the latest AI4IA conference?", intent="overview", year=2025)
        assert (route.intent, route.year) == ("overview", None)
        assert route.dropped == [{"slot": "year", "value": 2025, "reason": "not_in_prompt"}]

    def test_a_year_outside_the_archive_declines(self):
        route = check("What happened at AI4IA 2019?", intent="overview", year=2019)
        assert route.decline.reason_code == "year_out_of_range"

    def test_a_year_given_as_text_is_read(self):
        route = check("misinformation during elections, only 2024 talks", intent="topic",
                      topic="misinformation elections", year="2024")
        assert route.year == 2024

    @pytest.mark.parametrize("prompt,chapter", [("Chapter five", 5), ("chapter 13", 13), ("ch. 21", 21)])
    def test_chapter_numbers_in_words_or_digits(self, prompt, chapter):
        route = check(prompt, intent="theme_chapter", chapter=chapter)
        assert (route.intent, route.chapter) == ("theme_chapter", chapter)

    def test_a_chapter_outside_the_archive_declines(self):
        assert check("Chapter thirty", intent="theme_chapter", chapter=30).decline.reason_code == "chapter_not_found"

    def test_a_chapter_name_in_the_theme_field_gives_its_number(self):
        route = check("Indigenous Knowledge, Language & Localized AI", intent="theme_chapter",
                      theme="Indigenous Knowledge, Language & Localized AI")
        assert (route.intent, route.chapter) == ("theme_chapter", 21)

    def test_a_guessed_chapter_on_a_topic_is_dropped(self):
        route = check("Tell me about open data", intent="topic", topic="open data", chapter=5)
        assert (route.intent, route.chapter) == ("topic", None)

    def test_a_topic_narrower_than_the_theme_is_a_topic(self):
        route = check("AI in health for rural women", intent="theme_chapter", topic="health rural women",
                      theme="AI in Health, Science & Environment")
        assert route.intent == "topic"


class TestOtherIntents:
    def test_counting_without_a_subject_is_an_overview(self):
        route = check("How many speakers are there?", intent="counting", topic="speakers", aggregate="count",
                      count_unit="speakers")
        assert route.intent == "overview"

    def test_counting_keeps_its_unit(self):
        route = check("Which speakers talked the most about ethics?", intent="counting", topic="ethics",
                      aggregate="rank", count_unit="speakers")
        assert (route.intent, route.topic_terms, route.aggregate, route.count_unit) == (
            "counting", ["ethic"], "rank", "speakers")

    def test_a_loose_title_finds_one_talk(self):
        route = check("the haiku talk", intent="title", title="the haiku talk", topic="haiku")
        assert (route.intent, route.source_video) == ("title", HAIKU)

    def test_a_title_matching_several_talks_is_a_topic(self):
        route = check("the opening session", intent="title", title="opening session", topic="opening session")
        assert (route.intent, route.source_video) == ("topic", None)

    def test_digits_are_never_topic_words(self):
        route = check("first give me an overview of 2023", intent="overview", topic="2023", year=2023)
        assert (route.intent, route.topic_terms, route.year) == ("overview", [], 2023)

    def test_an_english_topic_must_come_from_the_prompt(self):
        assert check("AI in education", intent="topic", topic="quantum farming").topic_terms == []

    def test_a_translated_topic_is_kept(self):
        route = check("Qu'est-ce que Natali Helberger a dit sur la liberté de la presse ?", intent="speaker",
                      speakers=["Natali Helberger"], topic="press freedom", language="fr")
        assert (route.language, route.speakers, route.topic_terms) == ("fr", ["Natali Helberger"], ["press", "freedom"])

    @pytest.mark.parametrize("raw", [None, "speaker", {"intent": "dance"}, {"speakers": ["x"]}])
    def test_unusable_output_falls_back(self, raw):
        route = validate("anything", raw, CATALOG)
        assert (route.method, route.dropped) == ("fallback", [{"slot": "route", "reason": "invalid_output"}])


class TestRoute:
    def test_disabled_uses_the_rules(self, monkeypatch):
        monkeypatch.setattr(settings, "intent_router_enabled", False)
        assert intent_router.route("anything", CATALOG).method == "rule"

    def test_no_catalog_falls_back(self):
        assert intent_router.route("anything", None).dropped == [{"slot": "route", "reason": "catalog_unavailable"}]

    def test_a_failed_call_falls_back(self):
        route = intent_router.route("anything", CATALOG)  # conftest's _ask raises
        assert (route.method, route.dropped) == ("fallback", [{"slot": "route", "reason": "router_error"}])
        assert route.latency_ms is not None

    def test_the_call_has_its_own_timeout_and_one_attempt(self, monkeypatch):
        from app.services import llm_service

        seen = {}

        def fake_post(url, body, timeout, attempts, record_failure):
            seen.update(timeout=timeout, attempts=attempts, record_failure=record_failure,
                        system=body["messages"][0]["content"])
            return {"choices": [{"message": {"content": '{"intent": "topic", "topic": "education"}'}}]}, 0

        monkeypatch.setattr(intent_router, "_ask", _REAL_ASK)
        monkeypatch.setattr(llm_service, "_post_with_retries", fake_post)
        route = intent_router.route("AI in education", CATALOG)
        assert (route.method, route.intent, route.topic_terms) == ("router", "topic", ["education"])
        assert (seen["timeout"], seen["attempts"]) == (settings.intent_router_timeout_seconds, 1)
        assert seen["record_failure"] is False
        assert "Ch 13: AI in Healthcare & Medicine" in seen["system"]

    def test_version_changes_with_the_model(self, monkeypatch):
        before = intent_router.router_version()
        monkeypatch.setattr(settings, "intent_router_model", "another-model")
        assert intent_router.router_version() != before

    def test_route_round_trips(self):
        route = check("Show me Jessica Rose Smith", intent="speaker", speakers=["Jessica Rose Smith"])
        assert Route.from_dict(route.to_dict()).to_dict() == route.to_dict()


class TestStageResolve:
    PROMPT = "what did cordel green contributed in AI4IA?"

    def run(self, monkeypatch, prompt=None, context=None, job_id=None):
        from app.pipeline import stage_resolve
        from app.pipeline.base import PipelineContext
        from app.services import job_store
        from app.services import video_job_runner as runner

        for name in ("append_event", "load_job", "save_job"):
            monkeypatch.setattr(stage_resolve, name, getattr(job_store, name))
        monkeypatch.setattr(stage_resolve, "get_catalog", lambda: CATALOG)
        prompt = prompt or self.PROMPT
        job_id = job_id or runner.create_video_job({"prompt": prompt, "target_duration": 60}, user_id="u")
        context = context or PipelineContext(job_id=job_id, request={})
        return job_id, context, stage_resolve.run(job_id, {"prompt": prompt}, context)

    def answer(self, monkeypatch, raw):
        calls = []
        monkeypatch.setattr(intent_router, "_ask", lambda prompt, catalog, trace: calls.append(trace) or raw)
        return calls

    def test_the_route_reaches_stage_07_and_is_recorded(self, mock_job_store, processing_records_store, monkeypatch):
        calls = self.answer(monkeypatch, {"intent": "speaker", "speakers": ["cordel green"], "topic": "contributed"})
        job_id, context, result = self.run(monkeypatch)
        assert result.status == "completed"
        assert (context.route["method"], context.route["intent"], context.route["speakers"]) == (
            "router", "speaker", ["Cordel Green"])
        record = processing_records_store.resolutions[job_id]
        assert record["resolution_method"] == "router"
        assert record["slots"]["route"] == {"intent": "speaker", "speakers": ["Cordel Green"], "language": "en"}
        assert "router" in record["timings_ms"]
        assert (calls[0]["agent"], calls[0]["job_id"], calls[0]["user_id"]) == ("intent_router", job_id, "u")

    def test_a_rerun_reuses_the_first_route(self, mock_job_store, processing_records_store, monkeypatch):
        self.answer(monkeypatch, {"intent": "speaker", "speakers": ["cordel green"]})
        job_id, _, _ = self.run(monkeypatch)
        calls = self.answer(monkeypatch, {"intent": "topic", "topic": "contributed"})
        _, context, _ = self.run(monkeypatch, job_id=job_id)
        assert calls == [] and context.route["intent"] == "speaker"

    def test_a_cache_hit_asks_no_router(self, mock_job_store, processing_records_store, monkeypatch):
        from app.pipeline.base import PipelineContext

        calls = self.answer(monkeypatch, {"intent": "speaker", "speakers": ["cordel green"]})
        context = PipelineContext(job_id="x", request={}, storyline_cached=True)
        job_id, context, result = self.run(monkeypatch, context=context)
        assert calls == [] and context.route == {"method": "cached"}
        assert processing_records_store.resolutions[job_id]["resolution_method"] == "cached"

    def test_a_router_decline_fails_the_stage(self, mock_job_store, processing_records_store, monkeypatch):
        prompt = "What did Bartholomew Quenneville say about AI in music?"
        self.answer(monkeypatch, {"intent": "speaker", "speakers": ["Bartholomew Quenneville"], "topic": "music"})
        job_id, context, result = self.run(monkeypatch, prompt=prompt)
        assert (result.status, result.data["reason_code"]) == ("failed", "speaker_not_found")
        assert processing_records_store.resolutions[job_id]["resolution_method"] == "router"
        assert context.route is None

    def test_a_rules_decline_asks_no_router(self, mock_job_store, processing_records_store, monkeypatch):
        calls = self.answer(monkeypatch, {"intent": "topic"})
        _, _, result = self.run(monkeypatch, prompt="What did Green say about media?")
        assert result.data["reason_code"] == "speaker_too_many" and calls == []


class TestStorylineCache:
    def test_the_key_holds_the_router_version(self, monkeypatch):
        from app.services import storyline_cache

        before = storyline_cache._l1_key({"prompt": "p"})
        monkeypatch.setattr(settings, "intent_router_enabled", False)
        assert storyline_cache._l1_key({"prompt": "p"}) != before

    def test_a_fallback_storyline_is_not_cached(self, monkeypatch):
        from app.pipeline.base import PipelineContext
        from app.services import storyline_cache
        from app.services import video_job_runner as runner

        stored = []
        monkeypatch.setattr(storyline_cache, "store_storyline", lambda *a, **k: stored.append(a))
        data = {"editorial_quality_gates": {"final_timeline_approved": True}}
        for method in ("fallback", "router"):
            ctx = PipelineContext(job_id="j", request={}, validated_storyline_data=data, route={"method": method})
            runner._maybe_cache_storyline_result("j", {"prompt": "p"}, ctx)
        assert len(stored) == 1


def test_a_possessive_name_resolves():
    route = check("Cordel Green's views", intent="speaker", speakers=["Cordel Green's"])
    assert (route.decline, route.intent, route.speakers) == (None, "speaker", ["Cordel Green"])


def test_an_overview_with_a_subject_is_a_topic():
    route = check("What was said about deepfakes in 2024?", intent="overview", topic="deepfakes", year=2024)
    assert (route.intent, route.topic_terms, route.year) == ("topic", ["deepfake"], 2024)
    assert check("What are the major themes of AI4IA?", intent="overview", topic="major themes").intent == "overview"


def test_one_speaker_on_a_chronological_route_keeps_the_speaker():
    route = check("jessica rose smith's early work vs her latest thoughts on deepfakes", intent="chronological",
                  speakers=["jessica rose smith"], topic="deepfakes")
    assert (route.intent, route.speakers, route.topic_terms) == ("speaker", ["Jessica Rose Smith"], ["deepfake"])


class TestReviewFindings:
    def test_an_unnamed_chapter_outside_the_archive_is_dropped_not_declined(self):
        route = check("AI in agriculture", intent="theme_chapter", chapter=40, topic="agriculture")
        assert (route.decline, route.chapter, route.intent) == (None, None, "topic")

    def test_an_unnamed_facet_must_share_words_with_the_prompt(self):
        assert check("Tell me something interesting", intent="theme_chapter", chapter=13).intent == "topic"
        assert check("AI and health", intent="theme_chapter", theme="AI in Health, Science & Environment",
                     topic="health").intent == "theme_chapter"

    def test_an_english_prompt_keeps_the_topic_check_whatever_the_router_says(self):
        route = check("privacy law in Kenya", intent="topic", topic="nuclear weapons", language="fr")
        assert (route.language, route.topic_terms) == ("en", [])

    def test_an_odd_digit_is_not_a_year(self):
        assert check("deepfakes", intent="topic", topic="deepfakes", year="²").year is None

    def test_a_misspelled_excluded_speaker_is_excluded(self):
        route = check("regulation but not Cordell Grene", intent="topic", topic="regulation", exclude=["Cordell Grene"])
        assert (route.excluded_speakers, route.exclude_terms, route.decline) == (["Cordel Green"], [], None)

    def test_a_router_timeout_does_not_mark_the_host_down(self, monkeypatch):
        import requests

        from app.core import dependency_state
        from app.services import llm_service

        recorded = []
        monkeypatch.setattr(dependency_state, "record", lambda host, ok, **kw: recorded.append(ok))
        monkeypatch.setattr(llm_service.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
        with pytest.raises(RuntimeError):
            llm_service._post_with_retries("http://x/v1", {}, timeout=1, attempts=1, record_failure=False)
        assert recorded == []
