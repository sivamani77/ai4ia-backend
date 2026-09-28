"""L2a resolve: speakers, chapters and years checked against the archive."""
from __future__ import annotations

import pytest

from app.services.corpus_catalog import Catalog, build_catalog
from app.services.prompt_resolver import resolve

CATALOG = Catalog(frozenset(range(1, 26)), 2021, 2026)


@pytest.mark.parametrize("prompt,speaker", [
    ("What did Cordell Green say?", "Cordel Green"),                 # alias
    ("What are Cordel Green's views on regulation?", "Cordel Green"),
    ("according to Cordel Green, what is regulation?", "Cordel Green"),
    ("What did Natali Helberger say about media regulation?", "Natali Helberger"),  # panel only
    ("Dario Prishko on AI in Europe", "Dariia Opryshko"),           # alias of a panel moderator
    ("What did David Daou say?", "David Daou"),                      # review name: used, not suggested
    ("What did Helberger say about media?", "Natali Helberger"),     # unique surname
    ("What did Anna Paolini say?", "Anna Paolini"),                 # opening-session panels only
])
def test_resolves_speaker(prompt, speaker):
    result = resolve(prompt, CATALOG)
    assert result.decline is None
    assert result.speaker == speaker


@pytest.mark.parametrize("prompt,code,suggestions", [
    ("What did Cordle Grene say about media?", "speaker_did_you_mean", ["Cordel Green"]),
    ("What did Green say about media?", "speaker_too_many",
     ["Ben Green", "Cordel Green", "Kemar E. Green", "Pia-Milan Green"]),
    ("What did Andrew say about AI?", "speaker_ambiguous", ["Andrew Knight", "Andrew Maynard", "Andrew Smith"]),
    ("What did Singh say?", "speaker_ambiguous_partial", ["Ashita Singh"]),   # A Singh is in review
    ("What did John Smith say about AI?", "speaker_not_found", []),
    ("What did Ashok Jhunjhunwala say?", "speaker_no_recording", []),
    ("Chapter 30", "chapter_not_found", []),
    ("AI talks from 2019", "year_out_of_range", []),
])
def test_declines(prompt, code, suggestions):
    decline = resolve(prompt, CATALOG).decline
    assert decline.reason_code == code
    assert decline.suggestions == suggestions
    assert decline.message


@pytest.mark.parametrize("prompt", [
    "what did the speakers say about access to information",
    "What did Africa say about AI?",
    "What was said about deepfakes in 2024?",
    "predictions for AI in 2030",                                   # a year, but not about the archive
    "According to the Chinese regulation, what are the rules?",
    "What did Nigeria say about AI regulation?",                    # places and organisations
    "What did the UN say about digital rights?",
    "What did NATO propose?",
    "What did the European Union say about the AI Act?",
    "What did Microsoft Research say about fairness?",
    "Let's take for example the region's Creole languages",
    "AI governance and transparency",
    "Chapter 5",
])
def test_passes(prompt):
    assert resolve(prompt, CATALOG).decline is None


def test_chapter_and_year_checks_are_skipped_without_a_catalog():
    assert resolve("Chapter 30", None).decline is None
    assert resolve("AI talks from 2019", None).decline is None


def test_build_catalog():
    records = [{"primary_chapter": "Ch 3: X", "video_year": 2021, "primary_theme": "T",
                "source_video": "0001_abc_Ann Lee - Open Data"},
               {"primary_chapter": "Ch 12: Y", "video_year": "2026"},
               {"primary_chapter": None, "video_year": ""}]
    assert build_catalog(records) == Catalog(
        frozenset({3, 12}), 2021, 2026, frozenset({"open data"}), ("T",), {3: "Ch 3: X", 12: "Ch 12: Y"},
        {"open data": ("0001_abc_Ann Lee - Open Data",)})
    assert build_catalog([]) is None


def test_prompt_intent_uses_the_resolved_speaker():
    from app.pipeline.storyline_gen.prompt_intent import detect_prompt_intent

    profile = detect_prompt_intent("What did Helberger say about media?", [], [],
                                   requested_speaker="Natali Helberger")
    assert profile["requested_speaker"] == "Natali Helberger"
    assert "helberger" not in profile["topic_terms"]


class TestDeclinePath:
    def test_a_declined_job_carries_the_message_and_suggestions(self, mock_job_store, processing_records_store,
                                                                monkeypatch):
        from app.pipeline import stage_resolve
        from app.pipeline.base import PipelineContext
        from app.services import job_store
        from app.services import video_job_runner as runner

        monkeypatch.setattr(stage_resolve, "append_event", job_store.append_event)
        monkeypatch.setattr(stage_resolve, "get_catalog", lambda: CATALOG)
        prompt = "What did Cordle Grene say about media?"
        job_id = runner.create_video_job({"prompt": prompt, "target_duration": 60}, user_id="u")

        result = stage_resolve.run(job_id, {"prompt": prompt}, PipelineContext(job_id=job_id, request={}))
        data = result.data
        runner._fail_job(job_id, result.stage, data["error_code"], result.message, user_message=data["user_message"],
                         reason_code=data["reason_code"], suggestions=data["suggestions"])

        error = mock_job_store["jobs"][job_id]["error"]
        assert error["type"] == "declined"
        assert error["reason_code"] == "speaker_did_you_mean"
        assert error["suggestions"] == ["Cordel Green"]
        assert "Did you mean Cordel Green?" in error["user_message"]
        record = processing_records_store.resolutions[job_id]
        assert (record["layer_outcome"], record["reason_code"]) == ("declined", "speaker_did_you_mean")
        assert record["dropped_slots"] == [{"slot": "speaker", "value": "Cordle Grene",
                                            "reason": "speaker_did_you_mean"}]

    def test_a_resolved_speaker_reaches_the_context(self, mock_job_store, processing_records_store, monkeypatch):
        from app.pipeline import stage_resolve
        from app.pipeline.base import PipelineContext
        from app.services import job_store
        from app.services import video_job_runner as runner

        monkeypatch.setattr(stage_resolve, "append_event", job_store.append_event)
        monkeypatch.setattr(stage_resolve, "get_catalog", lambda: None)
        job_id = runner.create_video_job({"prompt": "p", "target_duration": 60}, user_id="u")
        context = PipelineContext(job_id=job_id, request={})

        result = stage_resolve.run(job_id, {"prompt": "What did Helberger say about media?"}, context)

        assert result.status == "completed"
        assert context.resolved_speaker == "Natali Helberger"
        assert processing_records_store.resolutions[job_id]["slots"] == {"speaker": "Natali Helberger"}
