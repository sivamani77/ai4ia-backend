"""Phase 6: L2b coverage — rule, evidence judge, outcomes (docs/intent-layer-design.md §8)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.storyline_gen import coverage, storyline_llm
from app.pipeline.validate_gen import narrative_assembly

FLOOR, LOW = 0.68, 0.60


def frag(fid, score, text="", theme="AI Ethics, Governance & Policy", **extra):
    return {"fragment_id": fid, "semantic_score": score, "text": text, "primary_theme": theme, **extra}


def profile(prompt, terms, speaker=None):
    return {"raw_prompt": prompt, "topic_terms": terms, "requested_speaker": speaker}


def no_llm(*args, **kwargs):
    raise AssertionError("the judge must not be called")


def llm_returning(payload):
    calls = []

    def call(user, **kwargs):
        calls.append((user, kwargs))
        return payload if isinstance(payload, str) else json.dumps(payload)

    call.calls = calls
    return call


def assess(intent, prof, pool, call=no_llm):
    return coverage.assess(intent, prof, pool, prof["raw_prompt"], call, floor=FLOOR, low=LOW)


class TestRule:
    @pytest.mark.parametrize("top1,missing,expected", [
        (0.79, [], "covered"),
        (0.79, ["quantum"], "judge"),    # "quantum computing in agriculture" scored 0.77
        (0.65, [], "judge"),             # the band between low and floor
        (0.59, [], "not_covered"),
        (None, [], "judge"),
    ])
    def test_verdicts(self, top1, missing, expected):
        assert coverage.rule_verdict(top1, missing, FLOOR, LOW) == expected

    def test_missing_terms_checks_the_best_matches_only(self):
        best = [frag(f"a{i}", 0.8, "AI in agriculture and cloud computing") for i in range(coverage.TERM_WINDOW)]
        passing_mention = frag("z", 0.6, "a word on quantum")
        assert coverage.missing_terms(best + [passing_mention], ["agriculture", "computing", "quantum"]) == ["quantum"]

    @pytest.mark.parametrize("prompt,terms,phrases", [
        ("deepfakes and space law", ["deepfake", "law", "space"], ["space law"]),
        ("deepfakes and elections", ["deepfake", "election"], []),
        ("AI governance and transparency", ["governance", "transparency"], []),
    ])
    def test_prompt_phrases(self, prompt, terms, phrases):
        assert coverage.prompt_phrases(prompt, terms) == phrases

    def test_adjacent_terms_must_match_as_a_phrase(self):
        # Matrix row 24: "civic space" and "law" in separate fragments made "space law" look covered.
        pool = [frag("a", 0.8, "deepfakes erode civic space"), frag("b", 0.8, "the law on deepfakes")]
        terms = ["deepfake", "law", "space"]
        assert coverage.missing_terms(pool, terms, ["space law"]) == ["space law"]
        assert coverage.missing_terms([frag("c", 0.8, "deepfakes and space laws")], terms, ["space law"]) == []


class TestAssess:
    def test_covered_by_rule_without_the_judge(self):
        pool = [frag("a", 0.79, "AI governance and transparency rules")]
        result = assess("topic", profile("AI governance and transparency", ["governance", "transparency"]), pool)
        assert (result.verdict, result.method, result.top_similarity) == ("covered", "rule", 0.79)

    def test_off_topic_is_declined_with_nearby_themes(self):
        pool = [frag("a", 0.55, "cake", theme="AI in Health, Science & Environment")]
        result = assess("topic", profile("Best recipes for chocolate cake", ["recipe", "chocolate", "cake"]), pool)
        assert (result.verdict, result.method, result.reason_code) == ("not_covered", "rule", "topic_not_covered")
        assert result.user_message == ("The AI4IA archive doesn't have talks on recipes chocolate cake. "
                                       "Related themes it does cover: AI in Health, Science & Environment.")
        assert result.suggestions == ["AI in Health, Science & Environment"]

    def test_judge_decides_a_missing_term(self):
        pool = [frag("a", 0.77, "AI for smallholder agriculture"), frag("b", 0.7, "computing power")]
        call = llm_returning({"subjects": [
            {"subject": "agriculture", "discussed": True, "fragment_ids": ["a", "invented"]},
            {"subject": "quantum computing", "discussed": False}]})
        result = assess("topic", profile("Quantum computing in agriculture", ["agriculture", "computing", "quantum"]),
                        pool, call)
        assert (result.verdict, result.method) == ("partial", "judge")
        assert result.missing_terms == ["quantum", "quantum computing"]
        assert (result.addressed_aspects, result.missing_aspects) == (["agriculture"], ["quantum computing"])
        assert result.supporting_ids == ["a"]  # ids it was never shown are dropped
        user, kwargs = call.calls[0]
        assert user.startswith("REQUEST: <untrusted-data>\nQuantum computing in agriculture\n</untrusted-data>")
        assert kwargs["system"] == coverage.JUDGE_SYSTEM and kwargs["reasoning_effort"] == "medium"

    def test_judge_not_covered_declines_with_its_aspects(self):
        pool = [frag("a", 0.66, "trading data")]
        call = llm_returning({"subjects": [{"subject": "cryptocurrency trading", "discussed": False}]})
        result = assess("topic", profile("Cryptocurrency trading strategies", ["cryptocurrency", "trading"]),
                        pool, call)
        assert result.verdict == "not_covered"
        assert result.user_message.startswith("The AI4IA archive doesn't have talks on cryptocurrency trading.")

    @pytest.mark.parametrize("reply", ["not json", '{"coverage": "covered"}', '{"subjects": []}', RuntimeError("down")])
    def test_judge_failure_never_blocks(self, reply):
        def call(*args, **kwargs):
            if isinstance(reply, Exception):
                raise reply
            return reply

        result = assess("topic", profile("Deepfakes and space law", ["deepfake", "space"]),
                        [frag("a", 0.69, "deepfakes")], call)
        assert (result.verdict, result.method) == ("covered", "judge_failed")

    @pytest.mark.parametrize("discussed,expected", [
        ((True, True), "covered"), ((True, False), "partial"), ((False, False), "not_covered"),
    ])
    def test_code_not_the_model_aggregates_subjects(self, discussed, expected):
        call = llm_returning({"subjects": [{"subject": f"s{i}", "discussed": d} for i, d in enumerate(discussed)]})
        result = assess("topic", profile("x y", ["x"]), [frag("a", 0.65, "x")], call)
        assert result.verdict == expected

    @pytest.mark.parametrize("intent", ["speaker", "title", "overview", "theme_chapter"])
    def test_intents_confirmed_by_resolve_are_not_checked(self, intent):
        result = assess(intent, profile("p", ["x"]), [frag("a", 0.3)])
        assert (result.verdict, result.method) == ("covered", "not_checked")

    def test_no_evidence_pool_never_declines(self):
        result = assess("speaker_topic", profile("What did Cordel Green say about x?", ["x"], "Cordel Green"), [])
        assert (result.verdict, result.method) == ("covered", "no_evidence_pool")


class TestSpeakerTopic:
    PROFILE = profile("What did Cordel Green say about quantum computing?", ["quantum", "computing"], "Cordel Green")

    def test_pool_is_the_speakers_own_fragments(self):
        pool = [frag("own", 0.70, "media", speaker_match=True),
                frag("by_name", 0.69, "media", speaker="Cordel Green"),     # a title match carries no flag
                frag("panel", 0.8, "x", speaker_match=True, speaker_match_panel=True),
                frag("other", 0.9, "quantum computing", speaker="Someone Else")]
        picked = coverage.evidence_pool("speaker_topic", self.PROFILE, pool)
        assert [c["fragment_id"] for c in picked] == ["own", "by_name"]

    def test_decline_names_the_speakers_themes(self):
        pool = [frag("a", 0.58, "media regulation", speaker_match=True,
                     theme="AI & Information Access / Media / Journalism")]
        result = assess("speaker_topic", self.PROFILE, pool)
        assert result.reason_code == "speaker_topic_not_covered"
        assert result.user_message == ("Cordel Green spoke at AI4IA, but not about quantum computing. "
                                       "Cordel Green's talks cover AI & Information Access / Media / Journalism.")
        assert result.suggestions == ["What did Cordel Green say?"]


def test_judge_sees_the_best_fragment_for_every_term():
    law = [frag(f"law{i}", 0.69 - i / 1000, "space law and liability") for i in range(12)]
    deepfake = frag("df", 0.60, "deepfakes spread during elections")
    picked = coverage._judge_fragments(law + [deepfake], ["deepfake", "space", "law"])
    assert len(picked) == coverage.JUDGE_FRAGMENTS
    assert (deepfake, ["deepfake"]) in picked


def test_term_evidence_comes_from_the_whole_archive():
    # Found live: for "deepfakes and space law" the retrieved pool held no deepfake talk;
    # its only "deepfake" hit was a title, and the judge declined the request.
    pool = [frag("law", 0.69, "space law treaties"), frag("title_only", 0.68, "stories about AI",
                                                              source_video="Ethics, AI and deepfakes")]
    archive = pool + [frag("smith", 0.5, "Deepfakes harm women. Harmful deepfakes spread. Deepfake laws lag.")]
    picked = dict((c["fragment_id"], focus) for c, focus in coverage._judge_fragments(pool, ["deepfake", "law"], archive))
    assert picked["smith"] == ["deepfake"]


@pytest.mark.parametrize("word,term,expected", [
    ("Deepfakes are here", "deepfake", True), ("indigenous languages", "indigenou", True),
    ("elections matter", "election", True), ("a romance novel", "roman", False),
    ("the Roman empire", "roman", True),
])
def test_term_endings(word, term, expected):
    assert coverage.has_term({"text": word}, term) is expected


def test_snippet_centres_on_the_topic_term():
    text = "intro " * 200 + "space law governs orbital debris."
    assert "space law" in coverage.snippet({"text": text}, ["space"])
    assert coverage.snippet({"text": "short text"}, ["absent"]) == "short text"


def test_topic_phrase_uses_topic_words_never_the_raw_prompt():
    prof = profile("Hi, my name is Ana and I want Quantum computing please", ["quantum", "computing"])
    assert coverage.topic_phrase(prof, []) == "Quantum computing"
    assert coverage.topic_phrase(prof, ["quantum computing"]) == "quantum computing"


class TestGapNarration:
    def test_storyline_payload_carries_the_gap_only_when_partial(self, monkeypatch):
        sent = []
        monkeypatch.setattr(storyline_llm, "call_strong_llm",
                            lambda user, **kw: sent.append(user) or '{"title": "t", "export_timeline": []}')
        args = ("Deepfakes and space law", 60, [], "doc", {}, {"raw_prompt": "x"}, {})
        storyline_llm.generate_llm_storyline(*args, intent_context={"coverage_gaps": ["space law"]})
        storyline_llm.generate_llm_storyline(*args, intent_context={})
        assert '"not_in_archive"' in sent[0] and "space law" in sent[0]
        assert "not_in_archive" not in sent[1]

    def test_assembly_prompt_states_the_gap(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(narrative_assembly, "post_with_retries",
                            lambda payload, timeout=300: seen.setdefault("u", payload["messages"][-1]["content"]) and "{}")
        narrative_assembly.assemble_narrative_with_llm(
            [{"fragment_id": "f", "decision": "speaker_clip", "approved": True}],
            [{"section_id": "S1", "title": "t", "purpose": "p", "card_text": "c"}], "p", 60,
            {"storyline": {"coverage_gaps": ["space law"]}})
        assert "NOT IN THE ARCHIVE" in seen["u"] and "space law" in seen["u"]


class TestStage07Decline:
    def run_stage(self, tmp_path, verdict, decline_enabled=True):
        from app.pipeline import stage_07_storyline
        from app.pipeline.base import PipelineContext

        def fake_script(job_id, stage, cmd, extra_env=None):
            path = cmd[cmd.index("--coverage-file") + 1]
            with open(path, "w") as fh:
                json.dump(verdict, fh)
            return 0, "", ""

        with patch.object(stage_07_storyline, "run_script", fake_script), \
             patch.object(stage_07_storyline, "download_if_missing", return_value={}), \
             patch.object(stage_07_storyline, "use_s3", return_value=False), \
             patch.object(stage_07_storyline, "update_stage", MagicMock()), \
             patch.object(stage_07_storyline, "append_event", MagicMock()), \
             patch.object(stage_07_storyline.settings, "coverage_decline_enabled", decline_enabled):
            return stage_07_storyline.run("vjob_cov", {"prompt": "p", "target_duration": 60}, PipelineContext(
                job_id="vjob_cov", request={}))

    NOT_COVERED = {"verdict": "not_covered", "method": "rule", "reason_code": "topic_not_covered",
                   "user_message": "The AI4IA archive doesn't have talks on cake.", "suggestions": ["AI in Health"]}

    def test_not_covered_declines_with_the_message(self, tmp_path, processing_records_store):
        processing_records_store.resolutions["vjob_cov"] = {"job_id": "vjob_cov", "layer_outcome": "in_progress"}
        result = self.run_stage(tmp_path, self.NOT_COVERED)
        assert result.status == "failed"
        assert result.data == {"error_code": "declined", "reason_code": "topic_not_covered",
                               "user_message": "The AI4IA archive doesn't have talks on cake.",
                               "suggestions": ["AI in Health"]}
        row = processing_records_store.resolutions["vjob_cov"]
        assert row["layer_outcome"] == "declined" and row["coverage"]["verdict"] == "not_covered"

    def test_a_declined_job_records_its_intent(self, tmp_path, processing_records_store):
        processing_records_store.resolutions["vjob_cov"] = {"job_id": "vjob_cov", "layer_outcome": "in_progress"}
        self.run_stage(tmp_path, {**self.NOT_COVERED, "intent": "topic", "intent_version": "3"})
        row = processing_records_store.resolutions["vjob_cov"]
        assert (row["intent"], row["intent_version"]) == ("topic", "3")
        assert "intent" not in row["coverage"]

    def test_shadow_mode_records_but_renders(self, tmp_path, processing_records_store):
        processing_records_store.resolutions["vjob_cov"] = {"job_id": "vjob_cov", "layer_outcome": "in_progress"}
        # Past the decline, the stage looks for the storyline the (faked) script never wrote.
        with pytest.raises(RuntimeError, match="No storyline JSON"):
            self.run_stage(tmp_path, self.NOT_COVERED, decline_enabled=False)
        row = processing_records_store.resolutions["vjob_cov"]
        assert row["layer_outcome"] == "in_progress" and row["coverage"]["verdict"] == "not_covered"


def test_completed_job_records_its_coverage_verdict(processing_records_store):
    from app.services import processing_records

    for job, verdict, expected in (("j1", "partial", "partial"), ("j2", "covered", "covered"), ("j3", None, "completed")):
        processing_records_store.resolutions[job] = {"job_id": job, "layer_outcome": "in_progress",
                                                     **({"coverage": {"verdict": verdict}} if verdict else {})}
        processing_records.finish(job, "completed")
        assert processing_records_store.resolutions[job]["layer_outcome"] == expected


class TestStateCoverageGaps:
    def timeline(self, *texts):
        return [{"type": "title_card"}] + [{"type": "narration_broll", "voiceover_text": t} for t in texts]

    def test_opens_with_the_gap_and_drops_invented_sentences(self):
        # From the phase 6 render: stage 07 invented "space-law frameworks" for a subject the archive lacks.
        from app.pipeline.validate_gen.archive_answer import state_coverage_gaps

        tl = self.timeline("Deepfakes blur truth. Experts propose extending space-law frameworks.",
                           "Space law treaties lack AI clauses! Detection tools are improving.")
        assert state_coverage_gaps(tl, ["space law"], ["deepfakes"]) == 2
        assert tl[1]["voiceover_text"] == ("The AI4IA archive has no talks on space law. This film looks at what "
                                           "its speakers said about deepfakes. Deepfakes blur truth.")
        assert tl[2]["voiceover_text"] == tl[2]["subtitle_text"] == "Detection tools are improving."

    def test_drops_sentences_built_on_a_distinctive_gap_word(self):
        # From the re-render: "AI-generated space imagery" invented next to the gap "space law".
        from app.pipeline.validate_gen.archive_answer import state_coverage_gaps

        tl = self.timeline("Scholars propose copyright for AI-generated space imagery. Deepfake law lags behind.")
        assert state_coverage_gaps(tl, ["space law"], ["deepfakes"]) == 1
        assert tl[1]["voiceover_text"].endswith("said about deepfakes. Deepfake law lags behind.")

    def test_nothing_to_state(self):
        from app.pipeline.validate_gen.archive_answer import state_coverage_gaps

        tl = self.timeline("Deepfakes blur truth.")
        assert state_coverage_gaps(tl, [], ["deepfakes"]) == 0
        assert tl[1]["voiceover_text"] == "Deepfakes blur truth."
