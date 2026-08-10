"""Golden-set regression tests for the storyline prompt-scoring pipeline.

Exercises detect_prompt_intent -> annotate_candidate_relevance -> bucket_candidates
-> compute_evidence_metrics -> evidence_strength_from_metrics on synthetic
candidates (no embeddings/ChromaDB/Ollama needed), so future weight/threshold
changes in prompt_intent.py, candidate_scoring.py, and director.py can be
checked against known-good/known-bad outcomes instead of eyeballed.

Confirmed incident this guards against: the real prompt "Ai for the
Information technology" retrieved 90 candidates (top semantic_score=0.720247)
that all landed in the `context` bucket and were discarded, because every
word in the prompt is in STOPWORDS (prompt_intent.py:14-32), so topic_terms
came back empty and every keyword-gated classification branch in
classify_candidate_relevance_v2 (candidate_scoring.py:444-461) was
unreachable regardless of how high semantic_score was.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.storyline_gen.candidate_scoring import annotate_candidate_relevance  # noqa: E402
from app.pipeline.storyline_gen.director import (  # noqa: E402
    bucket_candidates,
    compute_evidence_metrics,
    evidence_strength_from_metrics,
)
from app.pipeline.storyline_gen.prompt_intent import detect_prompt_intent  # noqa: E402
from app.pipeline.storyline_gen.selection import build_storyline_evidence_package  # noqa: E402


def make_candidate(
    fragment_id: str,
    semantic_score: float,
    text: str = "",
    *,
    summary: str = "",
    keywords: list | None = None,
    primary_theme: str = "",
    primary_chapter: str = "",
    speaker: str = "",
    source_video: str | None = None,
    quote_strength: float = 0.5,
    clarity_score: float = 0.5,
) -> dict:
    return {
        "fragment_id": fragment_id,
        "text": text,
        "summary": summary,
        "keywords": keywords or [],
        "semantic_score": semantic_score,
        "primary_theme": primary_theme,
        "primary_chapter": primary_chapter,
        "speaker": speaker,
        "source_video": source_video or f"video_{fragment_id}",
        "quote_strength": quote_strength,
        "clarity_score": clarity_score,
    }


def run_pipeline(prompt: str, candidates: list[dict]) -> tuple[dict, dict, dict]:
    """Run the real intent -> scoring -> bucketing chain. Returns (buckets, evidence_metrics, intent)."""
    intent = detect_prompt_intent(prompt, known_speakers=[], records=[])
    classified = annotate_candidate_relevance(candidates, prompt, intent, known_speakers=[])
    buckets = bucket_candidates(classified, intent)
    metrics = compute_evidence_metrics(
        buckets["selected_candidates"],
        buckets["supporting_candidates"],
        buckets["context_candidates"],
        buckets["rejected_candidates"],
    )
    return buckets, metrics, intent


class TestOriginalIncidentFixedByStopwordTrim:
    """The actual confirmed incident: 'Ai for the Information technology'
    retrieved 90 candidates (top semantic_score=0.720247) that all landed in
    `context` and were discarded, because "information"/"technology" were in
    STOPWORDS -- the archive's own subject-matter vocabulary -- so
    topic_terms came back empty. scripts/diagnostics/stopword_frequency_report.py
    confirmed both words are well below the 55% corpus document-frequency
    threshold (50.5% and 21.1%), unlike "ai" itself (100%, correctly still
    stopped), so they were removed from STOPWORDS/SPEAKER_PROMPT_STOPWORDS.
    This is now fixed at the root via the normal keyword-matching path --
    no semantic-only fallback needed for this specific prompt any more."""

    PROMPT = "Ai for the Information technology"

    def test_topic_terms_no_longer_empty(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        assert intent["topic_terms"] == ["information", "technology"]

    def test_on_topic_candidate_reaches_exact_or_strong_related_via_keywords(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_it",
            semantic_score=0.72,
            text="Deploying artificial intelligence across information technology infrastructure",
            quote_strength=0.5,
            clarity_score=0.8,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] in ("exact", "strong_related")
        # Reached via real keyword overlap, not the semantic-only fallback.
        assert classified[0]["semantic_only_match"] is False


class TestEmptyTopicTermsFallback:
    """Not every domain-generic prompt gets rescued by the stopword trim --
    "ai" itself stays stopped (100% corpus document frequency, confirmed by
    the diagnostic script -- it can't discriminate between candidates), and
    purely generic/meta prompts can still produce empty topic_terms. For
    those, direct_terms/expanded_hits/theme_match/chapter_match are all
    structurally unreachable in classify_candidate_relevance_v2, so only a
    dedicated semantic-only branch can rescue genuinely strong candidates."""

    PROMPT = "Tell me about the conference themes"

    def test_topic_terms_empty_for_failing_prompt(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        assert intent["topic_terms"] == []

    def test_high_semantic_candidates_promoted_to_strong_related(self):
        # Realistic shape of the real incident (90 retrieved candidates, all
        # landed in `context`, top semantic_score=0.720247) rather than a
        # single candidate -- one candidate alone can never satisfy the
        # strong_related_count>=4 gate in evidence_strength_from_metrics.
        scores = [0.72, 0.70, 0.68, 0.66, 0.65, 0.63, 0.61]
        candidates = [
            make_candidate(f"frag_{i}", score, text="reflections on the overall themes discussed at the conference")
            for i, score in enumerate(scores)
        ]
        buckets, metrics, _ = run_pipeline(self.PROMPT, candidates)

        selected = buckets["selected_candidates"]
        selected_scores = sorted((c["semantic_score"] for c in selected), reverse=True)
        assert selected_scores == [0.72, 0.70, 0.68, 0.66, 0.65]
        assert all(c["relevance_role"] == "strong_related" for c in selected)
        assert all(c.get("semantic_only_match") for c in selected)
        assert metrics["strong_related_count"] == 5
        assert evidence_strength_from_metrics(metrics) == "medium"

    def test_promoted_candidates_remain_eligible_for_full_soundbite(self):
        # Semantic-only promotion must not force muted/narration-only use --
        # a clear, strong speaker clip should still be usable as a real,
        # voiced soundbite. Only quote/clarity gate this, same as any other
        # strong_related candidate.
        candidate = make_candidate(
            "frag_strong",
            semantic_score=0.72,
            text="reflections on the overall themes discussed at the conference",
            quote_strength=0.5,
            clarity_score=0.8,
        )
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] == "strong_related"
        assert classified[0]["speaker_soundbite_allowed"] is True

    def test_must_not_claim_flags_semantic_only_evidence(self):
        scores = [0.72, 0.70, 0.68, 0.66, 0.65]
        candidates = [
            make_candidate(f"frag_{i}", s, text="general reflections on the conference themes")
            for i, s in enumerate(scores)
        ]
        buckets, _metrics, intent = run_pipeline(self.PROMPT, candidates)
        package = build_storyline_evidence_package(intent, buckets)
        assert any("not an exact match" in note.lower() for note in package["must_not_claim"])


class TestContextBucketFallback:
    """Isolates the Step-3 fix: when every candidate is below
    NO_TOPIC_TERMS_SEMANTIC_FLOOR (so the Step-2 branch never fires) but still
    clears the plain contextual floor (semantic_score>=0.60), bucket_candidates
    must not discard all of them just because none has keyword evidence."""

    PROMPT = "Please show the best clips about this theme"

    def test_topic_terms_empty(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        assert intent["topic_terms"] == []

    def test_context_candidates_promoted_when_nothing_clears_semantic_floor(self):
        scores = [0.64, 0.63, 0.62, 0.61, 0.60, 0.60]
        candidates = [
            make_candidate(f"frag_{i}", s, text="general AI conference discussion")
            for i, s in enumerate(scores)
        ]
        buckets, metrics, _ = run_pipeline(self.PROMPT, candidates)

        selected = buckets["selected_candidates"]
        assert len(selected) == 5  # max_low_evidence_weak default cap
        assert all(c.get("promoted_from_context") for c in selected)
        assert all(c.get("use_as") == "narration_support" for c in selected)
        assert all(c.get("speaker_soundbite_allowed") is False for c in selected)
        # the 6th candidate should be left behind in context, not silently dropped
        assert len(buckets["context_candidates"]) == 1
        assert metrics["context_promoted_count"] == 5


class TestNormalKeywordPathUnaffected:
    """Regression guards: the ordinary keyword-driven path must behave
    identically before and after the empty-topic_terms fallback lands."""

    PROMPT = "AI in healthcare"

    def test_topic_terms_non_empty(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        assert intent["topic_terms"] == ["healthcare"]

    def test_on_topic_candidate_reaches_exact_or_strong_related(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_health",
            semantic_score=0.55,
            text="AI in healthcare and medicine is transforming patient care",
            primary_theme="AI in Health, Science & Environment",
            quote_strength=0.6,
            clarity_score=0.6,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] in ("exact", "strong_related")

    def test_off_topic_candidate_stays_rejected(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_banking",
            semantic_score=0.55,
            text="welcome remarks, thank you for inviting me to discuss banking and marketing",
            primary_theme="Banking & Finance",
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] == "reject"


class TestUnstoppedTermsRegressionGuard:
    """Regression guards for the specific words the Step-4 diagnostic
    unstopped (artificial/intelligence/information/technology/data/content/
    system/tool, all confirmed <55% corpus document frequency): on-topic
    prompts built from them must now reach real evidence, and unrelated
    candidates must still be rejected -- the trim must not cause these words
    to over-match generically."""

    PROMPT = "data privacy in artificial intelligence systems"

    def test_topic_terms_include_unstopped_words(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        assert {"data", "privacy", "artificial", "system"} <= set(intent["topic_terms"])

    def test_on_topic_candidate_reaches_exact_or_strong_related(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_privacy",
            semantic_score=0.66,
            text="A talk on data privacy safeguards in artificial intelligence system design",
            quote_strength=0.7,
            clarity_score=0.7,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] in ("exact", "strong_related")

    def test_off_topic_candidate_with_no_overlap_stays_rejected(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_unrelated",
            semantic_score=0.5,
            text="welcome remarks and thank you for inviting me to this event",
            primary_theme="Banking & Finance",
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["relevance_role"] == "reject"


class TestThemeAndChapterMatch:
    """Exercises the theme_match/chapter_match branches in
    classify_candidate_relevance_v2 via PROMPT_TOPIC_CONFIG's healthcare
    theme_hint/chapter_hint (prompt_intent.py:81-92)."""

    PROMPT = "AI in healthcare"

    def test_theme_match_alone_reaches_strong_related(self):
        # Theme matches (via the theme_hint's own wording pulling in an
        # expanded synonym) but the candidate's own text has no direct
        # keyword overlap -- theme_match without direct evidence can reach
        # strong_related but not exact by design.
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_theme",
            semantic_score=0.55,
            text="a personal story about long-term recovery",
            primary_theme="AI in Health, Science & Environment",
            quote_strength=0.6,
            clarity_score=0.6,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["theme_match"] is True
        assert classified[0]["direct_evidence_terms"] == []
        assert classified[0]["relevance_role"] == "strong_related"

    def test_chapter_match_plus_direct_term_reaches_exact(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=[], records=[])
        candidate = make_candidate(
            "frag_chapter_direct",
            semantic_score=0.55,
            text="AI in healthcare is reshaping diagnosis and treatment",
            primary_chapter="Ch 13: AI in Healthcare & Medicine",
            quote_strength=0.6,
            clarity_score=0.6,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=[])
        assert classified[0]["chapter_match"] is True
        assert classified[0]["direct_evidence_terms"] == ["healthcare"]
        assert classified[0]["relevance_role"] == "exact"


class TestSpeakerRequestedPrompts:
    """Exercises the requested_speaker + scope=="all_themes" branch
    (candidate_scoring.py) via a 'show all themes from <speaker>' prompt."""

    PROMPT = "Show all themes from Jane Doe"
    KNOWN_SPEAKERS = ["Jane Doe"]

    def test_speaker_and_all_themes_scope_detected(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=self.KNOWN_SPEAKERS, records=[])
        assert intent["requested_speaker"] == "Jane Doe"
        assert intent["scope"] == "all_themes"

    def test_matching_speaker_strong_evidence_reaches_exact(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=self.KNOWN_SPEAKERS, records=[])
        candidate = make_candidate(
            "frag_jane_strong",
            semantic_score=0.5,
            text="Jane Doe reflects on a decade of AI4IA conferences",
            speaker="Jane Doe",
            quote_strength=0.6,
            clarity_score=0.8,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=self.KNOWN_SPEAKERS)
        assert classified[0]["speaker_match"] is True
        assert classified[0]["relevance_role"] == "exact"

    def test_matching_speaker_limited_evidence_reaches_strong_related(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=self.KNOWN_SPEAKERS, records=[])
        candidate = make_candidate(
            "frag_jane_limited",
            semantic_score=0.35,
            text="a brief aside from Jane Doe",
            speaker="Jane Doe",
            quote_strength=0.3,
            clarity_score=0.4,
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=self.KNOWN_SPEAKERS)
        assert classified[0]["speaker_match"] is True
        assert classified[0]["relevance_role"] == "strong_related"

    def test_non_matching_speaker_is_rejected(self):
        intent = detect_prompt_intent(self.PROMPT, known_speakers=self.KNOWN_SPEAKERS, records=[])
        candidate = make_candidate(
            "frag_someone_else",
            semantic_score=0.9,
            text="a completely unrelated speaker talking about something else",
            speaker="John Smith",
        )
        classified = annotate_candidate_relevance([candidate], self.PROMPT, intent, known_speakers=self.KNOWN_SPEAKERS)
        assert classified[0]["speaker_match"] is False
        assert classified[0]["relevance_role"] == "reject"


class TestMoreEmptyTopicTermsPrompts:
    """A couple more purely generic/meta prompts beyond the original
    incident, to confirm the semantic-only fallback isn't a one-off special
    case tied to one specific wording."""

    @pytest.mark.parametrize("prompt", [
        "Please give a good explainer video",
        "Show the top related evidence clips",
    ])
    def test_topic_terms_empty(self, prompt):
        intent = detect_prompt_intent(prompt, known_speakers=[], records=[])
        assert intent["topic_terms"] == []

    @pytest.mark.parametrize("prompt", [
        "Please give a good explainer video",
        "Show the top related evidence clips",
    ])
    def test_strong_semantic_candidate_still_promoted(self, prompt):
        candidates = [
            make_candidate(f"frag_{i}", score, text="a general reflection on the field")
            for i, score in enumerate([0.72, 0.70, 0.68, 0.67])
        ]
        buckets, metrics, _ = run_pipeline(prompt, candidates)
        assert len(buckets["selected_candidates"]) > 0
        assert all(c["relevance_role"] == "strong_related" for c in buckets["selected_candidates"])
