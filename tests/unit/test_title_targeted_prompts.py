"""Regression tests for prompts that name a source video by its title.

Confirmed incident this guards against: a prompt that was the exact title of
an archive video produced a final video containing none of that video's
footage. Three defects compounded, and each has a class below:

1. score_all_records only attaches semantic_score to ChromaDB's top-N ANN
   hits. Candidates that reach the merged set through the keyword/theme/
   speaker channels arrived with no semantic_score at all, which reads as 0.0
   downstream. Measured on the live corpus: all 33 retrieved fragments of
   "Latin America Event - Between Algorithms and Citizenship - Youth Facing
   AI" came back via the keyword channel with semantic_score=None.

2. semantic_score carries 0.30 of final_prompt_score and gates every role
   branch in classify_candidate_relevance_v2, so with it at 0.0 a 7-of-8-term
   title match scored 0.2816 -- under the 0.48 `exact` and 0.34
   `strong_related` gates -- and was classified weak_related.

3. bucket_candidates files weak_related under supporting_candidates unless it
   is a speaker_visual, and the export timeline is built from
   selected_candidates only. The named video's clips were therefore dropped
   before rendering.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402
from app.pipeline.storyline_gen.candidate_scoring import (  # noqa: E402
    annotate_candidate_relevance,
    classify_candidate_relevance_v2,
)
from app.pipeline.storyline_gen.director import bucket_candidates  # noqa: E402
from app.pipeline.storyline_gen.prompt_intent import (  # noqa: E402
    detect_prompt_intent,
    lookup_requested_source_video,
)
from app.pipeline.storyline_gen.selection import (  # noqa: E402
    apply_diversity_cap,
    select_balanced_candidates,
    source_video_cap,
)
from app.pipeline.storyline_gen.text_utils import (  # noqa: E402
    title_term_overlap,
    video_title_from_source,
)

# The real filename stem shape: <index>_<11-char youtube id>_<title>.
TARGET = (
    "0005_qifoTD_3vNc_Latin_America_Event_-_Between_Algorithms_"
    "and_Citizenship_-_Youth_Facing_AI"
)
TARGET_TITLE = "Latin America Event - Between Algorithms and Citizenship - Youth Facing AI"
OTHER = "0103_iHUEZ0WiErM_China Satellite Event"


def fragment(fragment_id: str, source_video: str = TARGET, **overrides) -> dict:
    record = {
        "fragment_id": fragment_id,
        "source_video": source_video,
        "text": "a transcript line that does not repeat the video title",
        "summary": "",
        "keywords": [],
        "semantic_score": 0.40,
        "primary_theme": "",
        "primary_chapter": "",
        "speaker": "Ana Ruiz",
        "quote_strength": 0.5,
        "clarity_score": 0.6,
        "duration": 40.0,
        "start": 10.0,
        "end": 50.0,
        "clip_usage_candidates": ["speaker_soundbite"],
    }
    record.update(overrides)
    return record


def corpus(*source_videos: str) -> list[dict]:
    return [{"source_video": sv, "fragment_id": f"{sv}_0001"} for sv in source_videos]


class TestVideoTitleFromSource:
    def test_strips_index_and_youtube_id_prefix(self):
        assert video_title_from_source({"source_video": TARGET}) == TARGET_TITLE

    def test_handles_youtube_ids_containing_underscores(self):
        # 42 of the 309 corpus ids contain an underscore; splitting on "_"
        # instead of taking a fixed 11 characters truncates them.
        record = {"source_video": "0005_qifoTD_3vNc_Latin America Event"}
        assert video_title_from_source(record) == "Latin America Event"

    def test_prefers_explicit_video_title_when_present(self):
        record = {"source_video": TARGET, "video_title": "Canonical Title"}
        assert video_title_from_source(record) == "Canonical Title"

    def test_missing_source_video_is_empty_not_an_error(self):
        assert video_title_from_source({}) == ""


class TestTitleTermOverlap:
    def test_singularizes_both_sides(self):
        # extract_topic_terms singularizes "algorithms" in the prompt to
        # "algorithm"; match_terms_in_text does not singularize the text side,
        # so this term used to miss the literal "Algorithms" in the title.
        matched, ratio = title_term_overlap(TARGET_TITLE, ["algorithm"])
        assert matched == ["algorithm"]
        assert ratio == 1.0

    def test_no_terms_is_zero_not_a_division_error(self):
        assert title_term_overlap(TARGET_TITLE, []) == ([], 0.0)


class TestLookupRequestedSourceVideo:
    def test_exact_title_resolves_to_its_own_video(self):
        source_video, ratio, matched = lookup_requested_source_video(
            {"latin", "america", "event", "between", "algorithm", "citizenship", "youth", "facing"},
            corpus(TARGET, OTHER),
        )
        assert source_video == TARGET
        assert ratio == 1.0
        assert len(matched) == 8

    def test_generic_topic_prompt_is_not_a_title_request(self):
        # "access to information" reduces to two terms; any title containing
        # both scored 1.0 on prompt-side coverage alone. The title-side ratio
        # is what rejects it.
        source_video, ratio, _ = lookup_requested_source_video(
            {"access", "information"},
            corpus("0267_BL6fn60b99Q_(6M) S Agwang - Access to information in the digital age"),
        )
        assert source_video is None
        assert ratio == 0.0

    def test_phrase_matching_several_titles_equally_is_not_a_title_request(self):
        # "satellite event" fits every "<Country> Satellite Event" equally, so
        # it is a topic, not a named video.
        source_video, _, _ = lookup_requested_source_video(
            {"satellite", "event"},
            corpus("0090_zy45aKgJUiY_Botswana Satellite event", OTHER),
        )
        assert source_video is None

    def test_speaker_name_is_stripped_from_the_title_side_too(self):
        # build_prompt_profile removes the speaker from the prompt's terms; if
        # the title keeps them, a title that is mostly a name never clears the
        # title-side ratio.
        source_video, _, _ = lookup_requested_source_video(
            {"inclusion", "work"},
            corpus("0044_AbCdEfGhIjK_Celine Caruso Dixon - Can We Use AI for Inclusion Work"),
            requested_speaker="Celine Caruso Dixon",
        )
        assert source_video == "0044_AbCdEfGhIjK_Celine Caruso Dixon - Can We Use AI for Inclusion Work"

    def test_empty_topic_terms_never_matches(self):
        assert lookup_requested_source_video(set(), corpus(TARGET)) == (None, 0.0, [])


class TestTitleMatchSurvivesToTheFinalVideo:
    """The end-to-end defect: intent -> classification -> bucketing."""

    def test_named_video_is_classified_exact_and_selected(self):
        records = corpus(TARGET, OTHER)
        intent = detect_prompt_intent(TARGET_TITLE, known_speakers=[], records=records)
        assert intent["requested_source_video"] == TARGET

        candidates = [fragment(f"f{i}") for i in range(4)]
        classified = annotate_candidate_relevance(candidates, TARGET_TITLE, intent, known_speakers=[])
        assert {c["relevance_role"] for c in classified} == {"exact"}
        assert all(c["final_video_allowed"] for c in classified)

        buckets = bucket_candidates(classified, intent)
        assert len(buckets["selected_candidates"]) == 4
        assert buckets["supporting_candidates"] == []

    def test_regression_zero_semantic_score_no_longer_buries_the_clip(self):
        # The exact failing condition: a keyword-channel candidate whose
        # semantic_score never got filled in.
        records = corpus(TARGET, OTHER)
        intent = detect_prompt_intent(TARGET_TITLE, known_speakers=[], records=records)
        enriched = classify_candidate_relevance_v2(fragment("f0", semantic_score=0.0), intent)
        assert enriched["relevance_role"] == "exact"
        assert enriched["final_video_allowed"] is True

    def test_other_videos_are_unaffected_by_the_title_match(self):
        records = corpus(TARGET, OTHER)
        intent = detect_prompt_intent(TARGET_TITLE, known_speakers=[], records=records)
        enriched = classify_candidate_relevance_v2(
            fragment("other", source_video=OTHER, semantic_score=0.0), intent
        )
        assert enriched["title_match"] is False
        assert enriched["relevance_role"] != "exact"

    def test_ordinary_topic_prompt_gets_no_title_bonus(self):
        records = corpus(TARGET, OTHER)
        prompt = "make a documentary on data privacy regulation"
        intent = detect_prompt_intent(prompt, known_speakers=[], records=records)
        assert intent["requested_source_video"] is None
        enriched = classify_candidate_relevance_v2(fragment("f0"), intent)
        assert enriched["title_score"] == 0.0


class TestSourceCapRelaxation:
    def test_requested_video_gets_the_relaxed_cap(self):
        assert source_video_cap(TARGET, 2, requested_source_video=TARGET) == (
            settings.max_uses_per_requested_source_video
        )

    def test_every_other_video_keeps_the_default_cap(self):
        assert source_video_cap(OTHER, 2, requested_source_video=TARGET) == 2

    def test_no_title_request_leaves_the_default_cap_everywhere(self):
        assert source_video_cap(TARGET, 2, requested_source_video=None) == 2

    def test_relaxed_cap_is_never_lower_than_the_default(self):
        assert source_video_cap(TARGET, 99, requested_source_video=TARGET) == 99

    def test_diversity_cap_keeps_more_clips_from_the_named_video(self):
        candidates = [fragment(f"f{i}") for i in range(8)]
        assert len(apply_diversity_cap(candidates, max_per_source=2, max_per_speaker=2)) == 2
        relaxed = apply_diversity_cap(
            candidates, max_per_source=2, max_per_speaker=2, requested_source_video=TARGET
        )
        assert len(relaxed) == settings.max_uses_per_requested_source_video

    def test_speaker_cap_does_not_re_impose_the_source_cap(self):
        # One talk is usually one speaker, so a 2-per-speaker cap would undo
        # the relaxed source cap on its own.
        candidates = [fragment(f"f{i}", speaker="Ana Ruiz") for i in range(8)]
        relaxed = apply_diversity_cap(
            candidates, max_per_source=2, max_per_speaker=2, requested_source_video=TARGET
        )
        assert len(relaxed) > 2

    def test_balanced_selection_honours_the_relaxed_cap(self):
        candidates = [
            fragment(f"t{i}", relevance_role="exact", final_prompt_score=0.9) for i in range(8)
        ] + [
            fragment(f"o{i}", source_video=OTHER, speaker="Li Wei",
                     relevance_role="exact", final_prompt_score=0.8) for i in range(4)
        ]
        selected = select_balanced_candidates(
            candidates, limit=8, requested_source_video=TARGET
        )
        from_target = [c for c in selected if c["source_video"] == TARGET]
        assert len(from_target) >= settings.max_uses_per_requested_source_video

    def test_balanced_selection_without_a_title_request_is_unchanged(self):
        candidates = [
            fragment(f"t{i}", relevance_role="exact", final_prompt_score=0.9) for i in range(8)
        ]
        selected = select_balanced_candidates(candidates, limit=8)
        # First pass caps at MAX_SOURCE_VIDEO_OCCURRENCES; the second pass
        # then backfills unused slots, which is pre-existing behaviour.
        assert len(selected) == 8

    def test_selection_score_survives_a_candidate_with_no_scores_at_all(self):
        # select_balanced_candidates read record["semantic_score"] directly,
        # so a keyword-channel candidate with every score zero raised
        # KeyError rather than being ranked last.
        bare = fragment("bare")
        for key in ("semantic_score", "final_prompt_score", "prompt_relevance_score"):
            bare.pop(key, None)
        assert select_balanced_candidates([bare], limit=1)[0]["selection_score"] >= 0.0


class TestSemanticScoreBackfill:
    """_backfill_semantic_scores reproduces what the ANN query would have
    returned: the collection is cosine-space with unit-norm vectors, so a dot
    product of normalized vectors equals Chroma's own 1 - distance."""

    @pytest.fixture
    def retrieval(self, monkeypatch):
        from app.pipeline.storyline_gen import retrieval as module

        vectors = {
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        class FakeCollection:
            def get(self, ids, include):
                present = [i for i in ids if i in vectors]
                return {"ids": present, "embeddings": [vectors[i] for i in present]}

        monkeypatch.setattr(module, "_get_chroma_collection", lambda: FakeCollection())
        monkeypatch.setattr(module, "_QUERY_VECTOR_CACHE", {})
        # Deliberately un-normalized, to prove the backfill normalizes it.
        monkeypatch.setattr(module, "embed_query", lambda p: np.array([3.0, 0.0, 0.0], dtype=np.float32))
        return module

    def test_fills_in_missing_scores_on_the_cosine_scale(self, retrieval):
        candidates = [{"fragment_id": "a"}, {"fragment_id": "b"}]
        assert retrieval._backfill_semantic_scores(candidates, "prompt") == 2
        assert candidates[0]["semantic_score"] == pytest.approx(1.0)
        assert candidates[1]["semantic_score"] == pytest.approx(0.0)
        assert candidates[0]["semantic_score_backfilled"] is True

    def test_leaves_ann_scored_candidates_alone(self, retrieval):
        candidates = [{"fragment_id": "a", "semantic_score": 0.77}]
        assert retrieval._backfill_semantic_scores(candidates, "prompt") == 0
        assert candidates[0]["semantic_score"] == 0.77
        assert "semantic_score_backfilled" not in candidates[0]

    def test_candidate_missing_from_the_store_still_gets_a_usable_score(self, retrieval):
        # Otherwise select_balanced_candidates raises KeyError on it.
        candidates = [{"fragment_id": "not-in-chroma"}]
        retrieval._backfill_semantic_scores(candidates, "prompt")
        assert candidates[0]["semantic_score"] == 0.0

    def test_query_vector_is_embedded_once_per_prompt(self, retrieval, monkeypatch):
        calls = []

        def counting_embed(prompt):
            calls.append(prompt)
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        monkeypatch.setattr(retrieval, "embed_query", counting_embed)
        retrieval._query_vector("same prompt")
        retrieval._query_vector("same prompt")
        assert len(calls) == 1
