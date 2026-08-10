"""Tests for the speaker-first documentary candidate selection (Phase A).

Root cause it fixes: sort_final_video_candidates ranked speaker-eligibility only
as a late tiebreaker, so the cap-to-N crowded out real speaker voices in favour
of marginally-higher-scoring narration-only clips — collapsing usable_speaker to
1-2 even when 12+ good speaker candidates existed in the pool.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.pipeline.storyline_gen.selection as selection_module  # noqa: E402


@pytest.fixture(scope="module")
def s07():
    return selection_module


def _narration(i: int, score: float):
    return {"fragment_id": f"narr_{i}", "source_video": f"vn_{i}", "speaker": f"sn_{i}",
            "relevance_role": "exact", "prompt_relevance_score": score,
            "speaker_soundbite_allowed": False}


def _speaker(i: int, score: float):
    return {"fragment_id": f"spk_{i}", "source_video": f"vs_{i}", "speaker": f"person_{i}",
            "relevance_role": "strong_related", "prompt_relevance_score": score,
            "speaker_soundbite_allowed": True}


class TestSpeakerFirstSelection:
    def test_speakers_survive_tight_cap(self, s07):
        # 20 high-scoring narration + 8 lower-scoring speakers; cap 10.
        # Old logic: ~0 speakers survive. New: speakers reserved first.
        selected = [_narration(i, 0.9 - i * 0.001) for i in range(20)] + \
                   [_speaker(i, 0.7 - i * 0.001) for i in range(8)]
        result, diag = s07.select_documentary_candidates(
            selected, target_duration_seconds=120, cap=10, requested_speaker=None)
        speakers = [c for c in result if c.get("speaker_soundbite_allowed")]
        assert len(speakers) >= 1
        assert diag["speaker_slots_reserved"] >= 1
        assert len(result) <= 10

    def test_all_speakers_kept_when_room(self, s07):
        selected = [_narration(i, 0.9) for i in range(20)] + [_speaker(i, 0.7) for i in range(8)]
        result, diag = s07.select_documentary_candidates(
            selected, target_duration_seconds=600, cap=50, requested_speaker=None)
        speakers = [c for c in result if c.get("speaker_soundbite_allowed")]
        assert len(speakers) == 8  # all speaker-eligible retained

    def test_short_duration_uses_one_speaker_per_section(self, s07):
        assert s07.speakers_per_section_for_duration(120) == 1

    def test_long_duration_allows_two_per_section(self, s07):
        assert s07.speakers_per_section_for_duration(600) == 2

    def test_section_count_scales_with_duration(self, s07):
        assert s07.estimate_section_count(60) < s07.estimate_section_count(600)

    def test_diversity_cap_within_reservation(self, s07):
        # 6 speaker clips all from the SAME speaker → max 2 reserved.
        selected = [
            {"fragment_id": f"spk_{i}", "source_video": f"vs_{i}", "speaker": "same_person",
             "relevance_role": "exact", "prompt_relevance_score": 0.8,
             "speaker_soundbite_allowed": True}
            for i in range(6)
        ]
        result, diag = s07.select_documentary_candidates(
            selected, target_duration_seconds=600, cap=50, requested_speaker=None)
        speakers = [c for c in result if c.get("speaker_soundbite_allowed")]
        assert len(speakers) == 2  # speaker diversity cap

    def test_empty_pool_returns_empty(self, s07):
        result, diag = s07.select_documentary_candidates(
            [], target_duration_seconds=600, cap=50, requested_speaker=None)
        assert result == []
        assert diag["speaker_slots_reserved"] == 0
