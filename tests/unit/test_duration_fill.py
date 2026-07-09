"""Phase 2/3 hardening tests: fill-to-target selection, plan-time window clamping,
and narration grounding checks."""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.pipeline.timeline_budget as tb  # noqa: E402
from app.pipeline.timeline_budget import (  # noqa: E402
    clamp_timeline_windows_to_source,
    fill_timeline_to_target,
    timeline_total_seconds,
)


@pytest.fixture(autouse=True)
def _no_real_probe(monkeypatch):
    """By default, sources probe as 'unknown length' so tests are deterministic."""
    monkeypatch.setattr(tb, "probe_source_video_duration", lambda sv, fid=None: 0.0)


def _speaker_candidate(i: int, seconds: float = 50.0, source: str | None = None, score: float = 1.0):
    return {
        "fragment_id": f"frag_{i:03d}",
        "source_video": source or f"video_{i:03d}",
        "start": 100.0,
        "end": 100.0 + seconds,
        "speaker_clip_allowed": True,
        "moment_text": "a real quote from the archive",
        "final_prompt_score": score,
    }


def _base_timeline(content_seconds: float = 100.0):
    return [
        {"type": "title_card", "segment_type": "title_card", "duration": 5.0},
        {"segment_type": "speaker_clip", "fragment_id": "existing_1",
         "source_video": "video_existing", "start": 0.0, "end": content_seconds,
         "duration": content_seconds},
        {"type": "closing_card", "segment_type": "closing_card", "duration": 5.0},
    ]


class TestFillTimelineToTarget:
    def test_fills_until_floor_met(self):
        timeline = _base_timeline(100.0)  # 110s planned
        pool = {"selected": [_speaker_candidate(i) for i in range(20)]}
        report = fill_timeline_to_target(timeline, pool, 300)
        assert report["status"] == "filled"
        assert report["total_after_seconds"] >= 300 * 1.02
        assert timeline_total_seconds(timeline) == report["total_after_seconds"]

    def test_added_segments_inserted_before_closing_card(self):
        timeline = _base_timeline(100.0)
        pool = {"selected": [_speaker_candidate(1)]}
        fill_timeline_to_target(timeline, pool, 200)
        assert (timeline[-1].get("type")) == "closing_card"
        assert timeline[-2].get("added_by_duration_fill") is True

    def test_already_sufficient_adds_nothing(self):
        timeline = _base_timeline(600.0)
        pool = {"selected": [_speaker_candidate(i) for i in range(5)]}
        report = fill_timeline_to_target(timeline, pool, 300)
        assert report["status"] == "already_sufficient"
        assert report["segments_added"] == []

    def test_per_source_cap_held_at_two(self):
        # 10 candidates ALL from one source: cap stays at 2 (no relaxation to 3),
        # so the same talk never dominates. Pool then exhausted → content_exhausted.
        timeline = _base_timeline(10.0)
        pool = {"selected": [_speaker_candidate(i, seconds=40.0, source="one_video") for i in range(10)]}
        report = fill_timeline_to_target(timeline, pool, 500)
        assert len(report["segments_added"]) == 2  # capped at 2 per source
        assert report["status"] == "content_exhausted"
        assert report["shortfall_seconds"] > 0

    def test_supporting_pool_flagged_needs_validation(self):
        timeline = _base_timeline(10.0)
        pool = {
            "selected": [],
            "supporting": [_speaker_candidate(i) for i in range(3)],
        }
        fill_timeline_to_target(timeline, pool, 120)
        added = [i for i in timeline if i.get("added_by_duration_fill")]
        assert added and all(i["needs_validation"] for i in added)
        assert all(i["fill_stage"] == "supporting" for i in added)

    def test_content_exhausted_reported_not_padded(self):
        timeline = _base_timeline(10.0)
        report = fill_timeline_to_target(timeline, {"selected": []}, 600)
        assert report["status"] == "content_exhausted"
        # nothing invented: timeline unchanged
        assert len(timeline) == 3

    def test_non_speaker_candidates_are_never_added(self):
        """Policy change: non-speaker candidates are NOT turned into silent
        archive_visual filler — they're skipped entirely. We ship a shorter
        honest video rather than pad with muted b-roll."""
        timeline = _base_timeline(10.0)
        candidate = _speaker_candidate(1)
        candidate.pop("speaker_clip_allowed")  # now non-speaker-eligible
        # ensure classify_timeline_segment_type won't call it a speaker_clip either
        candidate["use_as"] = "archive_visual"
        report = fill_timeline_to_target(timeline, {"selected": [candidate]}, 120)
        added = [i for i in timeline if i.get("added_by_duration_fill")]
        assert added == []
        assert report["status"] == "content_exhausted"

    def test_added_fill_is_real_speaker_clip_with_audio(self):
        timeline = _base_timeline(10.0)
        pool = {"selected": [_speaker_candidate(i) for i in range(4)]}
        fill_timeline_to_target(timeline, pool, 200)
        added = [i for i in timeline if i.get("added_by_duration_fill")]
        assert added, "expected speaker clips to be added"
        for seg in added:
            assert seg["segment_type"] == "speaker_clip"
            assert seg["audio_mode"] == "source_audio"
            assert seg["muted"] is False
            assert seg.get("render_duration", 0) > 0

    def test_window_clamped_to_probed_source(self, monkeypatch):
        monkeypatch.setattr(tb, "probe_source_video_duration", lambda sv, fid=None: 120.0)
        timeline = _base_timeline(10.0)
        candidate = _speaker_candidate(1, seconds=60.0)  # window 100→160 vs 120s stream
        fill_timeline_to_target(timeline, {"selected": [candidate]}, 60)
        added = [i for i in timeline if i.get("added_by_duration_fill")][0]
        assert added["end"] == 119.9


class TestClampTimelineWindows:
    def test_clamps_end_to_stream(self, monkeypatch):
        monkeypatch.setattr(tb, "probe_source_video_duration", lambda sv, fid=None: 639.0)
        timeline = [{"fragment_id": "f1", "source_video": "v1", "start": 630.8, "end": 679.7,
                     "duration": 48.9}]
        notes = clamp_timeline_windows_to_source(timeline)
        # start 630.8 within 1s of.. no: 630.8 < 638 so clamp end
        assert timeline[0]["end"] == 638.9
        assert notes and "clamped" in notes[0]

    def test_drops_window_beyond_stream(self, monkeypatch):
        monkeypatch.setattr(tb, "probe_source_video_duration", lambda sv, fid=None: 100.0)
        timeline = [{"fragment_id": "f1", "source_video": "v1", "start": 99.5, "end": 130.0}]
        notes = clamp_timeline_windows_to_source(timeline)
        assert timeline == []
        assert notes and "dropped" in notes[0]

    def test_unknown_stream_left_alone(self):
        timeline = [{"fragment_id": "f1", "source_video": "v1", "start": 10.0, "end": 500.0}]
        clamp_timeline_windows_to_source(timeline)
        assert timeline[0]["end"] == 500.0


class TestNarrationGrounding:
    @pytest.fixture(scope="class")
    def validate_mod(self):
        path = PROJECT_ROOT / "scripts" / "07b_validate_storyline_with_llm.py"
        spec = importlib.util.spec_from_file_location("validate_07b", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_grounded_text_passes(self, validate_mod):
        sources = "AI can be misused. We saw 40% of cases involve identity abuse, said Dr. Green."
        issues = validate_mod._narration_grounding_issues(
            "Experts warn that AI can be misused — 40% of cases involve identity abuse.",
            sources, {"cordel green"},
        )
        assert issues == []

    def test_invented_person_rejected(self, validate_mod):
        issues = validate_mod._narration_grounding_issues(
            "According to Dr. Fictional, the risks are growing.",
            "some transcript about AI risks", set(),
        )
        assert any("invented person" in i for i in issues)

    def test_unsourced_statistic_rejected(self, validate_mod):
        issues = validate_mod._narration_grounding_issues(
            "Studies show 87% of consumers cannot detect deepfakes.",
            "a transcript that never mentions figures", set(),
        )
        assert any("unsourced statistic" in i for i in issues)

    def test_allowed_speaker_name_accepted(self, validate_mod):
        issues = validate_mod._narration_grounding_issues(
            "As Professor Roumate explains, sovereignty matters.",
            "transcript text", {"fatima roumate"},
        )
        assert issues == []

    def test_build_grounding_material(self, validate_mod):
        blob, names = validate_mod._build_grounding_material([
            {"moment_text": "quote one", "speaker": "Jane Doe"},
            {"quote_text": "quote two", "speaker_attribution": "John Smith"},
        ])
        assert "quote one" in blob and "quote two" in blob
        assert names == {"jane doe", "john smith"}
