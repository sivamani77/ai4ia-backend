import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    build_duration_budget,
    normalize_candidate_time_bounds,
    segment_effective_duration,
    timeline_card_duration,
    timeline_segment_count_for_duration,
    timeline_total_seconds,
)


class TestSegmentCountBuckets:
    def test_long_targets_get_more_segments(self):
        assert timeline_segment_count_for_duration(60) == 4
        assert timeline_segment_count_for_duration(120) == 6
        assert timeline_segment_count_for_duration(300) == 8
        assert timeline_segment_count_for_duration(600) == 12
        assert timeline_segment_count_for_duration(900) == 15

    def test_card_duration(self):
        assert timeline_card_duration(60) == 4
        assert timeline_card_duration(600) == 5


class TestSegmentEffectiveDuration:
    def test_render_duration_wins_over_everything(self):
        item = {"render_duration": 8.2, "render_start": 658.6, "render_end": 666.8,
                "duration": 58.04, "start": 630.8, "end": 679.744}
        assert segment_effective_duration(item) == 8.2

    def test_render_window_wins_over_plan_fields(self):
        item = {"render_start": 100.0, "render_end": 122.0, "duration": 58.0}
        assert segment_effective_duration(item) == 22.0

    def test_duration_used_when_no_render_window(self):
        assert segment_effective_duration({"duration": 5.0}) == 5.0

    def test_start_end_fallback(self):
        assert segment_effective_duration({"start": 10.0, "end": 59.5}) == 49.5

    def test_start_seconds_alias(self):
        assert segment_effective_duration({"start_seconds": 10, "end_seconds": 25}) == 15.0

    def test_unmeasurable_is_zero(self):
        assert segment_effective_duration({"type": "title_card"}) == 0.0
        assert segment_effective_duration({"duration": "not a number"}) == 0.0

    def test_invalid_render_window_falls_through(self):
        # render_end <= render_start must not produce a negative duration
        item = {"render_start": 50.0, "render_end": 40.0, "duration": 12.0}
        assert segment_effective_duration(item) == 12.0


class TestTimelineTotal:
    def test_mixed_timeline_sums_like_renderer(self):
        timeline = [
            {"type": "title_card", "duration": 5.0},
            {"segment_type": "speaker_clip", "render_start": 658.6, "render_end": 666.8,
             "start": 630.8, "end": 679.744, "duration": 58.04},
            {"segment_type": "narration_broll", "duration": 78.28},
            {"type": "closing_card", "duration": 5.0},
        ]
        assert timeline_total_seconds(timeline) == 5.0 + 8.2 + 78.28 + 5.0

    def test_empty_and_none(self):
        assert timeline_total_seconds([]) == 0.0
        assert timeline_total_seconds(None) == 0.0


class TestBudgetReport:
    def test_shortfall_reported(self):
        timeline = [{"duration": 100.0}, {"duration": 200.0}]
        budget = build_duration_budget(timeline, 600, stage="test")
        assert budget["stage"] == "test"
        assert budget["planned_total_seconds"] == 300.0
        assert budget["meets_floor"] is False
        assert budget["shortfall_seconds"] == 300.0
        assert budget["overshoot_seconds"] == 0.0

    def test_floor_met_with_overshoot(self):
        budget = build_duration_budget([{"duration": 650.0}], 600, stage="test")
        assert budget["meets_floor"] is True
        assert budget["shortfall_seconds"] == 0.0
        assert budget["overshoot_seconds"] == 50.0
        assert budget["overshoot_warning"] is False

    def test_excessive_overshoot_flagged(self):
        budget = build_duration_budget([{"duration": 800.0}], 600, stage="test")
        assert budget["overshoot_warning"] is True

    def test_no_target_omits_floor_fields(self):
        budget = build_duration_budget([{"duration": 10.0}], None, stage="test")
        assert "meets_floor" not in budget
        assert budget["planned_total_seconds"] == 10.0

    def test_unmeasured_segments_counted(self):
        timeline = [{"type": "title_card"}, {"duration": 30.0}]
        budget = build_duration_budget(timeline, 60, stage="test")
        assert budget["unmeasured_segment_count"] == 1
        assert budget["segment_count"] == 2
        assert budget["segments"][0]["seconds"] == 0.0


class TestNormalizeBounds:
    def test_explicit_none_start_falls_back_to_alias(self):
        # 08's old copy failed this: dict.get(key, default) ignores explicit None
        start, end, dur = normalize_candidate_time_bounds(
            {"start": None, "start_seconds": 10.0, "end": None, "end_seconds": 40.0}
        )
        assert (start, end, dur) == (10.0, 40.0, 30.0)

    def test_duration_derived_from_bounds(self):
        assert normalize_candidate_time_bounds({"start": 5, "end": 17})[2] == 12.0

    def test_end_derived_from_duration(self):
        assert normalize_candidate_time_bounds({"start": 5, "duration": 10})[1] == 15.0
