"""Tests for the duration release gate in scripts/test_pipeline_timing.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gate():
    path = PROJECT_ROOT / "scripts" / "test_pipeline_timing.py"
    spec = importlib.util.spec_from_file_location("timing_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _isolate_exports(gate, tmp_path, monkeypatch):
    """Keep gate artifacts out of the real data/exports directory."""
    monkeypatch.setattr(gate, "EXPORTS_DIR", tmp_path)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _good_artifacts(tmp_path: Path, gate, target: int = 180):
    storyline = _write(tmp_path / "storyline.json", {"duration_budget": {"stage": "07"}})
    validated = _write(tmp_path / "validated.json", {
        "duration_budget": {"stage": "07b"},
        "validated_total_seconds": target * 1.05,
    })
    manifest = _write(tmp_path / "manifest.json", {
        "duration_budget": {"stage": "08", "meets_floor": True},
        "preflight_failures": [],
    })
    video = tmp_path / "out.mp4"
    video.write_bytes(b"mp4")
    _write(gate.EXPORTS_DIR / f"{video.stem}_export_manifest.json", {
        "duration_budget": {"stage": "09"},
        "timeline": [{}, {}, {}],
        "segments": [{}, {}, {}],
    })
    return storyline, validated, manifest, video


class TestCheckDurationGate:
    def test_all_good_passes(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        failures, warnings, actual = gate.check_duration_gate(180, s, v, m, video)
        assert failures == []
        assert actual == 190.0

    def test_below_target_is_warning_not_failure(self, gate, tmp_path, monkeypatch):
        # Policy: below target due to content exhaustion → warn + ship, not fail.
        monkeypatch.setattr(gate, "probe_duration", lambda p: 150.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        failures, warnings, _ = gate.check_duration_gate(180, s, v, m, video)
        assert failures == []
        assert any("below target" in w for w in warnings)

    def test_render_drift_is_warning_not_failure(self, gate, tmp_path, monkeypatch):
        # Rendered under the summed plan (fades/-shortest/relay drift) but all
        # segments present and target met → warning, not failure.
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(gate.EXPORTS_DIR / f"{video.stem}_export_manifest.json", {
            "duration_budget": {"stage": "09", "planned_total_seconds": 250.0},
            "timeline": [{}, {}, {}],
            "segments": [{}, {}, {}],  # all 3 rendered — no drop
        })
        failures, warnings, _ = gate.check_duration_gate(180, s, v, m, video)
        assert failures == []
        assert any("render drift" in w for w in warnings)

    def test_dropped_segment_count_still_fails(self, gate, tmp_path, monkeypatch):
        # The real integrity signal: a planned segment failed to render.
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(gate.EXPORTS_DIR / f"{video.stem}_export_manifest.json", {
            "duration_budget": {"stage": "09", "planned_total_seconds": 200.0},
            "timeline": [{}] * 12,
            "segments": [{}] * 9,  # 3 planned segments never rendered
        })
        failures, _, _ = gate.check_duration_gate(180, s, v, m, video)
        assert any("9/12" in f for f in failures)

    def test_overshoot_is_warning_not_failure(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 300.0)  # 1.67× of 180
        s, v, m, video = _good_artifacts(tmp_path, gate)
        failures, warnings, _ = gate.check_duration_gate(180, s, v, m, video)
        assert failures == []
        assert any("overshoot" in w for w in warnings)

    def test_missing_budget_fails(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(s, {"title": "no budget here"})
        failures, _, _ = gate.check_duration_gate(180, s, v, m, video)
        assert any("no duration_budget" in f for f in failures)

    def test_preflight_failures_fail_gate(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(m, {
            "duration_budget": {"meets_floor": True},
            "preflight_failures": ["item 3: voiceover audio missing"],
        })
        failures, _, _ = gate.check_duration_gate(180, s, v, m, video)
        assert any("preflight failure" in f for f in failures)

    def test_dropped_segments_fail_gate(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(gate.EXPORTS_DIR / f"{video.stem}_export_manifest.json", {
            "duration_budget": {"stage": "09"},
            "timeline": [{}] * 12,
            "segments": [{}] * 10,  # two planned segments never rendered
        })
        failures, _, _ = gate.check_duration_gate(180, s, v, m, video)
        assert any("10/12" in f for f in failures)

    def test_validated_below_target_is_warning(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        _write(v, {"duration_budget": {"stage": "07b"}, "validated_total_seconds": 100.0})
        failures, warnings, _ = gate.check_duration_gate(180, s, v, m, video)
        assert not any("validated" in f for f in failures)
        assert any("validated" in w for w in warnings)

    def test_skipped_stage_artifact_is_warning(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        failures, warnings, _ = gate.check_duration_gate(180, None, v, m, video)
        assert failures == []
        assert any("not produced by this run" in w for w in warnings)

    def test_missing_video_fails(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 0.0)
        s, v, m, video = _good_artifacts(tmp_path, gate)
        failures, _, _ = gate.check_duration_gate(180, s, v, m, tmp_path / "nope.mp4")
        assert any("final video missing" in f for f in failures)


class TestGateResultForRun:
    """Regression: a stage exiting non-zero (e.g. 09's own strict-duration
    ExportIntegrityError) must still produce a gate verdict when a video was
    actually written, instead of being treated as 'no video to check'."""

    def test_failed_stage_with_video_still_gated(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate, target=180)
        # A stage exited non-zero AND a segment was dropped — gate must still run
        # and surface the drop as a failure (not silently skip because ok=False).
        _write(gate.EXPORTS_DIR / f"{video.stem}_export_manifest.json", {
            "duration_budget": {"stage": "09", "planned_total_seconds": 200.0},
            "timeline": [{}] * 10,
            "segments": [{}] * 7,  # 3 dropped
        })
        run = {
            "ok": False,  # 09_export_video.py exited 1
            "storyline_path": s,
            "validated_path": v,
            "manifest_path": m,
            "output_video": video,
        }
        result = gate.gate_result_for_run(180, run)
        assert result["pipeline_ok"] is False
        assert any("7/10" in f for f in result["failures"])

    def test_failed_stage_without_video_reports_generic_failure(self, gate):
        run = {"ok": False, "storyline_path": None, "output_video": None}
        result = gate.gate_result_for_run(180, run)
        assert result["failures"] == ["pipeline failed before producing a video"]
        assert result["actual"] == 0.0

    def test_successful_run_passes_through_to_check(self, gate, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "probe_duration", lambda p: 190.0)
        s, v, m, video = _good_artifacts(tmp_path, gate, target=180)
        run = {
            "ok": True,
            "storyline_path": s,
            "validated_path": v,
            "manifest_path": m,
            "output_video": video,
        }
        result = gate.gate_result_for_run(180, run)
        assert result["pipeline_ok"] is True
        assert result["failures"] == []
        assert result["actual"] == 190.0


class TestStructuralChecks:
    """The documentary-quality contract: real speaker per section, no silent
    content segments, no duplicate fragments."""

    def _good_timeline(self):
        return [
            {"type": "title_card"},
            {"segment_type": "narration_broll", "section_id": "S1", "audio_mode": "ai_voiceover", "muted": True},
            {"segment_type": "speaker_clip", "section_id": "S1", "fragment_id": "a", "audio_mode": "source_audio"},
            {"segment_type": "narration_broll", "section_id": "S2", "audio_mode": "ai_voiceover", "muted": True},
            {"segment_type": "speaker_clip", "section_id": "S2", "fragment_id": "b", "audio_mode": "source_audio"},
            {"type": "closing_card"},
        ]

    def test_clean_timeline_passes(self, gate):
        assert gate._check_structure(self._good_timeline()) == []

    def test_silent_content_segment_flagged(self, gate):
        tl = self._good_timeline()
        tl.append({"segment_type": "archive_visual", "section_id": "S3",
                   "fragment_id": "c", "audio_mode": "silent", "muted": True})
        failures = gate._check_structure(tl)
        assert any("silent content" in f for f in failures)

    def test_duplicate_fragment_flagged(self, gate):
        tl = self._good_timeline()
        tl.append({"segment_type": "speaker_clip", "section_id": "S3",
                   "fragment_id": "a", "audio_mode": "source_audio"})  # 'a' already used
        failures = gate._check_structure(tl)
        assert any("duplicate fragments" in f for f in failures)

    def test_section_without_speaker_flagged(self, gate):
        tl = self._good_timeline()
        # S3 has only narration, no speaker clip
        tl.append({"segment_type": "narration_broll", "section_id": "S3", "audio_mode": "ai_voiceover", "muted": True})
        failures = gate._check_structure(tl)
        assert any("no speaker clip" in f and "S3" in f for f in failures)

    def test_narration_broll_muted_is_not_flagged_silent(self, gate):
        # narration_broll is muted=True (source muted) but carries AI voiceover —
        # must NOT be counted as a silent content segment.
        tl = [
            {"segment_type": "narration_broll", "section_id": "S1", "audio_mode": "ai_voiceover", "muted": True},
            {"segment_type": "speaker_clip", "section_id": "S1", "fragment_id": "a", "audio_mode": "source_audio"},
        ]
        assert gate._check_structure(tl) == []

    def test_empty_timeline_no_crash(self, gate):
        assert gate._check_structure([]) == []
