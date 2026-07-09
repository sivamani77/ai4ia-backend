"""Tests for the source-video-duration sidecar lookup used by
probe_source_video_duration, and the index-build script's probe/merge logic.

Fixes the blind spot where stages 07/07b run in S3 mode without the raw video
files present, so probe_source_video_duration silently returned 0.0 for every
clip and the plan-time duration clamp/floor check never saw real footage
limits. Only stage 09 (which does download videos) could ever discover a
shortfall — by then it's too late to cheaply add more content.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.pipeline.timeline_budget as tb  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_module_caches(monkeypatch):
    """Each test gets a clean in-memory index cache and per-key duration cache."""
    monkeypatch.setattr(tb, "_source_duration_index", None)
    monkeypatch.setattr(tb, "_source_duration_cache", {})


class TestSidecarLookup:
    def test_sidecar_hit_used_without_touching_disk(self, tmp_path, monkeypatch):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"video_A": {"video_stream_seconds": 123.4}}))
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", index_path)

        def _boom(*a, **k):
            raise AssertionError("should not shell out to ffprobe when the sidecar has the key")
        monkeypatch.setattr(tb.subprocess, "run", _boom)

        assert tb.probe_source_video_duration("video_A") == 123.4

    def test_sidecar_miss_falls_back_to_local_probe(self, tmp_path, monkeypatch):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"other_video": {"video_stream_seconds": 99.0}}))
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", index_path)
        monkeypatch.setattr(tb, "RAW_VIDEO_DIR", tmp_path)

        local_file = tmp_path / "video_B.mp4"
        local_file.write_bytes(b"fake")

        class FakeResult:
            stdout = "42.5\n"
        monkeypatch.setattr(tb.subprocess, "run", lambda *a, **k: FakeResult())

        assert tb.probe_source_video_duration("video_B") == 42.5

    def test_missing_sidecar_file_falls_back_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", tmp_path / "does_not_exist.json")
        monkeypatch.setattr(tb, "RAW_VIDEO_DIR", tmp_path)
        # no local file either — genuinely unknown
        assert tb.probe_source_video_duration("nonexistent") == 0.0

    def test_zero_duration_in_sidecar_falls_back_not_trusted(self, tmp_path, monkeypatch):
        """A 0.0 entry (failed probe at index-build time) must not shadow a
        successful local re-probe — 0.0 is 'unknown', not a real answer."""
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"video_C": {"video_stream_seconds": 0.0}}))
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", index_path)
        monkeypatch.setattr(tb, "RAW_VIDEO_DIR", tmp_path)

        local_file = tmp_path / "video_C.mp4"
        local_file.write_bytes(b"fake")

        class FakeResult:
            stdout = "77.0\n"
        monkeypatch.setattr(tb.subprocess, "run", lambda *a, **k: FakeResult())

        assert tb.probe_source_video_duration("video_C") == 77.0

    def test_result_is_cached_per_process(self, tmp_path, monkeypatch):
        index_path = tmp_path / "index.json"
        index_path.write_text(json.dumps({"video_D": {"video_stream_seconds": 55.0}}))
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", index_path)

        assert tb.probe_source_video_duration("video_D") == 55.0
        # second call must not even re-read the index file
        monkeypatch.setattr(tb, "SOURCE_DURATION_INDEX_PATH", tmp_path / "gone.json")
        assert tb.probe_source_video_duration("video_D") == 55.0

    def test_env_override_changes_index_path(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_index.json"
        custom.write_text(json.dumps({"video_E": {"video_stream_seconds": 10.0}}))
        monkeypatch.setenv("AI4IA_SOURCE_DURATION_INDEX", str(custom))
        # Re-import-style: the module-level constant is read once at import time in
        # production, but the env var is honoured by a fresh process — verify the
        # env-driven default computation directly.
        import importlib
        spec = importlib.util.spec_from_file_location(
            "timeline_budget_envtest", _BACKEND_DIR / "app" / "pipeline" / "timeline_budget.py"
        )
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        assert fresh.SOURCE_DURATION_INDEX_PATH == custom


class TestBuildSourceDurationIndex:
    @pytest.fixture(scope="class")
    def build_index_mod(self):
        path = PROJECT_ROOT / "scripts" / "build_source_duration_index.py"
        spec = importlib.util.spec_from_file_location("build_source_duration_index", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_skips_unchanged_files(self, build_index_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(build_index_mod, "RAW_VIDEO_DIR", tmp_path)
        video = tmp_path / "video_X.mp4"
        video.write_bytes(b"x" * 1000)

        calls = []
        monkeypatch.setattr(
            build_index_mod, "probe_video_stream_duration",
            lambda p: (calls.append(p), 60.0)[1],
        )
        monkeypatch.setattr(build_index_mod, "INDEX_PATH", tmp_path / "index.json")

        first = build_index_mod.build_index(full=False)
        build_index_mod.save_index(first)
        assert len(calls) == 1

        second = build_index_mod.build_index(full=False)
        assert len(calls) == 1  # unchanged size — not re-probed
        assert second["video_X"]["video_stream_seconds"] == 60.0

    def test_full_flag_reprobes_everything(self, build_index_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(build_index_mod, "RAW_VIDEO_DIR", tmp_path)
        video = tmp_path / "video_Y.mp4"
        video.write_bytes(b"y" * 500)
        monkeypatch.setattr(build_index_mod, "INDEX_PATH", tmp_path / "index.json")

        calls = []
        monkeypatch.setattr(
            build_index_mod, "probe_video_stream_duration",
            lambda p: (calls.append(p), 30.0)[1],
        )
        first = build_index_mod.build_index(full=False)
        build_index_mod.save_index(first)

        build_index_mod.build_index(full=True)
        assert len(calls) == 2  # re-probed despite unchanged size

    def test_changed_size_triggers_reprobe(self, build_index_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(build_index_mod, "RAW_VIDEO_DIR", tmp_path)
        video = tmp_path / "video_Z.mp4"
        video.write_bytes(b"z" * 100)
        monkeypatch.setattr(build_index_mod, "INDEX_PATH", tmp_path / "index.json")

        durations = iter([20.0, 25.0])
        monkeypatch.setattr(
            build_index_mod, "probe_video_stream_duration",
            lambda p: next(durations),
        )
        first = build_index_mod.build_index(full=False)
        build_index_mod.save_index(first)
        assert first["video_Z"]["video_stream_seconds"] == 20.0

        video.write_bytes(b"z" * 999)  # size changed
        second = build_index_mod.build_index(full=False)
        assert second["video_Z"]["video_stream_seconds"] == 25.0

    def test_failed_probe_is_not_cached_as_success(self, build_index_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(build_index_mod, "RAW_VIDEO_DIR", tmp_path)
        video = tmp_path / "video_bad.mp4"
        video.write_bytes(b"bad" * 10)
        monkeypatch.setattr(build_index_mod, "INDEX_PATH", tmp_path / "index.json")

        calls = []
        monkeypatch.setattr(
            build_index_mod, "probe_video_stream_duration",
            lambda p: (calls.append(p), 0.0)[1],
        )
        first = build_index_mod.build_index(full=False)
        assert first["video_bad"]["video_stream_seconds"] == 0.0

        # a 0.0 cached result must be retried next run, not skipped as "unchanged"
        build_index_mod.build_index(full=False)
        assert len(calls) == 2
