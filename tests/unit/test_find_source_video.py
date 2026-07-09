"""Tests for 09_export_video.py's find_source_video Unicode-punctuation fallback.

Companion fix to stage_09_export._video_id_prefix: once stage_09_export.py has
downloaded a source video to the local RAW_VIDEO_DIR under its REAL S3 filename
(with original fullwidth Unicode punctuation), the render script's own
find_source_video() must still be able to locate it even though the timeline
item's source_video/fragment_id string may have had that punctuation
normalized somewhere upstream (an LLM round-trip in storyline generation or
narrative assembly). Real failure: '(6N)：...XYNOGENY？' in the S3 filename vs
'(6N)_:...XYNOGENY?' in the manifest.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))


@pytest.fixture(scope="module")
def export_video():
    path = PROJECT_ROOT / "scripts" / "09_export_video.py"
    spec = importlib.util.spec_from_file_location("export_video_find_src", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVideoIdPrefix:
    def test_extracts_stable_prefix(self, export_video):
        assert export_video._video_id_prefix("0254_RXXzSGoo0KQ_(6N)： C Nelson") == "0254_RXXzSGoo0KQ_"

    def test_no_match_returns_none(self, export_video):
        assert export_video._video_id_prefix("no numeric prefix here") is None


class TestFindSourceVideo:
    def test_exact_match_still_works(self, export_video, tmp_path, monkeypatch):
        monkeypatch.setattr(export_video, "RAW_VIDEO_DIR", tmp_path)
        (tmp_path / "0100_abc_Some Title.mp4").write_bytes(b"x")
        result = export_video.find_source_video("0100_abc_Some Title")
        assert result == tmp_path / "0100_abc_Some Title.mp4"

    def test_id_prefix_fallback_survives_punctuation_drift(self, export_video, tmp_path, monkeypatch):
        monkeypatch.setattr(export_video, "RAW_VIDEO_DIR", tmp_path)
        real_file = tmp_path / "0254_RXXzSGoo0KQ_(6N)： C Nelson - XYNOGENY？ How We Share Decision-Making.mp4"
        real_file.write_bytes(b"x")

        requested = "0254_RXXzSGoo0KQ_(6N)_: C Nelson - XYNOGENY? How We Share Decision-Making"
        result = export_video.find_source_video(requested)
        assert result == real_file

    def test_fragment_id_prefix_fallback(self, export_video, tmp_path, monkeypatch):
        monkeypatch.setattr(export_video, "RAW_VIDEO_DIR", tmp_path)
        real_file = tmp_path / "0254_RXXzSGoo0KQ_(6N)： C Nelson.mp4"
        real_file.write_bytes(b"x")

        result = export_video.find_source_video(
            source_video=None,
            fragment_id="0254_RXXzSGoo0KQ_(6N)_: C Nelson_0007",
        )
        assert result == real_file

    def test_no_match_anywhere_returns_none(self, export_video, tmp_path, monkeypatch):
        monkeypatch.setattr(export_video, "RAW_VIDEO_DIR", tmp_path)
        assert export_video.find_source_video("0999_missing_Title") is None

    def test_source_video_preferred_over_fragment_id(self, export_video, tmp_path, monkeypatch):
        monkeypatch.setattr(export_video, "RAW_VIDEO_DIR", tmp_path)
        (tmp_path / "0100_abc_Correct.mp4").write_bytes(b"x")
        (tmp_path / "0200_xyz_Wrong.mp4").write_bytes(b"x")
        result = export_video.find_source_video(
            source_video="0100_abc_Correct",
            fragment_id="0200_xyz_Wrong_0001",
        )
        assert result == tmp_path / "0100_abc_Correct.mp4"
