"""Regression test: archive_visual / speaker_visual / generic segments whose
requested window starts past the source video's real stream length must have
their window shifted onto real footage, not seek past it and produce an
empty/zero-duration file (ffprobe reports N/A, then get_media_duration raises
and aborts the whole export).

Root cause: 0082_5aoew3DYz7Y's video stream is only ~295s but its audio stream
is ~5249s (source recording where video cut out early). A timeline item asking
for start=1358.4s is valid against the audio-derived duration but has no video
frames there at all.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.pipeline.export_gen.segment_pipeline as segment_pipeline_module  # noqa: E402


@pytest.fixture(scope="module")
def export_video():
    return segment_pipeline_module


class TestArchiveVisualWindowGuard:
    def test_window_shifted_onto_real_footage(self, export_video, tmp_path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")

        item = {
            "type": "archive_clip",
            "segment_type": "archive_visual",
            "audio_mode": "silent",
            "muted": True,
            "fragment_id": "0082_5aoew3DYz7Y_0010",
            "source_video": "0082_5aoew3DYz7Y",
            "start": 1358.4,
            "end": 1377.786,
        }

        captured_calls = []

        def fake_render_archive_clip(**kwargs):
            captured_calls.append(kwargs)
            kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["output_path"].write_bytes(b"fake rendered clip")

        with patch.object(export_video, "find_source_video", return_value=source), \
             patch.object(export_video, "get_video_stream_duration", return_value=295.2), \
             patch.object(export_video, "render_archive_clip", side_effect=fake_render_archive_clip), \
             patch.object(export_video, "get_media_duration", return_value=19.4):
            idx, path, info = export_video._render_one_segment(1, item, tmp_path / "out")

        assert path is not None and path.exists()
        assert info["duration"] == 19.4
        assert len(captured_calls) == 1
        call = captured_calls[0]
        # window must be fully inside the real video stream (295.2s), not past it
        assert call["end"] <= 295.2
        assert call["start"] < call["end"]
        # span (duration) preserved even though the window moved
        assert pytest.approx(call["end"] - call["start"], abs=0.01) == (1377.786 - 1358.4)

    def test_window_within_stream_is_untouched(self, export_video, tmp_path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fake video")

        item = {
            "type": "archive_clip",
            "segment_type": "archive_visual",
            "audio_mode": "silent",
            "muted": True,
            "fragment_id": "f1",
            "source_video": "v1",
            "start": 10.0,
            "end": 30.0,
        }

        captured_calls = []

        def fake_render_archive_clip(**kwargs):
            captured_calls.append(kwargs)
            kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
            kwargs["output_path"].write_bytes(b"fake rendered clip")

        with patch.object(export_video, "find_source_video", return_value=source), \
             patch.object(export_video, "get_video_stream_duration", return_value=295.2), \
             patch.object(export_video, "render_archive_clip", side_effect=fake_render_archive_clip), \
             patch.object(export_video, "get_media_duration", return_value=20.0):
            export_video._render_one_segment(1, item, tmp_path / "out")

        assert captured_calls[0]["start"] == 10.0
        assert captured_calls[0]["end"] == 30.0
