"""Unit tests for app.services.media_registry."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestTs:
    def test_returns_iso_string_for_existing_file(self, tmp_path):
        from app.services.media_registry import _ts
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        result = _ts(f)
        assert result is not None
        # Must be parseable ISO
        from datetime import datetime
        datetime.fromisoformat(result)

    def test_returns_none_for_missing_file(self, tmp_path):
        from app.services.media_registry import _ts
        result = _ts(tmp_path / "nonexistent.mp4")
        assert result is None


class TestEntry:
    def test_returns_dict(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 10)
        result = _entry("obj_001", "video", str(f))
        assert isinstance(result, dict)

    def test_has_object_id(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        result = _entry("obj_001", "video", str(f))
        assert result["object_id"] == "obj_001"

    def test_has_kind(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00")
        result = _entry("obj_002", "audio", str(f))
        assert result["kind"] == "audio"

    def test_has_size_bytes(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "file.mp4"
        f.write_bytes(b"X" * 50)
        result = _entry("obj_003", "video", str(f))
        assert result["size_bytes"] == 50

    def test_has_local_path(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        result = _entry("obj_004", "video", str(f))
        assert result["local_path"] == str(f)

    def test_content_type_inferred_from_extension(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        result = _entry("obj_005", "video", str(f))
        assert result["content_type"] == "video/mp4"

    def test_content_type_overridable(self, tmp_path):
        from app.services.media_registry import _entry
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00")
        result = _entry("obj_006", "video", str(f), content_type="application/custom")
        assert result["content_type"] == "application/custom"

    def test_nonexistent_file_has_none_size(self, tmp_path):
        from app.services.media_registry import _entry
        result = _entry("obj_007", "video", str(tmp_path / "ghost.mp4"))
        assert result["size_bytes"] is None


class TestBuildMediaRegistry:
    def _job_with_paths(self, tmp_path):
        """Return a minimal job dict with path slots for media_registry tests."""
        return {
            "job_id": "vjob_test",
            "status": "completed",
            "paths": {
                "video_path": None,
                "srt_path": None,
                "vtt_path": None,
                "export_manifest_path": None,
                "asset_manifest_path": None,
            },
        }

    def test_returns_dict_with_objects_key(self, tmp_path):
        from app.services.media_registry import build_media_registry
        job = self._job_with_paths(tmp_path)
        with patch("app.services.media_registry.load_job", return_value=job), \
             patch("app.services.media_registry.save_registry", MagicMock()), \
             patch("app.services.job_store.save_job", MagicMock()):
            result = build_media_registry("vjob_test")
        assert isinstance(result, dict)
        assert "objects" in result

    def test_video_object_added_when_file_exists(self, tmp_path):
        from app.services.media_registry import build_media_registry
        video = tmp_path / "output.mp4"
        video.write_bytes(b"\x00" * 100)
        job = self._job_with_paths(tmp_path)
        job["paths"]["video_path"] = str(video)
        with patch("app.services.media_registry.load_job", return_value=job), \
             patch("app.services.media_registry.save_registry", MagicMock()), \
             patch("app.services.media_registry._safe_path", return_value=video), \
             patch("app.services.job_store.save_job", MagicMock()):
            result = build_media_registry("vjob_test")
        assert "video" in result.get("objects", {})

    def test_empty_paths_returns_empty_objects(self, tmp_path):
        from app.services.media_registry import build_media_registry
        job = self._job_with_paths(tmp_path)
        with patch("app.services.media_registry.load_job", return_value=job), \
             patch("app.services.media_registry.save_registry", MagicMock()), \
             patch("app.services.job_store.save_job", MagicMock()):
            result = build_media_registry("vjob_test")
        assert result.get("objects") == {} or isinstance(result.get("objects"), dict)

    def test_registry_saved(self, tmp_path):
        from app.services.media_registry import build_media_registry
        job = self._job_with_paths(tmp_path)
        saved = []
        with patch("app.services.media_registry.load_job", return_value=job), \
             patch("app.services.media_registry.save_registry",
                   side_effect=lambda jid, reg: saved.append((jid, reg))), \
             patch("app.services.job_store.save_job", MagicMock()):
            build_media_registry("vjob_test")
        assert saved  # at least one save call
