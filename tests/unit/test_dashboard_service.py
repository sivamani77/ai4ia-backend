"""Unit tests for app.services.dashboard_service."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCountFiles:
    def test_nonexistent_dir_returns_zero(self, tmp_path):
        from app.services.dashboard_service import count_files
        assert count_files(tmp_path / "nope", "*.mp4") == 0

    def test_counts_matching_files(self, tmp_path):
        from app.services.dashboard_service import count_files
        (tmp_path / "a.mp4").write_text("")
        (tmp_path / "b.mp4").write_text("")
        (tmp_path / "c.wav").write_text("")
        assert count_files(tmp_path, "*.mp4") == 2

    def test_counts_all_with_wildcard(self, tmp_path):
        from app.services.dashboard_service import count_files
        for name in ("a.mp4", "b.wav", "c.json"):
            (tmp_path / name).write_text("")
        assert count_files(tmp_path, "*") == 3

    def test_empty_dir_returns_zero(self, tmp_path):
        from app.services.dashboard_service import count_files
        assert count_files(tmp_path, "*.mp4") == 0

    def test_pattern_with_glob(self, tmp_path):
        from app.services.dashboard_service import count_files
        (tmp_path / "storyline_20260101.json").write_text("")
        (tmp_path / "other.json").write_text("")
        assert count_files(tmp_path, "storyline_*.json") == 1


class TestLatestFile:
    def test_returns_none_when_dir_missing(self, tmp_path):
        from app.services.dashboard_service import latest_file
        assert latest_file(tmp_path / "missing", "*.json") is None

    def test_returns_none_when_no_match(self, tmp_path):
        from app.services.dashboard_service import latest_file
        assert latest_file(tmp_path, "*.json") is None

    def test_returns_latest_by_sort(self, tmp_path):
        from app.services.dashboard_service import latest_file
        (tmp_path / "storyline_20260101.json").write_text("")
        (tmp_path / "storyline_20260102.json").write_text("")
        result = latest_file(tmp_path, "storyline_*.json")
        assert result == "storyline_20260102.json"

    def test_returns_filename_not_path(self, tmp_path):
        from app.services.dashboard_service import latest_file
        (tmp_path / "file.json").write_text("")
        result = latest_file(tmp_path, "*.json")
        assert result == "file.json"
        assert "/" not in result


class TestCountClassifiedFragments:
    def test_returns_count_from_repo(self):
        from app.services.dashboard_service import count_classified_fragments
        mock_repo = MagicMock()
        mock_repo.return_value.count.return_value = 42
        with patch("app.db.repositories.fragment_repository.FragmentRepository", mock_repo):
            assert count_classified_fragments() == 42

    def test_returns_zero_on_exception(self):
        from app.services.dashboard_service import count_classified_fragments
        with patch("app.db.repositories.fragment_repository.FragmentRepository",
                   side_effect=Exception("DB down")):
            assert count_classified_fragments() == 0


class TestGetDashboard:
    def test_returns_dict_with_expected_sections(self, tmp_path):
        from app.services.dashboard_service import get_dashboard
        with patch("app.services.dashboard_service.RAW_VIDEO_DIR", tmp_path), \
             patch("app.services.dashboard_service.AUDIO_DIR", tmp_path), \
             patch("app.services.dashboard_service.TRANSCRIPTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.CLASSIFIED_FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EMBEDDINGS_DIR", tmp_path), \
             patch("app.services.dashboard_service.STORYLINES_DIR", tmp_path), \
             patch("app.services.dashboard_service.GENERATED_ASSETS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EXPORTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.LATEST_ASSET_MANIFEST", tmp_path / "manifest.json"), \
             patch("app.db.repositories.fragment_repository.FragmentRepository",
                   return_value=MagicMock(count=MagicMock(return_value=0))):
            result = get_dashboard()

        assert "archive" in result
        assert "latest" in result
        assert "models" in result
        assert "paths" in result

    def test_archive_section_has_required_keys(self, tmp_path):
        from app.services.dashboard_service import get_dashboard
        with patch("app.services.dashboard_service.RAW_VIDEO_DIR", tmp_path), \
             patch("app.services.dashboard_service.AUDIO_DIR", tmp_path), \
             patch("app.services.dashboard_service.TRANSCRIPTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.CLASSIFIED_FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EMBEDDINGS_DIR", tmp_path), \
             patch("app.services.dashboard_service.STORYLINES_DIR", tmp_path), \
             patch("app.services.dashboard_service.GENERATED_ASSETS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EXPORTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.LATEST_ASSET_MANIFEST", tmp_path / "m.json"), \
             patch("app.db.repositories.fragment_repository.FragmentRepository",
                   return_value=MagicMock(count=MagicMock(return_value=5))):
            result = get_dashboard()

        archive = result["archive"]
        for key in ("raw_videos", "audio_files", "classified_fragments", "embeddings_ready"):
            assert key in archive

    def test_models_section_values(self, tmp_path):
        from app.services.dashboard_service import get_dashboard
        with patch("app.services.dashboard_service.RAW_VIDEO_DIR", tmp_path), \
             patch("app.services.dashboard_service.AUDIO_DIR", tmp_path), \
             patch("app.services.dashboard_service.TRANSCRIPTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.CLASSIFIED_FRAGMENTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EMBEDDINGS_DIR", tmp_path), \
             patch("app.services.dashboard_service.STORYLINES_DIR", tmp_path), \
             patch("app.services.dashboard_service.GENERATED_ASSETS_DIR", tmp_path), \
             patch("app.services.dashboard_service.EXPORTS_DIR", tmp_path), \
             patch("app.services.dashboard_service.LATEST_ASSET_MANIFEST", tmp_path / "m.json"), \
             patch("app.db.repositories.fragment_repository.FragmentRepository",
                   return_value=MagicMock(count=MagicMock(return_value=0))):
            result = get_dashboard()

        models = result["models"]
        assert "embedding" in models
        assert "strong_llm" in models
