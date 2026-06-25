"""Unit tests for app.services.archive_service — pure-logic helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_fragment


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def _normalize(self, value):
        from app.services.archive_service import normalize_text
        return normalize_text(value)

    def test_none_returns_empty_string(self):
        assert self._normalize(None) == ""

    def test_string_unchanged(self):
        assert self._normalize("hello") == "hello"

    def test_list_joined_with_space(self):
        assert self._normalize(["a", "b", "c"]) == "a b c"

    def test_integer_stringified(self):
        assert self._normalize(42) == "42"

    def test_empty_list_gives_empty_string(self):
        assert self._normalize([]) == ""

    def test_list_with_mixed_types(self):
        result = self._normalize([1, "two", 3.0])
        assert result == "1 two 3.0"


# ---------------------------------------------------------------------------
# infer_speaker_from_filename
# ---------------------------------------------------------------------------

class TestInferSpeakerFromFilename:
    def _infer(self, name):
        from app.services.archive_service import infer_speaker_from_filename
        return infer_speaker_from_filename(name)

    def test_standard_format(self):
        result = self._infer("0139_abc123_John_Doe.mp4")
        assert result == "John Doe"

    def test_hyphenated_name(self):
        result = self._infer("0001_xyz_Jane_-_Smith.mp4")
        # underscores replaced with spaces, _-_ becomes " - "
        assert "Jane" in result

    def test_classified_suffix_removed(self):
        result = self._infer("0001_abc_Alice.classified")
        assert "classified" not in result.lower()

    def test_short_stem_fallback(self):
        result = self._infer("abc.mp4")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extra_whitespace_stripped(self):
        result = self._infer("0001_abc_Alice_Bob.mp4")
        assert "  " not in result

    def test_multipart_name(self):
        result = self._infer("0001_abc_Firstname_Lastname_Title.mp4")
        assert "Firstname" in result


# ---------------------------------------------------------------------------
# safe_read_json
# ---------------------------------------------------------------------------

class TestSafeReadJson:
    def test_valid_json(self, tmp_path):
        from app.services.archive_service import safe_read_json
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        assert safe_read_json(f) == {"key": "value"}

    def test_invalid_json_returns_none(self, tmp_path):
        from app.services.archive_service import safe_read_json
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        assert safe_read_json(f) is None

    def test_missing_file_returns_none(self, tmp_path):
        from app.services.archive_service import safe_read_json
        assert safe_read_json(tmp_path / "nonexistent.json") is None


# ---------------------------------------------------------------------------
# extract_tag_name
# ---------------------------------------------------------------------------

class TestExtractTagName:
    def _extract(self, tag, key_name="theme"):
        from app.services.archive_service import extract_tag_name
        return extract_tag_name(tag, key_name)

    def test_none_returns_none(self):
        assert self._extract(None) is None

    def test_string_tag(self):
        assert self._extract("AI Ethics") == "AI Ethics"

    def test_dict_with_key_name(self):
        assert self._extract({"theme": "T1"}, "theme") == "T1"

    def test_dict_with_name_field(self):
        assert self._extract({"name": "AI Ethics"}, "chapter") == "AI Ethics"

    def test_dict_fallback_to_theme(self):
        assert self._extract({"theme": "T1"}, "chapter") == "T1"

    def test_non_dict_non_string_stringified(self):
        result = self._extract(123, "theme")
        assert result == "123"


# ---------------------------------------------------------------------------
# search_fragments — scoring and filtering logic
# ---------------------------------------------------------------------------

class TestSearchFragments:
    def _frags(self):
        return [
            make_fragment("f1", speaker="Alice", theme="T1", chapter="Ch 1",
                          text="AI ethics is fundamental"),
            make_fragment("f2", speaker="Bob", theme="T2", chapter="Ch 5",
                          text="Information access matters"),
            make_fragment("f3", speaker="Alice", theme="T1", chapter="Ch 2",
                          text="AI governance structures"),
        ]

    def _mock_repo(self, frags):
        repo = MagicMock()
        repo.search.return_value = frags
        return repo

    def test_returns_list(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments(query="AI")
        assert isinstance(result, list)

    def test_query_scored_higher(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments(query="ethics")
        # f1 contains "ethics", should rank first
        assert result[0]["fragment_id"] == "f1"

    def test_limit_respected(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments(query="AI", limit=2)
        assert len(result) <= 2

    def test_limit_minimum_one(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments(limit=0)
        assert len(result) >= 1

    def test_limit_capped_at_200(self):
        from app.services.archive_service import search_fragments
        frags = [make_fragment(f"f{i}") for i in range(250)]
        repo = self._mock_repo(frags)
        with patch("app.services.archive_service.FragmentRepository", return_value=repo):
            search_fragments(limit=300)
        # Repo must be called with limit ≤ 200
        call_kwargs = repo.search.call_args[1]
        assert call_kwargs.get("limit", 200) <= 200

    def test_search_score_field_added(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments(query="AI")
        assert all("_search_score" in f for f in result)

    def test_clip_usage_filter_applied(self):
        from app.services.archive_service import search_fragments
        frags = [
            {**make_fragment("f1"), "clip_usage_candidates": ["highlight"]},
            {**make_fragment("f2"), "clip_usage_candidates": ["background"]},
        ]
        with patch("app.services.archive_service.FragmentRepository",
                   return_value=self._mock_repo(frags)):
            result = search_fragments(clip_usage="highlight")
        assert all("highlight" in str(f.get("clip_usage_candidates", [])).lower() for f in result)

    def test_empty_query_returns_all(self):
        from app.services.archive_service import search_fragments
        frags = self._frags()
        with patch("app.services.archive_service.FragmentRepository", return_value=self._mock_repo(frags)):
            result = search_fragments()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_themes_and_chapters — structure and caching
# ---------------------------------------------------------------------------

class TestGetThemesAndChapters:
    def _mock_repo_with_counts(self, counts: dict[str, int]):
        repo = MagicMock()
        repo.chapter_counts.return_value = counts
        return repo

    def _reset_cache(self):
        import app.services.archive_service as svc
        svc._themes_cache = ({}, 0.0)

    def test_returns_themes_key(self):
        from app.services.archive_service import get_themes_and_chapters
        self._reset_cache()
        with patch("app.services.archive_service.FragmentRepository",
                   return_value=self._mock_repo_with_counts({})):
            result = get_themes_and_chapters()
        assert "themes" in result

    def test_returns_correct_totals(self):
        from app.services.archive_service import get_themes_and_chapters
        self._reset_cache()
        with patch("app.services.archive_service.FragmentRepository",
                   return_value=self._mock_repo_with_counts({})):
            result = get_themes_and_chapters()
        assert result["total_themes"] == 7
        assert result["total_chapters"] == 25

    def test_fragment_counts_aggregated(self):
        from app.services.archive_service import get_themes_and_chapters
        self._reset_cache()
        counts = {"Ch 1: Foundations of AI Ethics & Philosophy": 5}
        with patch("app.services.archive_service.FragmentRepository",
                   return_value=self._mock_repo_with_counts(counts)):
            result = get_themes_and_chapters()

        # Find T1 theme
        t1 = next(t for t in result["themes"] if t["theme_id"] == "T1")
        ch1 = next(c for c in t1["chapters"] if c["chapter_id"] == "Ch 1")
        assert ch1["fragment_count"] == 5
        assert t1["fragment_count"] >= 5

    def test_cached_within_ttl(self):
        from app.services.archive_service import get_themes_and_chapters
        self._reset_cache()
        repo = self._mock_repo_with_counts({})
        with patch("app.services.archive_service.FragmentRepository", return_value=repo):
            get_themes_and_chapters()
            get_themes_and_chapters()
        repo.chapter_counts.assert_called_once()

    def test_cache_refreshed_after_ttl(self):
        import time
        from app.services.archive_service import get_themes_and_chapters
        import app.services.archive_service as svc

        # Seed with expired cache
        svc._themes_cache = ({"themes": [], "total_themes": 0, "total_chapters": 0},
                              time.monotonic() - 999)
        repo = self._mock_repo_with_counts({})
        with patch("app.services.archive_service.FragmentRepository", return_value=repo):
            get_themes_and_chapters()
        repo.chapter_counts.assert_called_once()

    def test_each_theme_has_chapters_list(self):
        from app.services.archive_service import get_themes_and_chapters
        self._reset_cache()
        with patch("app.services.archive_service.FragmentRepository",
                   return_value=self._mock_repo_with_counts({})):
            result = get_themes_and_chapters()
        for theme in result["themes"]:
            assert "chapters" in theme
            assert isinstance(theme["chapters"], list)
