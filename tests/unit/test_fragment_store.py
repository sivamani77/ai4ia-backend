"""Unit tests for app.services.fragment_store — all public functions."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_fragment


def _mock_repo(fragments: list[dict[str, Any]] | None = None) -> MagicMock:
    repo = MagicMock()
    frags = fragments or []
    repo.list_all.return_value = frags
    repo.count.return_value = len(frags)
    repo.get.side_effect = lambda fid: next(
        (dict(f) for f in frags if f.get("fragment_id") == fid), None
    )
    repo.search.side_effect = lambda **kwargs: [
        f for f in frags
        if (not kwargs.get("theme") or f.get("primary_theme") == kwargs["theme"])
        and (not kwargs.get("chapter") or f.get("primary_chapter") == kwargs["chapter"])
        and (not kwargs.get("speaker") or f.get("speaker") == kwargs["speaker"])
    ]
    repo.speakers_aggregate.return_value = [
        {"speaker": "Alice", "fragment_count": 2, "video_count": 1}
    ]
    repo.list_fragment_ids_by_source_video.side_effect = lambda source_video: [
        f["fragment_id"] for f in frags if f.get("source_video") == source_video
    ]
    repo.delete_many.side_effect = lambda fragment_ids: list(fragment_ids)
    return repo


@pytest.fixture(autouse=True)
def reset_speakers_cache():
    """Reset module-level speakers cache between tests."""
    import app.services.fragment_store as fs
    fs._speakers_cache = ([], 0.0)
    yield


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------

class TestReload:
    def test_returns_fragment_count(self):
        import app.services.fragment_store as fs
        repo = _mock_repo([make_fragment("f1"), make_fragment("f2")])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.reload()
        assert result == {"fragments": 2}

    def test_empty_store_returns_zero(self):
        import app.services.fragment_store as fs
        repo = _mock_repo([])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.reload()
        assert result == {"fragments": 0}


# ---------------------------------------------------------------------------
# all_fragments
# ---------------------------------------------------------------------------

class TestAllFragments:
    def test_returns_all(self):
        import app.services.fragment_store as fs
        frags = [make_fragment("f1"), make_fragment("f2", speaker="Bob")]
        repo = _mock_repo(frags)
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.all_fragments()
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

    def test_empty(self):
        import app.services.fragment_store as fs
        repo = _mock_repo([])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.all_fragments()
        assert result == []


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_existing_fragment(self):
        import app.services.fragment_store as fs
        frag = make_fragment("f42")
        repo = _mock_repo([frag])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.get("f42")
        assert result is not None
        assert result["fragment_id"] == "f42"

    def test_missing_fragment_returns_none(self):
        import app.services.fragment_store as fs
        repo = _mock_repo([])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.get("nonexistent")
        assert result is None

    def test_returns_dict(self):
        import app.services.fragment_store as fs
        frag = make_fragment("f1")
        repo = _mock_repo([frag])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.get("f1")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# by_facet
# ---------------------------------------------------------------------------

class TestByFacet:
    def _frags(self):
        return [
            make_fragment("f1", speaker="Alice", theme="T1", chapter="Ch 1"),
            make_fragment("f2", speaker="Bob", theme="T2", chapter="Ch 5"),
            make_fragment("f3", speaker="Alice", theme="T1", chapter="Ch 1"),
        ]

    def test_facet_theme(self):
        import app.services.fragment_store as fs
        repo = _mock_repo(self._frags())
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.by_facet("theme", "T1")
        assert len(result) == 2

    def test_facet_chapter(self):
        import app.services.fragment_store as fs
        repo = _mock_repo(self._frags())
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.by_facet("chapter", "Ch 5")
        assert len(result) == 1

    def test_facet_speaker(self):
        import app.services.fragment_store as fs
        repo = _mock_repo(self._frags())
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.by_facet("speaker", "Alice")
        assert len(result) == 2

    def test_facet_year_valid(self):
        import app.services.fragment_store as fs
        frags = [make_fragment("f1", year=2025), make_fragment("f2", year=2024)]

        mock_repo = MagicMock()
        mock_repo.search.side_effect = lambda **kwargs: [
            f for f in frags if f.get("year") == kwargs.get("year")
        ]
        with patch("app.services.fragment_store.FragmentRepository", return_value=mock_repo):
            result = fs.by_facet("year", "2025")
        assert all(f["year"] == 2025 for f in result)

    def test_facet_year_invalid_returns_empty(self):
        import app.services.fragment_store as fs
        repo = _mock_repo(self._frags())
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.by_facet("year", "not_a_year")
        assert result == []

    def test_unknown_facet_returns_empty(self):
        import app.services.fragment_store as fs
        repo = _mock_repo(self._frags())
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.by_facet("unknown_facet", "value")
        assert result == []

    def test_facet_edition(self):
        import app.services.fragment_store as fs
        frags = [make_fragment("f1", edition="AI4IA 2025")]
        mock_repo = MagicMock()
        mock_repo.search.side_effect = lambda **kwargs: [
            f for f in frags if f.get("edition") == kwargs.get("edition")
        ]
        with patch("app.services.fragment_store.FragmentRepository", return_value=mock_repo):
            result = fs.by_facet("edition", "AI4IA 2025")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# list_fragment_ids_by_source_video / delete_many
# ---------------------------------------------------------------------------

class TestDeleteBySourceVideo:
    def test_list_fragment_ids_by_source_video_returns_matching_ids(self):
        import app.services.fragment_store as fs
        frags = [
            make_fragment("v1_0001", source_video="v1"),
            make_fragment("v1_0002", source_video="v1"),
            make_fragment("v2_0001", source_video="v2"),
        ]
        repo = _mock_repo(frags)
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.list_fragment_ids_by_source_video("v1")
        assert result == ["v1_0001", "v1_0002"]

    def test_list_fragment_ids_by_source_video_no_match_returns_empty(self):
        import app.services.fragment_store as fs
        repo = _mock_repo([make_fragment("v1_0001", source_video="v1")])
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.list_fragment_ids_by_source_video("nonexistent")
        assert result == []

    def test_delete_many_returns_deleted_ids(self):
        import app.services.fragment_store as fs
        repo = _mock_repo()
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.delete_many(["v1_0001", "v1_0002"])
        assert result == ["v1_0001", "v1_0002"]
        repo.delete_many.assert_called_once_with(["v1_0001", "v1_0002"])

    def test_delete_many_empty_list(self):
        import app.services.fragment_store as fs
        repo = _mock_repo()
        repo.delete_many.side_effect = lambda ids: []
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.delete_many([])
        assert result == []


# ---------------------------------------------------------------------------
# speakers_summary — with TTL caching
# ---------------------------------------------------------------------------

class TestSpeakersSummary:
    def test_returns_list(self):
        import app.services.fragment_store as fs
        repo = _mock_repo()
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.speakers_summary()
        assert isinstance(result, list)

    def test_cached_within_ttl(self):
        import app.services.fragment_store as fs
        repo = _mock_repo()
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            fs.speakers_summary()
            fs.speakers_summary()
        # Second call should use cache — repo only called once
        repo.speakers_aggregate.assert_called_once()

    def test_cache_refreshed_after_ttl(self):
        import time
        import app.services.fragment_store as fs

        repo = _mock_repo()
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            # Seed cache with old timestamp
            fs._speakers_cache = ([{"speaker": "Old"}], time.monotonic() - 999)
            fs.speakers_summary()  # should refresh
        repo.speakers_aggregate.assert_called_once()

    def test_empty_cache_triggers_fetch(self):
        import app.services.fragment_store as fs
        repo = _mock_repo()
        with patch("app.services.fragment_store.FragmentRepository", return_value=repo):
            result = fs.speakers_summary()
        repo.speakers_aggregate.assert_called_once()
        assert isinstance(result, list)
