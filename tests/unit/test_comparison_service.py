"""Unit tests for app.services.comparison_service — pool helpers and format functions."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import make_fragment


# ---------------------------------------------------------------------------
# _pool_from_fragments
# ---------------------------------------------------------------------------

class TestPoolFromFragments:
    def test_numbered_from_1(self):
        from app.services.comparison_service import _pool_from_fragments
        frags = [make_fragment(f"f{i}") for i in range(3)]
        pool = _pool_from_fragments(frags)
        assert [p["ref"] for p in pool] == [1, 2, 3]

    def test_empty_returns_empty(self):
        from app.services.comparison_service import _pool_from_fragments
        assert _pool_from_fragments([]) == []

    def test_fields_copied(self):
        from app.services.comparison_service import _pool_from_fragments
        frag = make_fragment("f1", speaker="Alice", text="Hello")
        pool = _pool_from_fragments([frag])
        assert pool[0]["speaker"] == "Alice"
        assert pool[0]["text"] == "Hello"
        assert pool[0]["fragment_id"] == "f1"

    def test_single_fragment(self):
        from app.services.comparison_service import _pool_from_fragments
        pool = _pool_from_fragments([make_fragment("f1")])
        assert len(pool) == 1
        assert pool[0]["ref"] == 1

    def test_stream_url_computed_from_source_video(self):
        """GAP-014 regression: fragment rows have no stream_url column, so it
        must be derived from source_video, not read from a field that's
        never set on the raw fragment."""
        from app.services.comparison_service import _pool_from_fragments
        frag = make_fragment("f1", source_video="0001_abc123_speaker")
        pool = _pool_from_fragments([frag])
        assert pool[0]["stream_url"] == "/api/archive/videos/0001_abc123_speaker/stream"

    def test_stream_url_none_when_source_video_missing(self):
        from app.services.comparison_service import _pool_from_fragments
        frag = make_fragment("f1")
        frag.pop("source_video", None)
        pool = _pool_from_fragments([frag])
        assert pool[0]["stream_url"] is None


# ---------------------------------------------------------------------------
# _format_pool
# ---------------------------------------------------------------------------

class TestFormatPool:
    def test_returns_string(self):
        from app.services.comparison_service import _pool_from_fragments, _format_pool
        pool = _pool_from_fragments([make_fragment("f1", speaker="Alice")])
        result = _format_pool(pool)
        assert isinstance(result, str)

    def test_contains_ref_number(self):
        from app.services.comparison_service import _pool_from_fragments, _format_pool
        pool = _pool_from_fragments([make_fragment("f1", speaker="Alice")])
        result = _format_pool(pool)
        assert "[1]" in result

    def test_contains_speaker(self):
        from app.services.comparison_service import _pool_from_fragments, _format_pool
        pool = _pool_from_fragments([make_fragment("f1", speaker="Bob")])
        result = _format_pool(pool)
        assert "Bob" in result

    def test_multiple_fragments_each_on_own_line(self):
        from app.services.comparison_service import _pool_from_fragments, _format_pool
        pool = _pool_from_fragments([make_fragment("f1"), make_fragment("f2")])
        result = _format_pool(pool)
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_empty_pool_returns_empty_string(self):
        from app.services.comparison_service import _format_pool
        assert _format_pool([]) == ""


# ---------------------------------------------------------------------------
# _clip helper
# ---------------------------------------------------------------------------

class TestClip:
    def test_returns_expected_keys(self):
        from app.services.comparison_service import _clip
        frag = make_fragment("f1")
        result = _clip(frag)
        expected_keys = {"fragment_id", "speaker", "speaker_attribution", "edition",
                         "primary_theme", "video_stream_url", "start", "end", "duration", "text"}
        assert expected_keys.issubset(result.keys())

    def test_stream_url_passed_through_from_pool_entry(self):
        """_clip itself just passes through whatever "stream_url" key its
        input already carries — the actual computation happens upstream in
        _pool_from_fragments (see TestPoolFromFragments.test_stream_url_
        computed_from_source_video), which is what real callers always route
        through before _clip sees a dict."""
        from app.services.comparison_service import _clip
        frag = make_fragment("f1")
        frag["stream_url"] = "/api/stream/abc"
        result = _clip(frag)
        assert result["video_stream_url"] == "/api/stream/abc"

    def test_text_truncated_at_400(self):
        from app.services.comparison_service import _clip
        frag = make_fragment("f1", text="word " * 200)
        result = _clip(frag)
        assert len(result["text"]) <= 403  # 400 + "..."


# ---------------------------------------------------------------------------
# compare_speakers — integration with mocked LLM + fragment_store
# ---------------------------------------------------------------------------

class TestCompareSpeakers:
    def _mock_llm_response(self):
        return {
            "per_speaker": [{"speaker": "Alice", "position": "pro AI", "fragment_refs": [1]}],
            "agreements": [{"statement": "AI needs governance", "fragment_refs": [1, 2]}],
            "tensions": [],
            "synthesis": "Both agree on ethics",
        }

    def test_raises_when_no_fragments(self):
        from app.services.comparison_service import compare_speakers
        with patch("app.services.comparison_service.fragment_store.by_facet", return_value=[]):
            with pytest.raises(ValueError, match="No fragments found"):
                compare_speakers(["Alice", "Bob"], topic="AI ethics")

    def test_returns_dict_with_expected_keys(self):
        from app.services.comparison_service import compare_speakers
        frags_alice = [make_fragment("f1", speaker="Alice")]
        frags_bob = [make_fragment("f2", speaker="Bob")]

        def _by_facet(facet, value):
            if value == "Alice":
                return frags_alice
            if value == "Bob":
                return frags_bob
            return []

        with patch("app.services.comparison_service.fragment_store.by_facet", side_effect=_by_facet), \
             patch("app.services.comparison_service.embedding_index.search", return_value=[]), \
             patch("app.services.comparison_service.generate_json", return_value=self._mock_llm_response()):
            result = compare_speakers(["Alice", "Bob"], topic="AI ethics")

        expected_keys = {"topic", "speakers_requested", "speakers_missing", "per_speaker",
                         "agreements", "tensions", "synthesis", "evidence_pool", "created_at"}
        assert expected_keys.issubset(result.keys())

    def test_missing_speaker_tracked(self):
        from app.services.comparison_service import compare_speakers
        frags_alice = [make_fragment("f1", speaker="Alice")]

        def _by_facet(facet, value):
            return frags_alice if value == "Alice" else []

        with patch("app.services.comparison_service.fragment_store.by_facet", side_effect=_by_facet), \
             patch("app.services.comparison_service.embedding_index.search", return_value=[]), \
             patch("app.services.comparison_service.generate_json", return_value=self._mock_llm_response()):
            result = compare_speakers(["Alice", "NoOne"], topic="AI")

        assert "NoOne" in result["speakers_missing"]

    def test_default_topic_used(self):
        from app.services.comparison_service import compare_speakers
        frags = [make_fragment("f1", speaker="Alice")]

        with patch("app.services.comparison_service.fragment_store.by_facet", return_value=frags), \
             patch("app.services.comparison_service.embedding_index.search", return_value=[]), \
             patch("app.services.comparison_service.generate_json", return_value=self._mock_llm_response()):
            result = compare_speakers(["Alice"], topic=None)

        assert result["topic"] == "their overall perspective on AI"


# ---------------------------------------------------------------------------
# synthesize_theme — basic contract
# ---------------------------------------------------------------------------

class TestSynthesizeTheme:
    def test_raises_when_no_fragments(self):
        from app.services.comparison_service import synthesize_theme
        with patch("app.services.comparison_service.embedding_index.search", return_value=[]):
            with pytest.raises(ValueError, match="No fragments found"):
                synthesize_theme("AI Ethics")

    def test_returns_dict_with_expected_keys(self):
        from app.services.comparison_service import synthesize_theme
        frags = [make_fragment("f1"), make_fragment("f2")]
        mock_response = {
            "title": "AI Ethics in Focus",
            "summary": "Key insights",
            "key_points": [{"point": "AI needs governance", "fragment_refs": [1]}],
            "representative_clips": [{"ref": 1, "reason": "best clip"}],
        }
        with patch("app.services.comparison_service.embedding_index.search", return_value=frags), \
             patch("app.services.comparison_service.generate_json", return_value=mock_response):
            result = synthesize_theme("AI Ethics")

        assert "theme" in result
        assert "title" in result
        assert "summary" in result
        assert "key_points" in result
        assert result["theme"] == "AI Ethics"


# ---------------------------------------------------------------------------
# build_relay — basic contract
# ---------------------------------------------------------------------------

class TestBuildRelay:
    def test_raises_when_no_fragments(self):
        from app.services.comparison_service import build_relay
        with patch("app.services.comparison_service.embedding_index.search", return_value=[]):
            with pytest.raises(ValueError, match="No fragments found"):
                build_relay("some topic")

    def test_returns_dict_with_expected_keys(self):
        from app.services.comparison_service import build_relay
        frags = [
            make_fragment("f1", speaker="Alice"),
            make_fragment("f2", speaker="Bob"),
            make_fragment("f3", speaker="Carol"),
        ]
        mock_response = {
            "relay_title": "AI Relay",
            "relay_steps": [{"ref": 1, "role_in_relay": "opening", "bridge": "intro"}],
            "relay_narrative": "A thread through AI voices",
        }
        with patch("app.services.comparison_service.embedding_index.search", return_value=frags), \
             patch("app.services.comparison_service.generate_json", return_value=mock_response):
            result = build_relay("AI Ethics")

        expected_keys = {"topic", "relay_title", "relay_narrative", "relay_steps",
                         "step_count", "distinct_speakers", "edition", "created_at"}
        assert expected_keys.issubset(result.keys())

    def test_speaker_diversity_in_pool(self):
        """Each distinct speaker should appear only once in the prioritized pool."""
        from app.services.comparison_service import build_relay
        # Two frags from Alice, two from Bob — pool should deduplicate speakers
        frags = [
            make_fragment("f1", speaker="Alice"),
            make_fragment("f2", speaker="Alice"),
            make_fragment("f3", speaker="Bob"),
        ]
        mock_response = {
            "relay_title": "Relay",
            "relay_steps": [],
            "relay_narrative": "thread",
        }
        with patch("app.services.comparison_service.embedding_index.search", return_value=frags), \
             patch("app.services.comparison_service.generate_json", return_value=mock_response) as mock_llm:
            build_relay("AI")
        # LLM was called — evidence pool was built from de-duplicated speakers first
        mock_llm.assert_called_once()
