"""Unit tests for app.services.embedding_index — ChromaDB-based EmbeddingIndex."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from tests.conftest import make_fragment


# ---------------------------------------------------------------------------
# _build_where
# ---------------------------------------------------------------------------

class TestBuildWhere:
    def _build(self, filters):
        from app.services.embedding_index import _build_where
        return _build_where(filters)

    def test_empty_filters_returns_none(self):
        assert self._build({}) is None

    def test_single_theme_filter(self):
        result = self._build({"theme": "T1"})
        # Chroma metadata field is "primary_theme" (verbatim from
        # fragments_classified), the caller-facing key stays "theme".
        assert result == {"primary_theme": {"$eq": "T1"}}

    def test_single_chapter_filter(self):
        result = self._build({"chapter": "Ch 5"})
        assert result == {"primary_chapter": {"$eq": "Ch 5"}}

    def test_single_audio_quality_filter(self):
        result = self._build({"audio_quality_status": "good"})
        assert result == {"audio_quality_status": {"$eq": "good"}}

    def test_multiple_filters_uses_and(self):
        result = self._build({"theme": "T1", "chapter": "Ch 1"})
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_unknown_filter_key_ignored(self):
        result = self._build({"not_a_real_filter": "Alice"})
        assert result is None

    def test_speaker_filter(self):
        # speaker is a plain Chroma metadata field (written verbatim by
        # import_embeddings_to_chroma.py) — confirmed live: a semantic
        # search filtered by speaker was silently ignoring it and
        # returning unrelated top-K matches until this was added.
        result = self._build({"speaker": "Ashwini Kotaru"})
        assert result == {"speaker": {"$eq": "Ashwini Kotaru"}}

    def test_none_value_ignored(self):
        result = self._build({"theme": None})
        assert result is None

    def test_empty_string_value_ignored(self):
        result = self._build({"theme": ""})
        assert result is None

    def test_combined_valid_and_invalid_keys(self):
        result = self._build({"theme": "T1", "not_a_real_filter": "Alice"})
        # "speaker" is now a valid Chroma metadata key too (see field_map),
        # so use a genuinely unknown key to keep this a single-clause case.
        assert result == {"primary_theme": {"$eq": "T1"}}


# ---------------------------------------------------------------------------
# EmbeddingIndex.load
# ---------------------------------------------------------------------------

class TestEmbeddingIndexLoad:
    def _make_index(self):
        from app.services.embedding_index import EmbeddingIndex
        return EmbeddingIndex()

    def test_load_sets_loaded_true(self):
        idx = self._make_index()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            result = idx.load()

        assert idx._loaded is True
        assert result == {"fragments": 10}

    def test_load_handles_collection_unavailable(self):
        idx = self._make_index()
        with patch("app.services.embedding_index._make_client", side_effect=Exception("no chroma")):
            result = idx.load()

        assert idx._loaded is True
        assert idx._collection is None
        assert result == {"fragments": 0}

    def test_load_returns_count(self):
        idx = self._make_index()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            result = idx.load()

        assert result["fragments"] == 42


# ---------------------------------------------------------------------------
# EmbeddingIndex.available
# ---------------------------------------------------------------------------

class TestEmbeddingIndexAvailable:
    def test_available_true_when_collection_has_data(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            assert idx.available() is True

    def test_available_false_when_collection_none(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        with patch("app.services.embedding_index._make_client", side_effect=Exception("no chroma")):
            assert idx.available() is False

    def test_available_false_when_collection_empty(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index._make_client", return_value=mock_client):
            assert idx.available() is False

    def test_available_triggers_load_if_not_loaded(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        assert idx._loaded is False
        with patch("app.services.embedding_index._make_client", side_effect=Exception("no chroma")):
            idx.available()
        assert idx._loaded is True


# ---------------------------------------------------------------------------
# EmbeddingIndex.reload
# ---------------------------------------------------------------------------

class TestEmbeddingIndexReload:
    def test_reload_clears_state(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        idx._loaded = True
        idx._collection = MagicMock()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 7
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            result = idx.reload()

        assert idx._loaded is True
        assert result == {"fragments": 7}

    def test_reload_returns_new_count(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 99
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            result = idx.reload()

        assert result["fragments"] == 99


# ---------------------------------------------------------------------------
# EmbeddingIndex.search
# ---------------------------------------------------------------------------

class TestEmbeddingIndexSearch:
    def _make_loaded_index(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10

        # Simulate a query result
        mock_collection.query.return_value = {
            "ids": [["frag_0", "frag_1", "frag_2"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[
                {"fragment_id": "frag_0", "primary_theme": "T1"},
                {"fragment_id": "frag_1", "primary_theme": "T2"},
                {"fragment_id": "frag_2", "primary_theme": "T1"},
            ]],
        }
        idx._collection = mock_collection
        idx._loaded = True
        return idx, mock_collection

    def test_search_returns_list(self):
        idx, _ = self._make_loaded_index()
        frags = {f"frag_{i}": make_fragment(f"frag_{i}") for i in range(3)}

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many",
                   side_effect=lambda fids: {fid: frags[fid] for fid in fids if fid in frags}):
            results = idx.search("AI ethics", top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_search_includes_similarity(self):
        idx, _ = self._make_loaded_index()
        frags = {f"frag_{i}": make_fragment(f"frag_{i}") for i in range(3)}

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many",
                   side_effect=lambda fids: {fid: frags[fid] for fid in fids if fid in frags}):
            results = idx.search("AI ethics", top_k=3)

        for r in results:
            assert "similarity" in r
            assert isinstance(r["similarity"], float)

    def test_similarity_is_1_minus_distance(self):
        idx, _ = self._make_loaded_index()
        frags = {f"frag_{i}": make_fragment(f"frag_{i}") for i in range(3)}

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many",
                   side_effect=lambda fids: {fid: frags[fid] for fid in fids if fid in frags}):
            results = idx.search("AI ethics", top_k=3)

        # distances were [0.1, 0.2, 0.3] → similarities are [0.9, 0.8, 0.7]
        assert abs(results[0]["similarity"] - 0.9) < 0.01

    def test_search_raises_when_collection_none(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        idx._loaded = True
        idx._collection = None

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768):
            with pytest.raises(RuntimeError, match="ChromaDB collection not available"):
                idx.search("AI ethics")

    def test_top_k_respected(self):
        idx, mock_collection = self._make_loaded_index()
        frags = {f"frag_{i}": make_fragment(f"frag_{i}") for i in range(3)}

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many",
                   side_effect=lambda fids: {fid: frags[fid] for fid in fids if fid in frags}):
            results = idx.search("AI ethics", top_k=2)

        assert len(results) <= 2

    def test_embedding_stripped_from_results(self):
        idx, _ = self._make_loaded_index()
        frag_with_embedding = {**make_fragment("frag_0"), "embedding": [0.1] * 768}
        frags = {"frag_0": frag_with_embedding, "frag_1": make_fragment("frag_1"),
                 "frag_2": make_fragment("frag_2")}

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many",
                   side_effect=lambda fids: {fid: frags[fid] for fid in fids if fid in frags}):
            results = idx.search("test", top_k=3)

        for r in results:
            assert "embedding" not in r

    def test_fallback_to_chroma_metadata_when_store_misses(self):
        idx, _ = self._make_loaded_index()

        with patch("app.services.embedding_index.embed_query", return_value=[0.1] * 768), \
             patch("app.services.embedding_index.fragment_store.get_many", return_value={}):
            results = idx.search("test", top_k=3)

        # Falls back to Chroma metadata → fragment_id still present
        assert all("fragment_id" in r for r in results)


# ---------------------------------------------------------------------------
# EmbeddingIndex.delete
# ---------------------------------------------------------------------------

class TestEmbeddingIndexDelete:
    def test_delete_calls_collection_delete_with_ids(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        idx._collection = mock_collection
        idx._loaded = True

        result = idx.delete(["frag_0", "frag_1"])

        mock_collection.delete.assert_called_once_with(ids=["frag_0", "frag_1"])
        assert result == 2

    def test_delete_empty_list_is_a_noop(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        idx._collection = mock_collection
        idx._loaded = True

        result = idx.delete([])

        mock_collection.delete.assert_not_called()
        assert result == 0

    def test_delete_raises_when_collection_none(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        idx._loaded = True
        idx._collection = None

        with pytest.raises(RuntimeError, match="ChromaDB collection not available"):
            idx.delete(["frag_0"])

    def test_delete_triggers_load_if_not_loaded(self):
        from app.services.embedding_index import EmbeddingIndex
        idx = EmbeddingIndex()
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        with patch("app.services.embedding_index.settings.chroma_mode", "local"), \
             patch("app.services.embedding_index._make_client", return_value=mock_client):
            idx.delete(["frag_0"])

        assert idx._loaded is True
        mock_collection.delete.assert_called_once_with(ids=["frag_0"])


class TestHttpCollectionDelete:
    def test_delete_posts_ids_to_delete_endpoint(self):
        from app.services.embedding_index import _HttpCollection

        col = _HttpCollection("http://chroma:8000/api/v2", "col-id", "fragments", {"Accept": "application/json"})

        captured = {}

        class _FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return b""

        def fake_urlopen(req, timeout=30):
            import json as _json
            captured["url"] = req.full_url
            captured["body"] = _json.loads(req.data)
            return _FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            col.delete(["frag_0", "frag_1"])

        assert captured["url"] == f"{col._col_url}/delete"
        assert captured["body"] == {"ids": ["frag_0", "frag_1"]}


# ---------------------------------------------------------------------------
# Module-level API (backward-compatible singleton)
# ---------------------------------------------------------------------------

class TestModuleLevelApi:
    def test_search_delegates_to_instance(self):
        from app.services import embedding_index as ei
        mock_instance = MagicMock()
        mock_instance.search.return_value = []

        with patch.object(ei, "_get_instance", return_value=mock_instance):
            ei.search("test query", top_k=5, filters={"theme": "T1"})

        mock_instance.search.assert_called_once_with("test query", 5, {"theme": "T1"})

    def test_available_delegates_to_instance(self):
        from app.services import embedding_index as ei
        mock_instance = MagicMock()
        mock_instance.available.return_value = True

        with patch.object(ei, "_get_instance", return_value=mock_instance):
            result = ei.available()

        assert result is True

    def test_reload_delegates_to_instance(self):
        from app.services import embedding_index as ei
        mock_instance = MagicMock()
        mock_instance.reload.return_value = {"fragments": 5}

        with patch.object(ei, "_get_instance", return_value=mock_instance):
            result = ei.reload()

        assert result == {"fragments": 5}

    def test_delete_delegates_to_instance(self):
        from app.services import embedding_index as ei
        mock_instance = MagicMock()
        mock_instance.delete.return_value = 2

        with patch.object(ei, "_get_instance", return_value=mock_instance):
            result = ei.delete(["frag_0", "frag_1"])

        mock_instance.delete.assert_called_once_with(["frag_0", "frag_1"])
        assert result == 2
