"""Semantic search over fragment embeddings stored in ChromaDB.

Local dev  → PersistentClient reading data/chroma/ on disk.
Deployed   → HttpClient pointing at a remote Chroma server (e.g. Azure).

Switch via .env:
    CHROMA_MODE=local                         # default
    CHROMA_MODE=http
    CHROMA_HOST=my-chroma.azure.com
    CHROMA_PORT=8000
    CHROMA_SSL=true
    CHROMA_AUTH_TOKEN=<bearer-token>          # if the server requires auth
    CHROMA_COLLECTION=fragments               # default
"""
from __future__ import annotations

import logging
from threading import RLock
from typing import Any

logger = logging.getLogger("ai4ia.embedding_index")

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.paths import CHROMA_DIR
from app.core.settings import settings
from app.services import fragment_store
from app.services.llm_service import embed_query


class _HttpCollection:
    """Minimal ChromaDB collection wrapper using raw HTTP (avoids auth-init bug in 1.5.x)."""

    def __init__(self, base_url: str, collection_id: str, collection_name: str, headers: dict) -> None:
        self._base = base_url
        self._id = collection_id
        self._name = collection_name
        self._headers = headers
        self._col_url = f"{base_url}/tenants/default_tenant/databases/default_database/collections/{collection_id}"

    def count(self) -> int:
        import urllib.request, json as _json
        req = urllib.request.Request(f"{self._col_url}/count", headers=self._headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return int(r.read())

    def query(self, query_embeddings: list, n_results: int = 10,
              include: list | None = None, where: dict | None = None) -> dict:
        import urllib.request, json as _json
        body: dict = {"query_embeddings": query_embeddings, "n_results": n_results,
                      "include": include or ["distances", "metadatas"]}
        if where:
            body["where"] = where
        data = _json.dumps(body).encode()
        req = urllib.request.Request(f"{self._col_url}/query", data=data,
                                     headers={**self._headers, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return _json.loads(r.read())


def _make_http_collection(collection_name: str) -> "_HttpCollection":
    """Connect to ChromaDB over raw HTTP, bypassing the broken 1.5.x Python client init."""
    import urllib.request, json as _json
    protocol = "https" if settings.chroma_ssl else "http"
    base = f"{protocol}://{settings.chroma_host}:{settings.chroma_port}/api/v2"
    headers: dict = {"Accept": "application/json"}
    if settings.chroma_auth_token:
        headers["Authorization"] = f"Bearer {settings.chroma_auth_token}"

    url = f"{base}/tenants/default_tenant/databases/default_database/collections"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        collections = _json.loads(r.read())

    for col in collections:
        if col["name"] == collection_name:
            return _HttpCollection(base, col["id"], col["name"], headers)
    raise ValueError(f"Collection '{collection_name}' not found in ChromaDB")


def _make_client() -> chromadb.ClientAPI:
    if settings.chroma_mode == "http":
        # Return None — callers that need a raw collection use _make_http_collection()
        # This path is kept for local PersistentClient only
        raise RuntimeError("Use _make_http_collection() for http mode")

    # local — persistent on disk
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class EmbeddingIndex:
    def __init__(self) -> None:
        self._lock = RLock()
        self._collection: chromadb.Collection | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Load / reload
    # ------------------------------------------------------------------

    def load(self) -> dict[str, int]:
        with self._lock:
            try:
                if settings.chroma_mode == "http":
                    self._collection = _make_http_collection(settings.chroma_collection)
                else:
                    client = _make_client()
                    self._collection = client.get_collection(settings.chroma_collection)
                count = self._collection.count()
            except Exception:
                logger.warning("ChromaDB collection unavailable", exc_info=True)
                self._collection = None
                count = 0
            self._loaded = True
        return {"fragments": count}

    def reload(self) -> dict[str, int]:
        with self._lock:
            self._loaded = False
            self._collection = None
        return self.load()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._collection is not None and self._collection.count() > 0

    def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
        """Return top_k fragments ranked by cosine similarity."""
        if not self._loaded:
            self.load()

        if self._collection is None:
            raise RuntimeError(
                "ChromaDB collection not available. "
                "Local: run scripts/import_embeddings_to_chroma.py. "
                "Remote: check CHROMA_HOST / CHROMA_MODE in .env."
            )

        vec = embed_query(query)
        where = _build_where(filters or {})

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [vec],
            # over-fetch so post-filter still returns top_k
            "n_results": min(top_k * 3, self._collection.count()),
            "include": ["distances", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        ids = raw["ids"][0]
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]

        # One batched Supabase round trip for all candidate ids instead of one per row.
        frags_by_id = fragment_store.get_many(ids)

        results: list[dict] = []
        for rid, dist, meta in zip(ids, dists, metas):
            # Prefer the full Supabase row; fall back to Chroma metadata
            frag = frags_by_id.get(rid) or {"fragment_id": rid, **meta}
            out = dict(frag)
            out.pop("embedding", None)
            out["similarity"] = round(1.0 - float(dist), 4)  # cosine dist → similarity
            results.append(out)
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_where(filters: dict) -> dict | None:
    """Convert API filter dict to a ChromaDB where clause.

    Only fields stored as Chroma metadata scalars are filtered here.
    Fields that live only in Supabase (speaker, edition, year) are left
    to post-filtering by callers on the returned fragment dicts.
    """
    # Caller-facing filter keys ("theme", "chapter") stay as-is; the Chroma
    # metadata field names are primary_theme/primary_chapter (verbatim from
    # fragments_classified — see import_embeddings_to_chroma.py's
    # sanitize_metadata, which no longer renames these).
    field_map = {
        "theme": "primary_theme",
        "chapter": "primary_chapter",
        "audio_quality_status": "audio_quality_status",
    }
    clauses: list[dict] = []
    for filter_key, chroma_field in field_map.items():
        val = filters.get(filter_key)
        if val:
            clauses.append({chroma_field: {"$eq": str(val)}})

    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


# ---------------------------------------------------------------------------
# Backward-compatible module-level API  (archive.py / admin.py unchanged)
# ---------------------------------------------------------------------------

_instance: EmbeddingIndex | None = None


def _get_instance() -> EmbeddingIndex:
    global _instance
    if _instance is None:
        _instance = EmbeddingIndex()
    return _instance


def available() -> bool:
    return _get_instance().available()


def search(query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
    return _get_instance().search(query, top_k, filters)


def reload() -> dict[str, int]:
    return _get_instance().reload()
