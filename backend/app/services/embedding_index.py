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


def _make_client() -> chromadb.ClientAPI:
    if settings.chroma_mode == "http":
        kwargs: dict[str, Any] = {
            "host": settings.chroma_host,
            "port": settings.chroma_port,
            "ssl": settings.chroma_ssl,
        }
        if settings.chroma_auth_token:
            kwargs["headers"] = {"Authorization": f"Bearer {settings.chroma_auth_token}"}
        return chromadb.HttpClient(**kwargs)

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

        results: list[dict] = []
        for rid, dist, meta in zip(
            raw["ids"][0],
            raw["distances"][0],
            raw["metadatas"][0],
        ):
            # Prefer the full Supabase row; fall back to Chroma metadata
            frag = fragment_store.get(rid) or {"fragment_id": rid, **meta}
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
    clauses: list[dict] = []
    for key in ("theme", "chapter", "audio_quality_status"):
        val = filters.get(key)
        if val:
            clauses.append({key: {"$eq": str(val)}})

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
