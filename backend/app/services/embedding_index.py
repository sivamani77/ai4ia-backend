"""Semantic search over the precomputed fragment embedding matrix.

Matrix rows are L2-normalized and aligned 1:1 with JSONL line order, so
cosine similarity == dot product. The matrix is memory-mapped (loaded once).
"""
import json
from threading import Lock
from typing import Any

import numpy as np

from app.core.paths import EMBEDDING_MATRIX_PATH, EMBEDDING_JSONL_PATH
from app.services import fragment_store
from app.services.llm_service import embed_query

_lock = Lock()
_state: dict[str, Any] = {"loaded": False}


def _ensure() -> dict[str, Any]:
    if not _state.get("loaded"):
        with _lock:
            if not _state.get("loaded"):
                if not EMBEDDING_MATRIX_PATH.exists() or not EMBEDDING_JSONL_PATH.exists():
                    _state.update({"loaded": True, "matrix": None, "ids": []})
                else:
                    matrix = np.load(EMBEDDING_MATRIX_PATH, mmap_mode="r")
                    ids = []
                    with EMBEDDING_JSONL_PATH.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                ids.append(json.loads(line)["fragment_id"])
                    _state.update({"loaded": True, "matrix": matrix, "ids": ids})
    return _state


def available() -> bool:
    return _ensure()["matrix"] is not None


def search(query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:
    """Return top_k fragments ranked by cosine similarity, with optional facet filters."""
    state = _ensure()
    matrix = state["matrix"]
    if matrix is None:
        raise RuntimeError("Embedding matrix not available; run 06_generate_embeddings.py")

    vec = np.asarray(embed_query(query), dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    scores = matrix @ vec  # cosine, rows already L2-normalized
    order = np.argsort(-scores)

    filters = filters or {}
    results: list[dict] = []
    for row in order:
        frag = fragment_store.get(state["ids"][row])
        if frag is None:
            continue
        if not _passes(frag, filters):
            continue
        out = dict(frag)
        out.pop("embedding", None)
        out["similarity"] = round(float(scores[row]), 4)
        results.append(out)
        if len(results) >= top_k:
            break
    return results


def _passes(frag: dict, filters: dict) -> bool:
    for key in ("theme", "chapter", "speaker", "edition", "year"):
        want = filters.get(key)
        if not want:
            continue
        if key == "theme":
            have = str(frag.get("primary_theme") or "")
        elif key == "chapter":
            have = str(frag.get("primary_chapter") or "")
        else:
            have = str(frag.get(key) or "")
        if str(want).lower() not in have.lower():
            return False
    return True
