import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.core.paths import PROJECT_ROOT
from app.core.settings import settings
from app.pipeline.storyline_gen.candidate_scoring import (
    candidate_search_blob,
    match_terms_in_text,
)
from app.pipeline.storyline_gen.llm_client import (
    EMBED_MODEL,
    EMBED_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    log,
    post_with_retries,
)
from app.pipeline.storyline_gen.prompt_intent import candidate_speaker_matches
from app.pipeline.storyline_gen.text_utils import (
    normalize_name,
    parse_list_field,
)


# Allow the stage runner to override the embeddings directory via env var.
# In S3 deployment mode the stage runner downloads the embeddings from S3
# to a /tmp path and passes AI4IA_EMBEDDINGS_DIR so we read the correctly
# classified data instead of whatever is on the worker VM's local disk.
_env_embeddings_dir = os.environ.get("AI4IA_EMBEDDINGS_DIR")
EMBEDDINGS_DIR = Path(_env_embeddings_dir) if _env_embeddings_dir else PROJECT_ROOT / "data" / "embeddings"

MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"

# Runtime-configurable corpus state (set once in main() before any retrieval)
_ACTIVE_EMBEDDINGS_DIR: Path = EMBEDDINGS_DIR
_ACTIVE_MATRIX_PATH: Path = MATRIX_PATH
_ACTIVE_JSONL_PATH: Path = JSONL_PATH
_ACTIVE_CORPUS_VERSION: str = "original_v1"
_FILTERED_RECORDS: list | None = None
_FILTERED_MATRIX: Any = None
_CORPUS_DIAGNOSTICS: dict[str, Any] = {}

# ChromaDB-loaded corpus (populated by _load_corpus_from_chroma when CHROMA_MODE is set)
_CHROMA_RECORDS: list | None = None
_CHROMA_COLLECTION: Any = None
# Quality/audio-status where-clause built by activate_corpus(), reused by every
# live semantic ANN query so retrieval stays scoped to the same allowed set.
_ACTIVE_CHROMA_WHERE: dict | None = None


def _chroma_mode() -> bool:
    return os.environ.get("CHROMA_MODE", "").strip().lower() == "http"


def _get_chroma_collection():
    """Return a cached ChromaDB collection handle (one HTTP client per job)."""
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    import chromadb

    host = os.environ.get("CHROMA_HOST", "chromadb")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "fragments")
    auth_token = os.environ.get("CHROMA_AUTH_TOKEN", "")

    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
    client = chromadb.HttpClient(host=host, port=port, headers=headers)
    _CHROMA_COLLECTION = client.get_collection(collection_name)
    return _CHROMA_COLLECTION


def _chroma_quality_where(exclude_manual_review: bool, min_transcript_quality: float) -> dict | None:
    """Build a Chroma `where` clause mirroring activate_corpus()'s Python-side
    quality filter, so a live semantic ANN query stays scoped to the same
    allowed fragments instead of only being filtered after the fact."""
    clauses: list[dict] = []
    if exclude_manual_review:
        clauses.append({"audio_quality_status": {"$ne": "manual_review"}})
    if min_transcript_quality and min_transcript_quality > 0:
        clauses.append({"transcript_quality_score": {"$gte": float(min_transcript_quality)}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _load_corpus_from_chroma() -> list[dict[str, Any]]:
    """Load fragment metadata (NOT embedding vectors) from ChromaDB.

    Semantic ranking is done via a live ANN query against Chroma's own index
    (see _semantic_rank_via_chroma) instead of pulling every fragment's
    embedding vector over HTTP and brute-forcing similarity locally — this
    used to transfer and rescan the entire corpus on every single user
    request. Full metadata (no vectors) is still needed here for the
    keyword/theme/chapter/speaker retrieval channels, which scan by text and
    exact-match fields rather than by embedding similarity.
    """
    global _CHROMA_RECORDS
    if _CHROMA_RECORDS is not None:
        return _CHROMA_RECORDS

    collection = _get_chroma_collection()
    total = collection.count()
    if total == 0:
        raise RuntimeError("ChromaDB collection is empty — run ingestion first")

    log(f"Loading {total} fragment records (metadata only) from ChromaDB")

    batch_size = 500
    all_records = []
    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        for fid, meta in zip(result["ids"], result["metadatas"]):
            row = dict(meta or {})
            row["fragment_id"] = fid
            all_records.append(row)
        offset += batch_size

    _CHROMA_RECORDS = all_records
    log(f"ChromaDB corpus metadata loaded: {len(all_records)} records")
    return all_records


_QUERY_VECTOR_CACHE: dict[str, Any] = {}


def _query_vector(prompt: str) -> np.ndarray:
    """embed_query, memoized for the life of the job.

    Retrieval embeds the same prompt twice — once for the ANN query, once for
    the score backfill below — and the embedding host is a remote HTTP call.
    """
    cached = _QUERY_VECTOR_CACHE.get(prompt)
    if cached is None:
        cached = embed_query(prompt)
        _QUERY_VECTOR_CACHE[prompt] = cached
    return cached


def _backfill_semantic_scores(candidates: list[dict[str, Any]], prompt: str) -> int:
    """Give every candidate a real semantic_score, not just the ANN winners.

    score_all_records only scores Chroma's top-N ANN hits; everything else is
    returned as bare metadata. Candidates that reach the merged set through
    the keyword/theme/speaker/title channels therefore arrived with no
    semantic_score at all, which read as 0.0 downstream — and since
    semantic_score carries 0.30 of final_prompt_score and gates every role
    branch in classify_candidate_relevance_v2, that silently capped them at
    weak_related no matter how well they matched. bucket_candidates then files
    weak_related under supporting_candidates, which the export timeline never
    reads, so those clips could not appear in the rendered video.

    Fetching the stored vectors and scoring them directly reproduces exactly
    what the ANN query would have returned: the collection is cosine-space
    with unit-norm vectors, so a dot product of normalized vectors is the same
    number as Chroma's own `1 - distance`.

    Returns the number of candidates scored.
    """
    missing = [c for c in candidates if c.get("semantic_score") is None]
    if not missing:
        return 0

    by_id: dict[str, list[dict[str, Any]]] = {}
    for candidate in missing:
        fragment_id = str(candidate.get("fragment_id") or "")
        if fragment_id:
            by_id.setdefault(fragment_id, []).append(candidate)
    if not by_id:
        return 0

    query_vector = _query_vector(prompt)
    norm = float(np.linalg.norm(query_vector))
    if norm > 0:
        query_vector = query_vector / norm

    collection = _get_chroma_collection()
    ids = list(by_id)
    scored = 0
    batch_size = 200
    for offset in range(0, len(ids), batch_size):
        batch = ids[offset: offset + batch_size]
        try:
            fetched = collection.get(ids=batch, include=["embeddings"])
        except Exception as exc:
            log(f"Semantic backfill failed for {len(batch)} fragments: {exc}")
            continue
        embeddings = fetched.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            continue
        for fragment_id, vector in zip(fetched["ids"], embeddings):
            vector = np.asarray(vector, dtype=np.float32)
            vector_norm = float(np.linalg.norm(vector))
            if vector_norm > 0:
                vector = vector / vector_norm
            score = round(float(vector @ query_vector), 6)
            for candidate in by_id.get(fragment_id, []):
                candidate["semantic_score"] = score
                candidate["semantic_score_backfilled"] = True
                scored += 1

    # Anything the fetch could not cover still needs the key present:
    # select_balanced_candidates reads record["semantic_score"] directly and
    # would raise KeyError on a candidate whose other scores are all zero.
    for candidate in missing:
        if candidate.get("semantic_score") is None:
            candidate["semantic_score"] = 0.0

    return scored


def _semantic_rank_via_chroma(
    prompt: str,
    top_k: int,
    where: dict | None = None,
    allowed_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the top_k fragments most semantically similar to prompt, using
    ChromaDB's own ANN index — replaces a full-corpus brute-force scan.

    Over-fetches when allowed_ids is given so post-filtering to the same
    quality-gated set activate_corpus() computed still leaves top_k results.
    """
    collection = _get_chroma_collection()
    query_vector = _query_vector(prompt)

    n_results = top_k * 5 if allowed_ids is not None else top_k
    n_results = max(1, min(n_results, collection.count()))

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector.tolist()],
        "n_results": n_results,
        "include": ["metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    ranked: list[dict[str, Any]] = []
    for fid, dist, meta in zip(raw["ids"][0], raw["distances"][0], raw["metadatas"][0]):
        if allowed_ids is not None and fid not in allowed_ids:
            continue
        record = dict(meta or {})
        record["fragment_id"] = fid
        record["semantic_score"] = round(1.0 - float(dist), 6)
        ranked.append(record)
        if len(ranked) >= top_k:
            break

    for rank, record in enumerate(ranked, start=1):
        record["retrieval_rank"] = rank

    return ranked


DEFAULT_TOP_K = 90


def score_all_records(prompt: str, semantic_k: int = None) -> list[dict[str, Any]]:
    global _FILTERED_RECORDS, _FILTERED_MATRIX

    if semantic_k is None:
        semantic_k = DEFAULT_TOP_K

    if _chroma_mode():
        # Full corpus metadata (already quality-filtered by activate_corpus())
        # is used for the keyword/theme/speaker channels; semantic ranking is
        # a live ANN query instead of a local brute-force scan.
        records = _FILTERED_RECORDS if _FILTERED_RECORDS is not None else _load_corpus_from_chroma()
        allowed_ids = {r["fragment_id"] for r in records}
        top_semantic = _semantic_rank_via_chroma(
            prompt, semantic_k, where=_ACTIVE_CHROMA_WHERE, allowed_ids=allowed_ids
        )
        semantic_ids = {r["fragment_id"] for r in top_semantic}
        rest = [r for r in records if r.get("fragment_id") not in semantic_ids]
        return top_semantic + rest

    if _FILTERED_RECORDS is not None and _FILTERED_MATRIX is not None:
        records = _FILTERED_RECORDS
        matrix = _FILTERED_MATRIX
    else:
        if not _ACTIVE_MATRIX_PATH.exists():
            raise FileNotFoundError(f"Missing embedding matrix: {_ACTIVE_MATRIX_PATH}")
        if not _ACTIVE_JSONL_PATH.exists():
            raise FileNotFoundError(f"Missing embedding JSONL: {_ACTIVE_JSONL_PATH}")
        matrix = np.load(_ACTIVE_MATRIX_PATH)
        records = load_embedding_records()

    if matrix.shape[0] != len(records):
        raise RuntimeError(
            f"Matrix/metadata mismatch. Matrix rows={matrix.shape[0]}, records={len(records)}"
        )

    query_vector = embed_query(prompt)
    similarities = matrix @ query_vector
    top_indices = np.argsort(similarities)[::-1]

    scored = []
    for rank, idx in enumerate(top_indices, start=1):
        record = dict(records[int(idx)])
        record["retrieval_rank"] = rank
        record["semantic_score"] = round(float(similarities[int(idx)]), 6)
        scored.append(record)

    return scored


def embed_query(query: str) -> np.ndarray:
    """
    Embed the user prompt for retrieval.

    Supports both Ollama embedding APIs:
    - /api/embed       -> {"model": ..., "input": ...}
    - /api/embeddings -> {"model": ..., "prompt": ...}

    Some Ollama deployments expose only one of these endpoints.
    """
    normalized_query = query.strip()

    if not normalized_query.startswith("search_query:"):
        normalized_query = "search_query: " + normalized_query

    errors = []

    # Newer Ollama endpoint
    embed_url = f"{OLLAMA_BASE_URL}/api/embed"
    embed_payload = {
        "model": EMBED_MODEL,
        "input": normalized_query,
    }

    try:
        data = post_with_retries(embed_url, embed_payload, timeout=EMBED_TIMEOUT_SECONDS)

        if "embedding" in data:
            return np.array(data["embedding"], dtype=np.float32)

        if "embeddings" in data and data["embeddings"]:
            return np.array(data["embeddings"][0], dtype=np.float32)

        errors.append(f"/api/embed returned unexpected response keys: {list(data.keys())}")
    except Exception as exc:
        errors.append(f"/api/embed failed: {exc}")

    # Older Ollama endpoint
    embeddings_url = f"{OLLAMA_BASE_URL}/api/embeddings"
    embeddings_payload = {
        "model": EMBED_MODEL,
        "prompt": normalized_query,
    }

    try:
        data = post_with_retries(embeddings_url, embeddings_payload, timeout=EMBED_TIMEOUT_SECONDS)

        if "embedding" in data:
            return np.array(data["embedding"], dtype=np.float32)

        if "embeddings" in data and data["embeddings"]:
            return np.array(data["embeddings"][0], dtype=np.float32)

        errors.append(f"/api/embeddings returned unexpected response keys: {list(data.keys())}")
    except Exception as exc:
        errors.append(f"/api/embeddings failed: {exc}")

    raise RuntimeError("Embedding request failed. " + " | ".join(errors))


def load_embedding_records() -> list[dict[str, Any]]:
    if _chroma_mode():
        return _load_corpus_from_chroma()

    records = []
    with _ACTIVE_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                record.pop("embedding", None)
                record.pop("embedding_text", None)
                records.append(record)
    return records


def score_and_retrieve(prompt: str, top_k: int) -> list[dict[str, Any]]:
    return score_all_records(prompt)[:top_k]


def merge_candidate_sets(*candidate_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        for item in candidate_set:
            fragment_id = str(item.get("fragment_id") or "")
            if not fragment_id:
                continue
            channels = list(dict.fromkeys(item.get("retrieval_channels") or []))
            if fragment_id in merged:
                existing = merged[fragment_id]
                existing["retrieval_channels"] = list(
                    dict.fromkeys((existing.get("retrieval_channels") or []) + channels)
                )
                continue
            candidate = dict(item)
            candidate["retrieval_channels"] = channels
            merged[fragment_id] = candidate
            ordered.append(candidate)
    return ordered


def hybrid_retrieve_candidates(
    profile: dict,
    records: list[dict],
    embeddings_index,
    top_k: int = 100,
) -> list[dict]:
    requested_speaker = profile.get("requested_speaker")
    topic_terms = profile.get("topic_terms") or []
    expanded_terms = profile.get("expanded_terms") or []
    search_terms = list(dict.fromkeys(topic_terms + expanded_terms))
    theme_hint = profile.get("theme_hint")
    chapter_hint = profile.get("chapter_hint")
    requested_source_video = profile.get("requested_source_video")

    def with_channel(items: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
        result = []
        for item in items:
            candidate = dict(item)
            candidate["retrieval_channels"] = list(dict.fromkeys((candidate.get("retrieval_channels") or []) + [channel]))
            result.append(candidate)
        return result

    semantic_pool = with_channel(list(embeddings_index[:top_k]), "semantic")

    keyword_pool: list[dict[str, Any]] = []
    theme_pool: list[dict[str, Any]] = []
    speaker_pool: list[dict[str, Any]] = []
    panel_pool: list[dict[str, Any]] = []
    title_pool: list[dict[str, Any]] = []

    for record in records:
        blob = candidate_search_blob(record)
        if requested_source_video and str(record.get("source_video") or "") == requested_source_video:
            title_pool.append(dict(record))
        if search_terms:
            direct_hits = match_terms_in_text(blob, topic_terms)
            expanded_hits = match_terms_in_text(blob, expanded_terms)
            if direct_hits or expanded_hits:
                keyword_pool.append(dict(record))
        if theme_hint and normalize_name(str(record.get("primary_theme") or "")) == normalize_name(str(theme_hint)):
            theme_pool.append(dict(record))
        if chapter_hint and normalize_name(str(record.get("primary_chapter") or "")) == normalize_name(str(chapter_hint)):
            theme_pool.append(dict(record))
        if requested_speaker:
            speaker_match, speaker_reason = candidate_speaker_matches(record, requested_speaker)
            if speaker_match:
                candidate = dict(record)
                candidate["speaker_match"] = True
                candidate["speaker_match_reason"] = speaker_reason
                speaker_pool.append(candidate)
            else:
                panel_match, panel_reason = candidate_speaker_matches(record, requested_speaker, allow_panel=True)
                if panel_match:
                    candidate = dict(record)
                    candidate["speaker_match"] = True
                    candidate["speaker_match_panel"] = True
                    candidate["speaker_match_reason"] = panel_reason
                    panel_pool.append(candidate)

    keyword_pool = sorted(
        with_channel(keyword_pool, "keyword"),
        key=lambda item: (
            len(match_terms_in_text(candidate_search_blob(item), topic_terms)),
            len(match_terms_in_text(candidate_search_blob(item), expanded_terms)),
            float(item.get("semantic_score") or 0.0),
        ),
        reverse=True,
    )[: max(10, top_k // 3)]
    theme_pool = sorted(
        with_channel(theme_pool, "chapter" if chapter_hint else "theme"),
        key=lambda item: float(item.get("semantic_score") or 0.0),
        reverse=True,
    )[: max(10, top_k // 3)]
    def by_quality(item: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(item.get("quote_strength") or 0.0),
            float(item.get("clarity_score") or 0.0),
            float(item.get("semantic_score") or 0.0),
        )

    speaker_pool = sorted(with_channel(speaker_pool, "speaker"), key=by_quality, reverse=True)
    # Panel clips are a fallback: used only when the speaker has too few solo
    # clips, and ranked after them.
    if len(speaker_pool) < settings.speaker_panel_fallback_min_clips:
        speaker_pool += sorted(with_channel(panel_pool, "speaker_panel"), key=by_quality, reverse=True)
    speaker_pool = speaker_pool[:top_k]

    # The title channel is unbounded on purpose: when the prompt names a
    # specific video, every fragment of that video is in scope and the
    # diversity caps downstream decide how many are actually used. Capping it
    # here would hand that decision to whatever arbitrary slice sorted first.
    title_pool = sorted(
        with_channel(title_pool, "title"),
        key=lambda item: (
            float(item.get("quote_strength") or 0.0),
            float(item.get("clarity_score") or 0.0),
        ),
        reverse=True,
    )

    if requested_speaker:
        return merge_candidate_sets(title_pool, speaker_pool, keyword_pool, theme_pool)
    return merge_candidate_sets(title_pool, semantic_pool, keyword_pool, theme_pool)


def compute_retrieval_diagnostics(profile: dict[str, Any], retrieved: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, Any]:
    semantic_count = sum(1 for item in retrieved if "semantic" in (item.get("retrieval_channels") or []))
    keyword_count = sum(1 for item in retrieved if "keyword" in (item.get("retrieval_channels") or []))
    chapter_count = sum(1 for item in retrieved if any(ch in (item.get("retrieval_channels") or []) for ch in ("chapter", "theme")))
    speaker_count = sum(1 for item in retrieved if "speaker" in (item.get("retrieval_channels") or []))
    return {
        "semantic_count": min(top_k, semantic_count),
        "keyword_count": keyword_count,
        "chapter_count": chapter_count,
        "speaker_count": speaker_count,
        "merged_count": len(retrieved),
        "selected_count": len(buckets.get("selected_candidates") or []),
        "supporting_count": len(buckets.get("supporting_candidates") or []),
        "context_count": len(buckets.get("context_candidates") or []),
        "rejected_count": len(buckets.get("rejected_candidates") or []),
        "speaker_required": bool(profile.get("requested_speaker")),
    }


def retrieve_candidates(
    prompt: str,
    intent: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    scored = score_all_records(prompt, semantic_k=max(top_k, DEFAULT_TOP_K))
    hybrid = hybrid_retrieve_candidates(intent, scored, scored, top_k=top_k)
    if _chroma_mode():
        backfilled = _backfill_semantic_scores(hybrid, prompt)
        if backfilled:
            log(f"Backfilled semantic_score for {backfilled} non-ANN candidates")
    return hybrid[: max(top_k, len(hybrid))]


def utility_boost(record: dict[str, Any]) -> float:
    usage = set(parse_list_field(record.get("clip_usage_candidates")))

    boost = 0.0

    if "speaker_soundbite" in usage:
        boost += 0.08
    if "narration_support" in usage:
        boost += 0.06
    if "evidence_clip" in usage:
        boost += 0.05
    if "context_visual" in usage:
        boost += 0.03

    quote_strength = float(record.get("quote_strength") or 0)
    clarity_score = float(record.get("clarity_score") or 0)
    narration_support_score = float(record.get("narration_support_score") or 0)
    classification_confidence = float(record.get("classification_confidence") or 0)

    boost += quote_strength * 0.06
    boost += clarity_score * 0.04
    boost += narration_support_score * 0.04
    boost += classification_confidence * 0.03

    duration = float(record.get("duration") or 0)

    if 20 <= duration <= 100:
        boost += 0.04
    elif duration > 160:
        boost -= 0.05

    return boost


def bucket_candidates_by_year(candidates: list[dict[str, Any]]) -> dict[int | None, list[dict[str, Any]]]:
    buckets: dict[int | None, list[dict[str, Any]]] = {}
    for candidate in candidates:
        year = candidate.get("video_year")  # int or None
        buckets.setdefault(year, []).append(candidate)
    for year in buckets:
        buckets[year] = sorted(
            buckets[year],
            key=lambda c: float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or c.get("semantic_score") or 0),
            reverse=True,
        )
    return buckets


def activate_corpus(
    embeddings_dir: str,
    corpus_version: str,
    min_transcript_quality: float,
    exclude_manual_review: bool,
) -> dict[str, Any]:
    """
    Configure the active embedding corpus globals, load and optionally filter
    records+matrix, and return corpus diagnostics.
    """
    global _ACTIVE_EMBEDDINGS_DIR, _ACTIVE_MATRIX_PATH, _ACTIVE_JSONL_PATH
    global _ACTIVE_CORPUS_VERSION, _FILTERED_RECORDS, _FILTERED_MATRIX, _CORPUS_DIAGNOSTICS
    global _ACTIVE_CHROMA_WHERE

    active_dir = Path(embeddings_dir)
    _ACTIVE_EMBEDDINGS_DIR = active_dir
    _ACTIVE_MATRIX_PATH = active_dir / "fragment_embedding_matrix.npy"
    _ACTIVE_JSONL_PATH = active_dir / "classified_fragments_embeddings.jsonl"
    _ACTIVE_CORPUS_VERSION = corpus_version

    log(f"Embedding dir: {active_dir}")
    log(f"Corpus version: {corpus_version}")

    use_chroma = _chroma_mode()

    if use_chroma:
        # Metadata only — no embedding vectors. Semantic ranking happens via
        # a live ANN query (score_all_records), not a local brute-force scan,
        # so there is no local matrix to build here.
        all_records = _load_corpus_from_chroma()
        matrix = None
    else:
        if not _ACTIVE_MATRIX_PATH.exists():
            raise FileNotFoundError(f"Missing embedding matrix: {_ACTIVE_MATRIX_PATH}")
        if not _ACTIVE_JSONL_PATH.exists():
            raise FileNotFoundError(f"Missing embedding JSONL: {_ACTIVE_JSONL_PATH}")
        all_records = load_embedding_records()
        matrix = np.load(_ACTIVE_MATRIX_PATH)

        if matrix.shape[0] != len(all_records):
            raise RuntimeError(
                f"Matrix/metadata mismatch: matrix rows={matrix.shape[0]}, records={len(all_records)}"
            )

    log(f"Embedding records loaded: {len(all_records)}")

    # Count version and audio_status distributions before filtering
    version_counts_before: dict[str, int] = {}
    audio_counts_before: dict[str, int] = {}
    for rec in all_records:
        v = rec.get("source_transcript_version") or "unknown"
        version_counts_before[v] = version_counts_before.get(v, 0) + 1
        a = rec.get("audio_quality_status") or "unknown"
        audio_counts_before[a] = audio_counts_before.get(a, 0) + 1

    embedding_dim = int(matrix.shape[1]) if (matrix is not None and matrix.ndim == 2) else 0

    # Determine which source_transcript_version values are allowed
    # "mixed_cleaned_v2" means use all records regardless of version label
    # "original_v1" or "cleaned_v2" restrict to that exact version
    # Any other string is treated as a metadata label only (no version filtering)
    if corpus_version == "mixed_cleaned_v2":
        allowed_source_versions = None  # no version filtering
    elif corpus_version in {"original_v1", "cleaned_v2"}:
        allowed_source_versions: set | None = {corpus_version}
    else:
        allowed_source_versions = None  # unknown label — don't drop records

    filtered_records: list[dict[str, Any]] = []
    filtered_vectors: list[Any] = []

    for idx, record in enumerate(all_records):
        if allowed_source_versions is not None:
            rec_version = record.get("source_transcript_version") or "original_v1"
            if rec_version not in allowed_source_versions:
                continue

        if exclude_manual_review and record.get("audio_quality_status") == "manual_review":
            continue

        quality = record.get("transcript_quality_score")
        if quality is None:
            quality = 1.0
        if float(quality) < min_transcript_quality:
            continue

        filtered_records.append(record)
        if not use_chroma:
            filtered_vectors.append(matrix[idx])

    records_after = len(filtered_records)
    log(f"Records after filtering: {records_after}")

    if use_chroma:
        # No local matrix in Chroma mode — semantic ranking is a live ANN
        # query scoped by _ACTIVE_CHROMA_WHERE below, not a local brute force.
        _FILTERED_MATRIX = None
        _ACTIVE_CHROMA_WHERE = _chroma_quality_where(exclude_manual_review, min_transcript_quality)
    elif filtered_vectors:
        _FILTERED_MATRIX = np.stack(filtered_vectors, axis=0)
        _ACTIVE_CHROMA_WHERE = None
    else:
        _FILTERED_MATRIX = np.empty((0, embedding_dim), dtype=matrix.dtype)
        _ACTIVE_CHROMA_WHERE = None

    _FILTERED_RECORDS = filtered_records

    log(f"Embedding matrix shape: {_FILTERED_MATRIX.shape if _FILTERED_MATRIX is not None else 'n/a (chroma ANN mode)'}")

    # Count version and audio_status distributions after filtering
    version_counts_after: dict[str, int] = {}
    audio_counts_after: dict[str, int] = {}
    for rec in filtered_records:
        v = rec.get("source_transcript_version") or "unknown"
        version_counts_after[v] = version_counts_after.get(v, 0) + 1
        a = rec.get("audio_quality_status") or "unknown"
        audio_counts_after[a] = audio_counts_after.get(a, 0) + 1

    diagnostics = {
        "total_embedding_records": len(all_records),
        "records_after_filtering": records_after,
        "source_version_counts_before": version_counts_before,
        "source_version_counts_after": version_counts_after,
        "audio_status_counts_before": audio_counts_before,
        "audio_status_counts_after": audio_counts_after,
        "cleaned_v2_records": version_counts_before.get("cleaned_v2", 0),
        "original_v1_records": version_counts_before.get("original_v1", 0),
        "manual_review_records": audio_counts_before.get("manual_review", 0),
        "min_transcript_quality": min_transcript_quality,
        "exclude_manual_review": exclude_manual_review,
        "corpus_version": corpus_version,
        "embeddings_dir": str(active_dir),
        "embedding_dimension": embedding_dim,
    }
    _CORPUS_DIAGNOSTICS = diagnostics
    return diagnostics


def get_filtered_records_or_load() -> list[dict[str, Any]]:
    """Return the corpus records activate_corpus() already filtered, or load them fresh."""
    return _FILTERED_RECORDS if _FILTERED_RECORDS is not None else load_embedding_records()
