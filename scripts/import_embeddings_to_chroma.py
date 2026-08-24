"""
Import fragment embeddings from the cleaned JSONL into a local ChromaDB collection.

Usage:
    python scripts/import_embeddings_to_chroma.py [--reset]

Options:
    --reset   Drop and recreate the collection before importing (default: skip
              fragments already present, only add new ones).

ChromaDB is stored at:  data/chroma/
Collection name:        fragments

Each record in ChromaDB:
    id         → fragment_id
    embedding  → 768-dim nomic-embed-text:v1.5 vector (pre-computed)
    document   → embedding_text (the text that was embedded)
    metadata   → scalar fields: video_id, source_video, fragment_index,
                 start, end, duration, word_count, theme, chapter,
                 storyline_role, summary, classification_confidence,
                 audio_quality_status, transcript_quality_score,
                 embedding_model, embedding_created_at
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))
# Use the full classified JSONL (data/embeddings/) which has quote_strength,
# clarity_score, clip_usage_candidates — required for speaker clip selection.
JSONL_PATH = PROJECT_ROOT / "data" / "embeddings" / "classified_fragments_embeddings.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fragments")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
BATCH_SIZE = 500



# Fields excluded from Chroma metadata: "embedding" is stored as the actual
# vector (not metadata), "fragment_id" is stored as the Chroma record id, and
# "embedding_text" is stored verbatim as the Chroma "document" — duplicating
# any of these into metadata as well would be pure waste, not added fidelity.
_METADATA_EXCLUDED_FIELDS = {"embedding", "fragment_id", "embedding_text"}


def sanitize_metadata(row: dict) -> dict:
    """Pass through every field from the JSONL record as Chroma metadata,
    verbatim by field name — every consumer of this collection (scripts/
    07_generate_storyline.py's CHROMA_MODE=http path, embedding_index.py)
    must see the exact same field names the JSONL/fragments_classified data
    uses (primary_theme, primary_chapter, speaker, keywords, ...), not a
    curated/renamed subset. ChromaDB metadata values must be str/int/float/
    bool, so lists and dicts are JSON-encoded and None becomes "".
    """
    metadata: dict = {}
    for key, value in row.items():
        if key in _METADATA_EXCLUDED_FIELDS:
            continue
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, (list, dict)):
            metadata[key] = json.dumps(value, ensure_ascii=False)
        else:
            metadata[key] = str(value)
    return metadata


def _facets_for(row: dict, facets_cache: dict) -> dict:
    """derive_facets() is keyed off (title, source_video) and is pure/cheap,
    but every fragment from the same video would otherwise recompute it —
    cache per source_video within one import run."""
    from app.services.enrichment import derive_facets

    source_video = row.get("source_video") or ""
    if source_video not in facets_cache:
        facets_cache[source_video] = derive_facets(row.get("video_title") or source_video, source_video)
    return facets_cache[source_video]


def to_postgres_row(row: dict, facets_cache: dict) -> dict:
    """Shape one JSONL record into the Postgres `fragments` row — same field
    set as FragmentRepository.upsert_many() expects."""
    facets = _facets_for(row, facets_cache)
    start = float(row.get("start") or 0)
    end = float(row.get("end") or 0)
    return {
        "fragment_id": row["fragment_id"],
        "source_video": row.get("source_video"),
        "video_id": row.get("video_id"),
        "video_title": row.get("video_title"),
        "speaker": row.get("speaker"),
        "speaker_attribution": row.get("speaker_attribution"),
        "text": row.get("text") or "",
        "summary": row.get("summary"),
        "start": start,
        "end": end,
        "duration": float(row.get("duration") or max(end - start, 0)),
        "primary_theme": row.get("primary_theme"),
        "primary_chapter": row.get("primary_chapter"),
        "theme_tags": row.get("theme_tags") or [],
        "chapter_tags": row.get("chapter_tags") or [],
        "clip_usage_candidates": row.get("clip_usage_candidates") or [],
        "quote_strength": float(row.get("quote_strength") or 0),
        "clarity_score": float(row.get("clarity_score") or 0),
        "year": facets.get("year"),
        "edition": facets.get("edition"),
        "session_type": facets.get("session_type"),
        "keywords": row.get("keywords") or [],
        "extra": {},
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping line {i}: {e}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import embeddings into ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection before importing")
    parser.add_argument("--skip-postgres", action="store_true",
                        help="Only write to Chroma, skip the companion Postgres fragments upsert")
    args = parser.parse_args()

    if not JSONL_PATH.exists():
        print(f"[error] JSONL not found: {JSONL_PATH}")
        sys.exit(1)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    import chromadb
    print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT} ...")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[reset] Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    print(f"Collection '{COLLECTION_NAME}': {existing_count} existing records")

    print(f"Loading JSONL from {JSONL_PATH} ...")
    t0 = time.time()
    rows = load_jsonl(JSONL_PATH)
    print(f"  Loaded {len(rows)} records in {time.time() - t0:.1f}s")

    # Skip fragments already in Chroma (incremental import)
    if existing_count > 0 and not args.reset:
        existing_ids = set(collection.get(include=[])["ids"])
        rows = [r for r in rows if r["fragment_id"] not in existing_ids]
        print(f"  {len(rows)} new records to import (skipping {existing_count} already present)")

    if not rows:
        print("Nothing to import.")
        return

    # Validate first record has an embedding
    sample = rows[0]
    if "embedding" not in sample or not sample["embedding"]:
        print("[error] First record has no embedding field — wrong file?")
        sys.exit(1)
    print(f"  Embedding dim: {len(sample['embedding'])} | model: {sample.get('embedding_model')}")

    # Batch import — Chroma gets vectors, Postgres gets the SQL-queryable row
    # (theme/chapter/speaker aggregates, filtered search).
    pg_repo = None
    facets_cache: dict = {}
    if not args.skip_postgres:
        from app.db.repositories.fragment_repository import FragmentRepository
        pg_repo = FragmentRepository()

    total = len(rows)
    imported = 0
    t1 = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        collection.add(
            ids=[r["fragment_id"] for r in batch],
            embeddings=[r["embedding"] for r in batch],
            documents=[r.get("embedding_text") or r.get("text") or "" for r in batch],
            metadatas=[sanitize_metadata(r) for r in batch],
        )
        if pg_repo is not None:
            try:
                pg_repo.upsert_many([to_postgres_row(r, facets_cache) for r in batch])
            except Exception as e:
                print(f"  [warn] Postgres upsert failed for batch starting at {i}: {e}")
        imported += len(batch)
        pct = imported / total * 100
        elapsed = time.time() - t1
        rate = imported / elapsed if elapsed > 0 else 0
        print(f"  [{pct:5.1f}%] {imported}/{total}  ({rate:.0f} rec/s)")

    elapsed_total = time.time() - t0
    final_count = collection.count()

    print()
    print(f"Done in {elapsed_total:.1f}s")
    print(f"Collection '{COLLECTION_NAME}' now has {final_count} records")

    # Quick sanity check — query with a random embedding from the first record
    print()
    print("Sanity check — querying with first record's own embedding ...")
    results = collection.query(
        query_embeddings=[rows[0]["embedding"]],
        n_results=3,
        include=["distances", "metadatas", "documents"],
    )
    for rank, (rid, dist, meta) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ), 1):
        print(f"  #{rank}  dist={dist:.4f}  id={rid[:60]}...")
        print(f"        theme={meta.get('primary_theme') or '(none)'}  chapter={meta.get('primary_chapter') or '(none)'}  speaker={meta.get('speaker') or '(none)'}")


if __name__ == "__main__":
    main()
