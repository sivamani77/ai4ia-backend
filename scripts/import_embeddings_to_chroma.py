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
# Use the full classified JSONL (data/embeddings/) which has quote_strength,
# clarity_score, clip_usage_candidates — required for speaker clip selection.
JSONL_PATH = PROJECT_ROOT / "data" / "embeddings" / "classified_fragments_embeddings.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "fragments")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
BATCH_SIZE = 500


def sanitize_metadata(row: dict) -> dict:
    """Extract only scalar-safe metadata fields for ChromaDB."""
    def _str(v) -> str:
        return str(v) if v is not None else ""

    def _float(v) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _int(v) -> int:
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    # clip_usage_candidates is a list — store as comma-separated string
    clip_candidates = row.get("clip_usage_candidates") or []
    if isinstance(clip_candidates, list):
        clip_candidates_str = ",".join(clip_candidates)
    else:
        clip_candidates_str = _str(clip_candidates)

    return {
        "video_id":                  _str(row.get("video_id")),
        "source_video":              _str(row.get("source_video")),
        "fragment_index":            _int(row.get("fragment_index")),
        "start":                     _float(row.get("start")),
        "end":                       _float(row.get("end")),
        "duration":                  _float(row.get("duration")),
        "word_count":                _int(row.get("word_count")),
        "theme":                     _str(row.get("theme") or row.get("primary_theme")),
        "chapter":                   _str(row.get("chapter") or row.get("primary_chapter")),
        "storyline_role":            _str(row.get("storyline_role")),
        "summary":                   _str(row.get("summary")),
        "speaker":                   _str(row.get("speaker") or row.get("speaker_name")),
        "speaker_attribution":       _str(row.get("speaker_attribution")),
        "quote_strength":            _float(row.get("quote_strength")),
        "clarity_score":             _float(row.get("clarity_score")),
        "narration_support_score":   _float(row.get("narration_support_score")),
        "clip_usage_candidates":     clip_candidates_str,
        "classification_confidence": _float(row.get("classification_confidence")),
        "audio_quality_status":      _str(row.get("audio_quality_status")),
        "transcript_quality_score":  _float(row.get("transcript_quality_score")),
        "video_year":                _int(row.get("video_year")),
        "embedding_model":           _str(row.get("embedding_model")),
        "embedding_created_at":      _str(row.get("embedding_created_at")),
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

    # Batch import
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
        print(f"        theme={meta['theme'] or '(none)'}  chapter={meta['chapter'] or '(none)'}")


if __name__ == "__main__":
    main()
