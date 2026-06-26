"""
Re-embed all fragments from Supabase with rich structured embedding_text.

The original embeddings used empty Theme/Chapter/Keywords fields.
This script fetches full classified metadata from Supabase and re-embeds
each fragment as:
    search_document: Theme: <primary_theme> Chapter: <primary_chapter>
    Speaker: <speaker> Keywords: <kw1, kw2, ...>
    Summary: <summary> Transcript: <text[:600]>

Usage (on worker VM or any machine with Ollama + Supabase access):
    python scripts/reembed_fragments.py [--reset] [--dry-run] [--batch-size N]

Options:
    --reset       Drop and recreate the ChromaDB collection before importing
    --dry-run     Build embedding_text and show samples, do not call Ollama
    --batch-size  Number of fragments to embed per batch (default: 50)
    --resume      Skip fragments already present in ChromaDB (default: True)

Outputs:
    data/embeddings_cleaned/classified_fragments_embeddings.jsonl  (updated in-place)
    ChromaDB collection 'fragments' (upserted)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import os
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reembed")

JSONL_PATH = PROJECT_ROOT / "data" / "embeddings_cleaned" / "classified_fragments_embeddings.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "fragments"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
OLLAMA_BASE_URL = None  # set from settings
BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Embedding text builder
# ---------------------------------------------------------------------------

def build_embedding_text(frag: dict) -> str:
    """Build a rich embedding text using all available metadata."""
    theme = frag.get("primary_theme") or ""
    chapter = frag.get("primary_chapter") or ""
    speaker = frag.get("speaker") or ""
    keywords = frag.get("keywords") or []
    summary = frag.get("summary") or ""
    text = frag.get("text") or ""

    kw_str = ", ".join(keywords[:20]) if keywords else ""

    parts = ["search_document:"]
    if theme:
        parts.append(f"Theme: {theme}")
    if chapter:
        parts.append(f"Chapter: {chapter}")
    if speaker:
        parts.append(f"Speaker: {speaker}")
    if kw_str:
        parts.append(f"Keywords: {kw_str}")
    if summary:
        parts.append(f"Summary: {summary[:200]}")
    if text:
        parts.append(f"Transcript: {text[:600]}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Ollama embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], base_url: str) -> list[list[float]]:
    """Embed a batch of texts via Ollama in one call. Returns list of 768-dim vectors."""
    import requests

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=60 + len(texts) * 2,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama returns {"embeddings": [[...], [...]]} for batch input
            embeddings = data.get("embeddings")
            if embeddings and isinstance(embeddings[0], list):
                return embeddings
            # Fallback: single embedding returned (shouldn't happen for array input)
            if "embedding" in data:
                return [data["embedding"]]
            raise RuntimeError(f"Unexpected response keys: {list(data.keys())}")
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Ollama batch embed failed after 3 attempts: {exc}") from exc
            time.sleep(2)
    return []  # unreachable


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_chroma_collection(reset: bool, chroma_mode: str, chroma_host: str, chroma_port: int,
                           chroma_ssl: bool, chroma_auth_token: str):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if chroma_mode == "http":
        kwargs = {"host": chroma_host, "port": chroma_port, "ssl": chroma_ssl}
        if chroma_auth_token:
            kwargs["headers"] = {"Authorization": f"Bearer {chroma_auth_token}"}
        client = chromadb.HttpClient(**kwargs)
    else:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Dropped existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def sanitize_metadata(frag: dict) -> dict:
    def _s(v): return str(v) if v is not None else ""
    def _f(v):
        try: return float(v) if v is not None else 0.0
        except: return 0.0
    def _i(v):
        try: return int(v) if v is not None else 0
        except: return 0

    return {
        "video_id":                  _s(frag.get("video_id")),
        "source_video":              _s(frag.get("source_video")),
        "fragment_index":            _i((frag.get("extra") or {}).get("fragment_index")),
        "start":                     _f(frag.get("start")),
        "end":                       _f(frag.get("end")),
        "duration":                  _f(frag.get("duration")),
        "word_count":                _i((frag.get("extra") or {}).get("word_count")),
        "theme":                     _s(frag.get("primary_theme")),
        "chapter":                   _s(frag.get("primary_chapter")),
        "speaker":                   _s(frag.get("speaker")),
        "storyline_role":            "",
        "summary":                   _s(frag.get("summary")),
        "classification_confidence": _f((frag.get("extra") or {}).get("classification_confidence")),
        "audio_quality_status":      "",
        "transcript_quality_score":  0.0,
        "embedding_model":           EMBEDDING_MODEL,
        "embedding_created_at":      "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-embed fragments with rich structured text")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate ChromaDB collection")
    parser.add_argument("--dry-run", action="store_true", help="Show sample embedding texts, no Ollama calls")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-resume", action="store_true", help="Re-embed even if already in ChromaDB")
    args = parser.parse_args()

    # Load settings
    from app.core.settings import settings
    ollama_url = settings.ollama_base_url
    chroma_mode = settings.chroma_mode
    chroma_host = settings.chroma_host
    chroma_port = settings.chroma_port
    chroma_ssl = settings.chroma_ssl
    chroma_auth_token = settings.chroma_auth_token

    log.info("Ollama: %s  |  ChromaDB mode: %s", ollama_url, chroma_mode)

    # Fetch all fragments from Supabase
    log.info("Fetching all fragments from Supabase...")
    from app.db.repositories.fragment_repository import FragmentRepository
    repo = FragmentRepository()
    total = repo.count()
    log.info("Total fragments in Supabase: %d", total)

    # Fetch all fragments (list_all paginates internally in 1000-row pages)
    all_frags = [dict(r) for r in repo.list_all(limit=total + 100)]

    log.info("Loaded %d fragments", len(all_frags))

    if args.dry_run:
        log.info("DRY RUN — showing 5 sample embedding texts:")
        for frag in all_frags[:5]:
            et = build_embedding_text(frag)
            log.info("  [%s] %s", frag.get("fragment_id", "?")[:40], et[:120])
        return

    # Connect to ChromaDB
    log.info("Connecting to ChromaDB...")
    collection = get_chroma_collection(
        reset=args.reset,
        chroma_mode=chroma_mode,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_ssl=chroma_ssl,
        chroma_auth_token=chroma_auth_token,
    )
    existing_count = collection.count()
    log.info("ChromaDB collection '%s' has %d existing entries", COLLECTION_NAME, existing_count)

    # Get existing IDs if resuming
    existing_ids: set[str] = set()
    if not args.no_resume and existing_count > 0 and not args.reset:
        log.info("Loading existing ChromaDB IDs to skip already-embedded fragments...")
        offset_c = 0
        while True:
            batch = collection.get(limit=5000, offset=offset_c, include=[])
            ids = batch.get("ids", [])
            if not ids:
                break
            existing_ids.update(ids)
            offset_c += len(ids)
            if len(ids) < 5000:
                break
        log.info("Found %d already embedded, will skip them", len(existing_ids))

    # Build work list
    to_embed = [f for f in all_frags if f.get("fragment_id") not in existing_ids]
    log.info("Fragments to embed: %d (skipping %d)", len(to_embed), len(all_frags) - len(to_embed))

    if not to_embed:
        log.info("Nothing to do — all fragments already embedded.")
        return

    # Load existing JSONL for update
    existing_jsonl: dict[str, dict] = {}
    if JSONL_PATH.exists():
        with JSONL_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing_jsonl[rec["fragment_id"]] = rec
                except Exception:
                    pass
    log.info("Loaded %d existing JSONL records", len(existing_jsonl))

    # Process in batches
    batch_size = args.batch_size
    done = 0
    errors = 0
    t0 = time.monotonic()

    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        frag_ids = [f["fragment_id"] for f in batch]
        embedding_texts = [build_embedding_text(f) for f in batch]

        try:
            vectors = embed_texts(embedding_texts, ollama_url)
        except Exception as exc:
            log.error("Batch %d failed: %s", i // batch_size, exc)
            errors += len(batch)
            continue

        # Upsert into ChromaDB
        metadatas = [sanitize_metadata(f) for f in batch]
        documents = embedding_texts

        try:
            collection.upsert(
                ids=frag_ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            log.error("ChromaDB upsert failed for batch %d: %s", i // batch_size, exc)
            errors += len(batch)
            continue

        # Update JSONL records
        for frag, vec, et in zip(batch, vectors, embedding_texts):
            fid = frag["fragment_id"]
            rec = existing_jsonl.get(fid, {})
            rec.update({
                "fragment_id": fid,
                "source_video": frag.get("source_video", ""),
                "video_id": frag.get("video_id", ""),
                "text": frag.get("text", ""),
                "primary_theme": frag.get("primary_theme"),
                "primary_chapter": frag.get("primary_chapter"),
                "speaker": frag.get("speaker"),
                "keywords": frag.get("keywords") or [],
                "summary": frag.get("summary"),
                "embedding_text": et,
                "embedding_model": EMBEDDING_MODEL,
                "embedding": vec,
            })
            existing_jsonl[fid] = rec

        done += len(batch)
        elapsed = time.monotonic() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(to_embed) - done) / rate if rate > 0 else 0
        log.info(
            "Progress: %d/%d  errors=%d  %.1f/s  ETA %.0fs",
            done, len(to_embed), errors, rate, eta
        )

    # Write updated JSONL
    log.info("Writing updated JSONL to %s ...", JSONL_PATH)
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for rec in existing_jsonl.values():
            # Don't write the embedding vector to JSONL (large, not needed)
            out = {k: v for k, v in rec.items() if k != "embedding"}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    final_count = collection.count()
    log.info(
        "Done. Embedded: %d  Errors: %d  ChromaDB total: %d",
        done, errors, final_count,
    )
    if errors:
        log.warning("%d fragments failed to embed. Re-run without --no-resume to retry.", errors)


if __name__ == "__main__":
    main()
