#!/usr/bin/env python3
"""One-time (and safe-to-rerun) backfill: load every fragment from the
ChromaDB `fragments` collection into the Postgres `fragments` table.

Reads Chroma, not local JSON: data/fragments_classified/*.classified.json
only covers 211 of the 309 videos in data/metadata/youtube/ai4ia_video_list.txt
— it's a stale/incomplete local checkout. Chroma has the full 309, kept
current by scripts/import_embeddings_to_chroma.py, so it's the more
complete and more current source for this backfill.

No local data/ or S3 sync required — this only needs network access to
Chroma and Postgres, which the API pod already has (same CHROMA_*/
DATABASE_URL config it uses to serve /archive/speakers and the dashboard).

Successor to the old migrate_fragments_to_supabase.py. That script only
inserted a fragment_id once and skipped it forever after — so a fragment
re-classified with corrected data (e.g. a speaker roster fix) would never
get its DB row updated. This script always upserts (ON CONFLICT DO UPDATE,
see FragmentRepository.upsert_many), so it's safe to re-run any time
Chroma picks up new or corrected fragments.

Run from the repo root, or in any pod with Chroma + Postgres access:
    python scripts/migrate_fragments_to_postgres.py
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.db.repositories.fragment_repository import FragmentRepository
from app.services.enrichment import derive_facets

PAGE_SIZE = 1000
BATCH_SIZE = 200


def _infer_topic(source_video: str) -> str:
    """source_video looks like '0182_IeXbR5L4EWE_Ashwini Kotaru - ...' —
    strip the numeric prefix and video_id to get a human title."""
    parts = (source_video or "").split("_", 2)
    return parts[2].strip() if len(parts) > 2 else (source_video or "")


def _get_chroma_collection():
    import chromadb
    from app.core.settings import settings

    if settings.chroma_mode == "http":
        kwargs = {"host": settings.chroma_host, "port": settings.chroma_port, "ssl": settings.chroma_ssl}
        if settings.chroma_auth_token:
            kwargs["headers"] = {"Authorization": f"Bearer {settings.chroma_auth_token}"}
        client = chromadb.HttpClient(**kwargs)
    else:
        from chromadb.config import Settings as ChromaSettings
        from app.core.paths import CHROMA_DIR
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
        )

    return client.get_or_create_collection(
        name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )


def _parse_json_field(value, default):
    if not value:
        return default
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed else default


def _to_row(fragment_id: str, meta: dict) -> dict:
    source_video = meta.get("source_video") or None
    video_title = _infer_topic(source_video) if source_video else None
    facets = derive_facets(video_title or "", source_video or "")

    start = float(meta.get("start") or 0)
    end = float(meta.get("end") or 0)

    return {
        "fragment_id": fragment_id,
        "source_video": source_video,
        "video_id": meta.get("video_id") or None,
        "video_title": video_title,
        "speaker": meta.get("speaker") or None,
        "speaker_attribution": meta.get("speaker_attribution") or None,
        "text": meta.get("text") or "",
        "summary": meta.get("summary") or None,
        "start": start,
        "end": end,
        "duration": float(meta.get("duration") or max(end - start, 0)),
        "primary_theme": meta.get("primary_theme") or None,
        "primary_chapter": meta.get("primary_chapter") or None,
        "theme_tags": _parse_json_field(meta.get("theme_tags"), []),
        "chapter_tags": _parse_json_field(meta.get("chapter_tags"), []),
        "clip_usage_candidates": _parse_json_field(meta.get("clip_usage_candidates"), []),
        "quote_strength": float(meta.get("quote_strength") or 0),
        "clarity_score": float(meta.get("clarity_score") or 0),
        "year": facets.get("year"),
        "edition": facets.get("edition"),
        "session_type": facets.get("session_type"),
        "keywords": _parse_json_field(meta.get("keywords"), []),
        "extra": {},
    }


def main() -> None:
    print("Connecting to ChromaDB ...")
    collection = _get_chroma_collection()
    total = collection.count()
    print(f"Collection has {total} fragments.")
    if not total:
        print("Nothing to migrate.")
        return

    repo = FragmentRepository()
    total_upserted = total_failed = 0
    offset = 0

    while offset < total:
        page = collection.get(limit=PAGE_SIZE, offset=offset, include=["metadatas"])
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        if not ids:
            break

        rows = []
        for fid, meta in zip(ids, metadatas):
            try:
                rows.append(_to_row(fid, meta or {}))
            except Exception as e:
                print(f"    FAIL build row {fid}: {e}")
                total_failed += 1

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i: i + BATCH_SIZE]
            try:
                repo.upsert_many(batch)
                total_upserted += len(batch)
            except Exception as e:
                print(f"    FAIL batch upsert ({len(batch)} rows): {e}")
                total_failed += len(batch)

        offset += len(ids)
        print(f"  [{min(offset, total)}/{total}] upserted so far: {total_upserted}")

    print(f"\nDone. Total upserted={total_upserted}  failed={total_failed}")


if __name__ == "__main__":
    main()
