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
from app.pipeline import s3_helpers
from app.services.enrichment import derive_facets
from app.services.speaker_aliases import canonicalize as canonicalize_speaker
from app.services.video_naming import parse_source_video

PAGE_SIZE = 1000
BATCH_SIZE = 200
_ROSTER_SCRATCH_NAME = "migrate_fragments_to_postgres_rosters"


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


_roster_cache: dict[str, str | None] = {}


def _lookup_roster_speaker(video_id: str | None) -> str | None:
    """Fallback for fragments Chroma never got a speaker for. Root cause
    (confirmed for 8 videos in the 309-video catalog): 05_classify_fragments.py
    ran before that video's roster (04b_extract_speakers.py output) existed,
    so its fallback-fill found nothing — and nothing ever revisits a video
    once its roster shows up later. The roster itself is usually correct
    (LLM extraction succeeded even when the deterministic title-hint didn't),
    so re-check it here rather than leaving the fragment speakerless.
    Lazy + cached: only fetched for fragments that actually need it."""
    if not video_id:
        return None
    if video_id in _roster_cache:
        return _roster_cache[video_id]

    speaker = None
    if s3_helpers.use_s3():
        key = f"raw/metadata/speakers/{video_id}.json"
        if s3_helpers.key_exists(key):
            local = s3_helpers.tmp_dir(_ROSTER_SCRATCH_NAME) / f"{video_id}.json"
            try:
                s3_helpers.download(key, local)
                roster = json.loads(local.read_text(encoding="utf-8"))
                speaker = roster.get("primary_speaker") or None
            except Exception:
                speaker = None
    else:
        from app.core.paths import SPEAKERS_DIR
        local = SPEAKERS_DIR / f"{video_id}.json"
        if local.exists():
            try:
                roster = json.loads(local.read_text(encoding="utf-8"))
                speaker = roster.get("primary_speaker") or None
            except Exception:
                speaker = None

    _roster_cache[video_id] = speaker
    return speaker


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
    # Prefer parsing video_id/title from source_video ourselves — Chroma's
    # own video_id field was computed with the same underscore-splitting
    # bug (see parse_source_video's docstring) and is wrong for ~42
    # videos; source_video itself is reliable, so re-derive from it.
    parsed_video_id, video_title = parse_source_video(source_video or "")
    video_id = parsed_video_id or meta.get("video_id") or None
    facets = derive_facets(video_title or "", source_video or "")

    start = float(meta.get("start") or 0)
    end = float(meta.get("end") or 0)

    speaker = meta.get("speaker") or None
    speaker_attribution = meta.get("speaker_attribution") or None
    if not speaker:
        roster_speaker = _lookup_roster_speaker(video_id)
        if roster_speaker:
            speaker = roster_speaker
            speaker_attribution = speaker_attribution or "video-level"
    speaker = canonicalize_speaker(speaker)

    return {
        "fragment_id": fragment_id,
        "source_video": source_video,
        "video_id": video_id,
        "video_title": video_title or None,
        "speaker": speaker,
        "speaker_attribution": speaker_attribution,
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

    if s3_helpers.use_s3():
        s3_helpers.cleanup_tmp_dir(_ROSTER_SCRATCH_NAME)


if __name__ == "__main__":
    main()
