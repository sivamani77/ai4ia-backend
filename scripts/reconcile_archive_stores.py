#!/usr/bin/env python3
"""Compare the three independent stores behind the Archive feature — S3
raw video files, the Postgres `fragments`/`videos` tables, and ChromaDB —
and report where they disagree.

Nothing in the running system currently notices when these drift (see
this session's investigation of missing speakers/videos in the Archive
UI — every gap found was discovered by manually cross-querying these
three stores by hand). This script automates that cross-query as a
report; it does not fix anything.

Run from the repo root, or in any pod with S3 + Postgres + Chroma access:
    python scripts/reconcile_archive_stores.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.db.postgres import pg_fetch_all
from app.pipeline import s3_helpers
from app.services.video_naming import parse_source_video

_S3_RAW_PREFIX = "raw/youtube/ai4ia_channel/"


def _s3_video_ids() -> set[str]:
    ids: set[str] = set()
    for key in s3_helpers.list_keys(_S3_RAW_PREFIX):
        if not key.endswith(".mp4"):
            continue
        stem = Path(key).stem
        video_id, _ = parse_source_video(stem)
        if video_id:
            ids.add(video_id)
    return ids


def _postgres_videos_table_ids() -> set[str]:
    rows = pg_fetch_all("select video_id from videos")
    return {r["video_id"] for r in rows if r.get("video_id")}


def _postgres_fragment_video_ids() -> set[str]:
    rows = pg_fetch_all("select distinct video_id from fragments where video_id is not null")
    return {r["video_id"] for r in rows if r.get("video_id")}


def _chroma_video_ids() -> set[str]:
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
        client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False))

    collection = client.get_or_create_collection(name=settings.chroma_collection, metadata={"hnsw:space": "cosine"})
    total = collection.count()

    ids: set[str] = set()
    offset = 0
    while offset < total:
        page = collection.get(limit=1000, offset=offset, include=["metadatas"])
        got = page.get("ids") or []
        for meta in page.get("metadatas") or []:
            source_video = meta.get("source_video") or ""
            video_id, _ = parse_source_video(source_video)
            if video_id:
                ids.add(video_id)
        offset += len(got)
        if not got:
            break
    return ids


def _report(label: str, ids: set[str]) -> None:
    print(f"  {label}: {len(ids)}")
    for vid in sorted(ids)[:20]:
        print(f"    {vid}")
    if len(ids) > 20:
        print(f"    ... and {len(ids) - 20} more")


def main() -> None:
    print("Gathering video ids from each store ...")
    s3_ids = _s3_video_ids()
    print(f"  S3 raw videos: {len(s3_ids)}")
    pg_videos_ids = _postgres_videos_table_ids()
    print(f"  Postgres videos table: {len(pg_videos_ids)}")
    pg_fragment_ids = _postgres_fragment_video_ids()
    print(f"  Postgres fragments (distinct video_id): {len(pg_fragment_ids)}")
    chroma_ids = _chroma_video_ids()
    print(f"  Chroma (distinct video_id): {len(chroma_ids)}")

    print("\n=== Drift report ===\n")

    _report("In S3 but no Postgres fragments (downloaded, never classified/backfilled)",
            s3_ids - pg_fragment_ids)
    _report("In Postgres fragments but not in S3 (fragments exist, raw file missing)",
            pg_fragment_ids - s3_ids)
    _report("In Chroma but not in Postgres fragments (backfill hasn't picked these up)",
            chroma_ids - pg_fragment_ids)
    _report("In Postgres fragments but not in Chroma (unexpected — check for stale rows)",
            pg_fragment_ids - chroma_ids)
    _report("In S3 but not in Postgres videos table (registry sync hasn't run/seen these)",
            s3_ids - pg_videos_ids)

    if s3_ids == pg_fragment_ids == chroma_ids:
        print("\nAll three stores agree on the same video set. No drift.")


if __name__ == "__main__":
    main()
