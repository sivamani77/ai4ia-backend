#!/usr/bin/env python3
"""One-time (and safe-to-rerun) repair: re-sync the Postgres `videos` table
from S3, using the fixed youtube-id parser.

Why this is needed: the videos table's row-per-video "video_id" was
originally computed by archive_service.py's own copy of the naive
stem.split("_") parser, which mis-parses the 42 of 309 videos whose
YouTube id itself contains an underscore (e.g. "9_sJS37LjJ4" truncated to
"9") — scripts/reconcile_archive_stores.py caught this: those 42 videos
were missing from the videos table under their correct id (present under
a wrong/truncated one instead, some of which collided and overwrote each
other, which is why the table had 302 rows instead of 309).

This ONLY touches the `videos` table (a lookup/display cache this session
added — title, duration, s3_key for the Archive UI's video list). It does
not touch fragments, Chroma, S3, or speakers — none of that was ever
wrong; re-running the ingestion pipeline is not needed for this.

Run from the repo root, or in any pod with S3 + Postgres access:
    python scripts/sync_videos_registry.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.db.repositories.video_repository import VideoRepository
from app.services.archive_service import _list_videos_from_s3


def main() -> None:
    print("Scanning S3 raw video prefix ...")
    videos = _list_videos_from_s3()
    print(f"Found {len(videos)} videos in S3.")
    if not videos:
        print("Nothing to sync.")
        return

    repo = VideoRepository()
    repo.upsert_many(videos)
    print(f"Upserted {len(videos)} rows into the videos table (correct ids now).")

    correct_ids = [v["video_id"] for v in videos if v.get("video_id")]
    removed = repo.delete_not_in(correct_ids)
    print(f"Removed {removed} stale/wrongly-keyed row(s) left over from the old parser.")

    total = repo.count()
    print(f"\nDone. videos table now has {total} rows (expected {len(videos)}).")


if __name__ == "__main__":
    main()
