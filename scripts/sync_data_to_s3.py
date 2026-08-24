"""
Sync required pipeline data from local filesystem to S3.

Uploads:
  data/embeddings/            → raw/embeddings/            (classified JSONL + numpy matrix — stage 07)
  data/transcripts_cleaned/   → raw/transcripts_cleaned/   (stage 07b primary)
  data/transcripts/           → raw/transcripts/           (stage 07b fallback)
  data/fragments_classified/  → raw/fragments_classified/  (scripts/migrate_fragments_to_postgres.py source)
  data/metadata/speakers/     → raw/metadata/speakers/     (scripts/migrate_fragments_to_postgres.py source)
  data/raw/youtube/ai4ia_channel/ → raw/youtube/ai4ia_channel/  (video titles, optional — migration degrades gracefully without it)

Run once from the project root:
  python scripts/sync_data_to_s3.py [--dry-run] [--skip-existing]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.pipeline.s3_helpers import _s3, _bucket, list_keys  # noqa: E402

SYNC_MANIFEST = [
    # (local_dir, s3_prefix, suffixes_to_include)
    # ("data/embeddings",                     "raw/embeddings",             None),
    # ("data/transcripts_cleaned",            "raw/transcripts_cleaned",    (".json", ".txt", ".srt")),
    # ("data/transcripts",                    "raw/transcripts",            (".json", ".txt", ".srt")),
    ("data/fragments_classified",           "raw/fragments_classified",   (".json",)),
    ("data/metadata/speakers",              "raw/metadata/speakers",      (".json",)),
    ("data/raw/youtube/ai4ia_channel",      "raw/youtube/ai4ia_channel",  (".json",)),
]


def sync(dry_run: bool, skip_existing: bool) -> None:
    s3 = _s3()
    bucket = _bucket()

    for local_rel, s3_prefix, suffixes in SYNC_MANIFEST:
        local_dir = ROOT / local_rel
        if not local_dir.exists():
            print(f"[SKIP] {local_rel} — directory not found locally")
            continue

        files = sorted(f for f in local_dir.rglob("*") if f.is_file())
        if suffixes:
            files = [f for f in files if f.suffix in suffixes]

        if not files:
            print(f"[SKIP] {local_rel} — no matching files")
            continue

        if skip_existing:
            existing = set(list_keys(s3_prefix + "/"))
        else:
            existing = set()

        print(f"\n[SYNC] {local_rel} → s3://{s3_prefix}/  ({len(files)} files)")
        uploaded = skipped = 0
        for fpath in files:
            rel = fpath.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{rel}"
            if s3_key in existing:
                skipped += 1
                continue
            if dry_run:
                print(f"  [DRY] {s3_key}")
            else:
                s3.upload_file(str(fpath), bucket, s3_key)
            uploaded += 1

        if dry_run:
            print(f"  → would upload {uploaded}, skip {skipped}")
        else:
            print(f"  → uploaded {uploaded}, skipped {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync pipeline data to S3")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files already present in S3")
    args = parser.parse_args()

    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPLOAD'}, skip_existing={args.skip_existing}")
    sync(dry_run=args.dry_run, skip_existing=args.skip_existing)
    print("\nDone.")


if __name__ == "__main__":
    main()
