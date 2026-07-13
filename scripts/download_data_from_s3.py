"""
Download S3 assets to a local directory (e.g. the EBS-backed data dir on a VM).

Standalone script — only needs boto3 + python-dotenv, no `app` package import,
so it also works when copied on its own to a VM (no full repo checkout needed).

Mirrors S3 object keys 1:1 under the target directory, e.g.
  s3://<bucket>/raw/embeddings/foo.json  →  <target>/raw/embeddings/foo.json

Credentials/bucket are read from the usual AI4IA_OBJECT_STORAGE_* env vars,
loaded from a .env file (default: .env next to this script, override with
--env-file).

Usage:
  python download_data_from_s3.py
  python download_data_from_s3.py --prefix raw/embeddings
  python download_data_from_s3.py --target /home/ubuntu/ai4ia/data --dry-run
  python download_data_from_s3.py --skip-existing
  python download_data_from_s3.py --env-file /home/ash/ai4ia-optm/ai4ia-backend/.env
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_TARGET = "/home/ubuntu/ai4ia/data"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


def s3_client():
    import boto3

    kwargs = {
        "aws_access_key_id": os.environ.get("AI4IA_OBJECT_STORAGE_ACCESS_KEY") or None,
        "aws_secret_access_key": os.environ.get("AI4IA_OBJECT_STORAGE_SECRET_KEY") or None,
        "region_name": os.environ.get("AI4IA_OBJECT_STORAGE_REGION") or None,
    }
    endpoint = os.environ.get("AI4IA_OBJECT_STORAGE_ENDPOINT")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def list_keys(s3, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_all(target: Path, prefix: str, dry_run: bool, skip_existing: bool) -> None:
    s3 = s3_client()
    bucket = os.environ.get("AI4IA_OBJECT_STORAGE_BUCKET")
    if not bucket:
        raise SystemExit("AI4IA_OBJECT_STORAGE_BUCKET is not set (check your --env-file)")

    keys = [k for k in list_keys(s3, bucket, prefix) if not k.endswith("/")]
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return

    print(f"Found {len(keys)} objects under s3://{bucket}/{prefix}")
    print(f"Target: {target}\n")

    downloaded = skipped = 0
    for key in keys:
        rel = key[len(prefix):].lstrip("/") if prefix else key
        local_path = target / rel

        if skip_existing and local_path.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY] {key} → {local_path}")
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
        downloaded += 1

    verb = "would download" if dry_run else "downloaded"
    print(f"\n→ {verb} {downloaded}, skipped {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download S3 assets to a local directory")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Local directory to download into (default: {DEFAULT_TARGET})")
    parser.add_argument("--prefix", default="", help="Only download keys under this S3 prefix (default: entire bucket)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without downloading")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist locally")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help=f"Path to .env with AI4IA_OBJECT_STORAGE_* vars (default: {DEFAULT_ENV_FILE})")
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser()
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print(f"Warning: env file not found at {env_file} — relying on already-exported env vars")

    target = Path(args.target).expanduser().resolve()
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {'DRY RUN' if args.dry_run else 'DOWNLOAD'}, skip_existing={args.skip_existing}")
    download_all(target, args.prefix, dry_run=args.dry_run, skip_existing=args.skip_existing)
    print("\nDone.")


if __name__ == "__main__":
    main()
