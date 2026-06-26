#!/usr/bin/env python3
"""
Standalone S3 connection test.

Reads credentials from .env at the repo root, then:
  1. Lists the first 10 objects in the bucket
  2. Uploads a small test file
  3. Downloads it back and verifies content
  4. Deletes the test file
Prints PASS / FAIL for each step.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env from repo root before importing anything else
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
env_file = repo_root / ".env"

if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file, override=False)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("FAIL  boto3 not installed — run: pip install boto3")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET   = os.environ.get("AI4IA_OBJECT_STORAGE_BUCKET", "")
KEY_ID   = os.environ.get("AI4IA_OBJECT_STORAGE_ACCESS_KEY", "")
SECRET   = os.environ.get("AI4IA_OBJECT_STORAGE_SECRET_KEY", "")
REGION   = os.environ.get("AI4IA_OBJECT_STORAGE_REGION", "us-east-1")
ENDPOINT = os.environ.get("AI4IA_OBJECT_STORAGE_ENDPOINT", "") or None

TEST_KEY     = "_s3_connection_test/probe.txt"
TEST_CONTENT = b"ai4ia-s3-probe-ok"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"{mark}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)


def _build_client():
    kwargs = {
        "aws_access_key_id":     KEY_ID or None,
        "aws_secret_access_key": SECRET or None,
        "region_name":           REGION or None,
    }
    if ENDPOINT:
        kwargs["endpoint_url"] = ENDPOINT
    return boto3.client("s3", **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BUCKET:
        print("FAIL  AI4IA_OBJECT_STORAGE_BUCKET is not set in .env")
        sys.exit(1)
    if not KEY_ID or KEY_ID.startswith("<"):
        print("FAIL  AI4IA_OBJECT_STORAGE_ACCESS_KEY is not set in .env")
        sys.exit(1)
    if not SECRET or SECRET.startswith("<"):
        print("FAIL  AI4IA_OBJECT_STORAGE_SECRET_KEY is not set in .env")
        sys.exit(1)

    print(f"Bucket   : {BUCKET}")
    print(f"Region   : {REGION}")
    print(f"Endpoint : {ENDPOINT or '(AWS default)'}")
    print("-" * 50)

    try:
        s3 = _build_client()
    except Exception as exc:
        print(f"FAIL  Could not create S3 client: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1 — list first 10 objects
    # ------------------------------------------------------------------
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=10)
        objects = resp.get("Contents", [])
        _status("List objects (first 10)", True,
                f"{len(objects)} object(s) returned")
        for obj in objects:
            print(f"      {obj['Key']}  ({obj['Size']} bytes)")
    except (ClientError, NoCredentialsError) as exc:
        _status("List objects (first 10)", False, str(exc))

    # ------------------------------------------------------------------
    # Step 2 — upload test file
    # ------------------------------------------------------------------
    try:
        s3.upload_fileobj(
            io.BytesIO(TEST_CONTENT),
            BUCKET,
            TEST_KEY,
            ExtraArgs={"ContentType": "text/plain"},
        )
        _status("Upload test file", True, TEST_KEY)
    except Exception as exc:
        _status("Upload test file", False, str(exc))
        sys.exit(1)   # can't continue without the uploaded object

    # ------------------------------------------------------------------
    # Step 3 — download and verify
    # ------------------------------------------------------------------
    try:
        buf = io.BytesIO()
        s3.download_fileobj(BUCKET, TEST_KEY, buf)
        downloaded = buf.getvalue()
        ok = downloaded == TEST_CONTENT
        _status(
            "Download & verify content",
            ok,
            "content matches" if ok else f"got {downloaded!r}, expected {TEST_CONTENT!r}",
        )
    except Exception as exc:
        _status("Download & verify content", False, str(exc))

    # ------------------------------------------------------------------
    # Step 4 — delete test file
    # ------------------------------------------------------------------
    try:
        s3.delete_object(Bucket=BUCKET, Key=TEST_KEY)
        _status("Delete test file", True, TEST_KEY)
    except Exception as exc:
        _status("Delete test file", False, str(exc))


if __name__ == "__main__":
    main()
