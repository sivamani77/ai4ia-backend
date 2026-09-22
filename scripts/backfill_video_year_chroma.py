"""Backfill video_year / video_upload_date onto the ChromaDB fragments.

Every fragment in the collection carries video_year="" and
video_upload_date="" — 5112 of 5112 as of this writing. Two things depend on
those fields and are silently inert without them:

- choose_video_mode's "chronological_relay" branch needs >= 2 distinct
  non-null years and can never fire, so the archive has no temporal mode;
- the per-year diversity cap in select_balanced_candidates buckets every
  candidate under one key, so it never actually caps anything.

Why they are empty: these fragments were produced by
04_regenerate_fragments_from_clean_transcripts.py, which never stamped the
field. The current ingest path does not have this gap —
ingest_job_runner._stage_create_fragments runs 04_create_fragments.py and
downloads {stem}.info.json specifically so make_fragment can read
upload_date — so this is a one-off repair of historical rows, not a
recurring sync.

The year comes from each video's YouTube .info.json upload_date, read from
the local raw dir when present and from S3 otherwise. Fragments are matched
to their video through video_naming.parse_source_video, never split("_"):
42 of the 309 videos have a YouTube id containing an underscore, and
splitting truncates their id so they would be left unmatched. Measured
coverage with the correct parser is 309/309.

Which collection this writes to is decided entirely by CHROMA_HOST /
CHROMA_PORT / CHROMA_COLLECTION, so run it against the deployed host to
repair deployed data and against localhost to rehearse.

Usage:

    # report what would change, touching nothing (default)
    python scripts/backfill_video_year_chroma.py

    # write the metadata back
    python scripts/backfill_video_year_chroma.py --apply

    # also set the fragments.year column in Postgres
    python scripts/backfill_video_year_chroma.py --apply --postgres
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.paths import RAW_VIDEO_DIR  # noqa: E402
from app.services.video_naming import parse_source_video  # noqa: E402

_S3_RAW_PREFIX = "raw/youtube/ai4ia_channel/"

# Chroma metadata cannot hold None, so an unknown year stays "" — the same
# shape the field already has, which every reader already tolerates.
_EMPTY = ""


def log(message: str) -> None:
    print(message, flush=True)


def chroma_collection():
    import chromadb

    host = os.environ.get("CHROMA_HOST", "chromadb")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "fragments")
    auth_token = os.environ.get("CHROMA_AUTH_TOKEN", "")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
    log(f"ChromaDB target: {host}:{port}/{collection_name}")
    client = chromadb.HttpClient(host=host, port=port, headers=headers)
    return client.get_collection(collection_name)


def upload_dates_from_local() -> dict[str, str]:
    """{youtube_id: upload_date} from .info.json files in the raw video dir."""
    if not RAW_VIDEO_DIR.exists():
        return {}
    found: dict[str, str] = {}
    for path in RAW_VIDEO_DIR.glob("*.info.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        video_id, upload_date = info.get("id"), info.get("upload_date")
        if video_id and upload_date:
            found[str(video_id)] = str(upload_date)
    return found


def upload_dates_from_s3() -> dict[str, str]:
    """Same map, read from the raw/ prefix in object storage."""
    import boto3

    from app.services.storage_service import (
        _OBJECT_STORAGE_ACCESS_KEY,
        _OBJECT_STORAGE_BUCKET,
        _OBJECT_STORAGE_ENDPOINT,
        _OBJECT_STORAGE_REGION,
        _OBJECT_STORAGE_SECRET_KEY,
    )

    kwargs: dict = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
    s3 = boto3.client("s3", **kwargs)

    found: dict[str, str] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_OBJECT_STORAGE_BUCKET, Prefix=_S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".info.json"):
                continue
            try:
                body = s3.get_object(Bucket=_OBJECT_STORAGE_BUCKET, Key=key)["Body"].read()
                info = json.loads(body.decode("utf-8"))
            except Exception:
                continue
            video_id, upload_date = info.get("id"), info.get("upload_date")
            if video_id and upload_date:
                found[str(video_id)] = str(upload_date)
    return found


def load_upload_dates() -> dict[str, str]:
    local = upload_dates_from_local()
    if local:
        log(f"Read {len(local)} upload dates from {RAW_VIDEO_DIR}")
        return local
    log("No local .info.json files — reading from object storage")
    remote = upload_dates_from_s3()
    log(f"Read {len(remote)} upload dates from s3://{_S3_RAW_PREFIX}")
    return remote


def year_of(upload_date: str) -> int | None:
    return int(upload_date[:4]) if upload_date and len(upload_date) >= 4 else None


def plan_updates(
    records: list[tuple[str, dict]],
    upload_dates: dict[str, str],
) -> tuple[list[str], list[dict], set[str], int, Counter]:
    """Decide what to write without writing anything.

    Returns (ids, metadatas, unmatched_source_videos, already_correct, years).
    """
    pending_ids: list[str] = []
    pending_metadatas: list[dict] = []
    unmatched: set[str] = set()
    already_correct = 0
    years: Counter = Counter()

    for fragment_id, metadata in records:
        metadata = dict(metadata or {})
        source_video = str(metadata.get("source_video") or "")
        video_id, _ = parse_source_video(source_video)
        upload_date = upload_dates.get(video_id or "")
        if not upload_date:
            if source_video:
                unmatched.add(source_video)
            continue

        year = year_of(upload_date)
        if (str(metadata.get("video_year") or "") == str(year if year is not None else _EMPTY)
                and str(metadata.get("video_upload_date") or "") == upload_date):
            already_correct += 1
            continue

        # Chroma's update REPLACES a document's metadata rather than merging
        # it, so the full existing dict is carried over with the two fields
        # set. Passing only the changed keys would drop source_video, speaker,
        # primary_theme and everything else retrieval depends on.
        metadata["video_year"] = year if year is not None else _EMPTY
        metadata["video_upload_date"] = upload_date
        pending_ids.append(fragment_id)
        pending_metadatas.append(metadata)
        years[year] += 1

    return pending_ids, pending_metadatas, unmatched, already_correct, years


def load_all_metadata(collection) -> list[tuple[str, dict]]:
    total = collection.count()
    log(f"Collection holds {total} fragments")
    records: list[tuple[str, dict]] = []
    offset = 0
    while offset < total:
        page = collection.get(limit=500, offset=offset, include=["metadatas"])
        records.extend(zip(page["ids"], page["metadatas"]))
        offset += 500
    log(f"Loaded metadata for {len(records)} fragments")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the metadata back. Without it the script only reports.",
    )
    parser.add_argument(
        "--postgres", action="store_true",
        help="Also set the fragments.year column in Postgres.",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    upload_dates = load_upload_dates()
    if not upload_dates:
        raise SystemExit("No upload dates available from either source — nothing to backfill.")

    collection = chroma_collection()
    records = load_all_metadata(collection)
    pending_ids, pending_metadatas, unmatched, already_correct, years = plan_updates(
        records, upload_dates
    )

    log("")
    log(f"  fragments to update   : {len(pending_ids)}")
    log(f"  already correct       : {already_correct}")
    log(f"  unmatched videos      : {len(unmatched)}")
    for source_video in sorted(unmatched)[:10]:
        log(f"      {source_video}")
    log(f"  resulting year spread : "
        f"{dict(sorted(years.items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    if not args.apply:
        log("\nDry run — nothing written. Re-run with --apply to write.")
        return
    if not pending_ids:
        log("\nNothing to write.")
        return

    for start in range(0, len(pending_ids), args.batch_size):
        ids = pending_ids[start: start + args.batch_size]
        metadatas = pending_metadatas[start: start + args.batch_size]
        collection.update(ids=ids, metadatas=metadatas)
        log(f"  updated {min(start + len(ids), len(pending_ids))}/{len(pending_ids)}")
    log("ChromaDB metadata updated.")

    if args.postgres:
        from app.db.postgres import pg_execute

        updated = 0
        for fragment_id, metadata in zip(pending_ids, pending_metadatas):
            year = metadata.get("video_year")
            if year == _EMPTY:
                continue
            pg_execute(
                "update fragments set year = %(year)s where fragment_id = %(fragment_id)s",
                {"year": int(year), "fragment_id": fragment_id},
            )
            updated += 1
        log(f"Postgres fragments.year updated for {updated} rows.")


if __name__ == "__main__":
    main()
