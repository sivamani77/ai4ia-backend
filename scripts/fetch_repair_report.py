#!/usr/bin/env python3
"""Download a speaker-repair report from S3 and print its summary. Read-only.

The cluster's repair (app/services/speaker_repair.py) writes each run's
report to s3://<bucket>/reports/data_repairs/<repair>/<mode>.json.

Usage (S3 keys from the local .env.example):

    set -a; . <(grep -E '^AWS_' .env.example); set +a
    python scripts/fetch_repair_report.py --mode dry_run
    python scripts/fetch_repair_report.py --mode apply --full
    python scripts/fetch_repair_report.py --mode apply --out /tmp/apply.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import boto3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["dry_run", "apply", "rollback", "cleanup_preview", "cleanup"], required=True)
    ap.add_argument("--repair", default="speaker_registry_v1")
    ap.add_argument("--bucket", default=os.environ.get("AI4IA_OBJECT_STORAGE_BUCKET") or "ai4ia-data")
    ap.add_argument("--full", action="store_true", help="also print every changed video")
    ap.add_argument("--out", help="save the whole report to this file")
    args = ap.parse_args()

    key = f"reports/data_repairs/{args.repair}/{args.mode}.json"
    try:
        body = boto3.client("s3").get_object(Bucket=args.bucket, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read s3://{args.bucket}/{key}: {exc}", file=sys.stderr)
        return 2
    report = json.loads(body)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(body)

    print(f"s3://{args.bucket}/{key}")
    print(f"status:   {report.get('status')}  (mode {report.get('mode')}, pod {report.get('pod')})")
    print(f"registry: {report.get('registry')}")
    print(f"time:     {report.get('started_at')} -> {report.get('finished_at')}")
    if report.get("error"):
        print(f"ERROR:    {report['error']}")
    for store in ("postgres", "chroma"):
        if store in report:
            print(f"{store:9} {report[store]}")
    for field in ("only_in_postgres", "only_in_chroma", "tables", "files", "restored", "cleanup", "cleanup_done"):
        if field in report:
            print(f"{field}: {report[field]}")
    for field in ("videos_not_in_registry", "speakers_not_in_registry"):
        if report.get(field):
            print(f"{field} ({len(report[field])}): {report[field]}")
    for check in report.get("checks", []):
        print(f"CHECK {check['name']}: {'PASS' if check['ok'] else 'FAIL'} ({check['detail']})")
    if args.full:
        for change in report.get("video_changes", []):
            print(f"  {change['video_id']}: {change['old']} -> {change['new_speaker'] or change['new_roster']!r} "
                  f"[{change['new_attribution']}] ({change['fragments']} fragments)")
    else:
        print(f"video_changes: {len(report.get('video_changes', []))} (use --full to list)")
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
