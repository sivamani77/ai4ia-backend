"""
Migrate legacy filesystem jobs into Supabase.

Run from the project root:
    python scripts/migrate_jobs_to_supabase.py

Sources:
    backend/runtime/jobs/*.json
    data/jobs/*/job_status.json + events.jsonl

Idempotent: existing jobs are updated in-place. Existing migrated events for a
job are replaced before events.jsonl is re-imported, so re-running is safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from backend/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.db import get_event_repository, get_supabase_client  # noqa: E402
from app.services.job_store import load_job, save_job  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_JOBS_DIR = PROJECT_ROOT / "backend" / "runtime" / "jobs"
VIDEO_JOBS_DIR = PROJECT_ROOT / "data" / "jobs"


def _job_exists(job_id: str) -> bool:
    try:
        load_job(job_id)
        return True
    except FileNotFoundError:
        return False


def _migrate_job_file(file: Path) -> tuple[str, str]:
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except Exception as exc:
        return "failed", f"FAILED (read)  {file}: {exc}"

    job_id = data.get("job_id") or data.get("id") or file.parent.name
    if not job_id:
        return "skipped", f"SKIPPED        {file}: no job_id field"

    try:
        existed = _job_exists(job_id)
        data.setdefault("job_id", job_id)
        save_job(job_id, data)
    except Exception as exc:
        return "failed", f"FAILED (write) {job_id}: {exc}"

    action = "UPDATED" if existed else "INSERTED"
    return "migrated", f"{action:<14} {job_id}"


def _iter_event_rows(events_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise ValueError(f"{events_path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{events_path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def _migrate_events(job_id: str, events_path: Path) -> tuple[int, str | None]:
    if not events_path.exists():
        return 0, None

    try:
        rows = _iter_event_rows(events_path)
    except Exception as exc:
        return 0, str(exc)

    client = get_supabase_client()
    event_repository = get_event_repository()

    try:
        client.table("job_events").delete().eq("job_id", job_id).execute()
        for row in rows:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            event_repository.add_event(
                {
                    "job_id": job_id,
                    "created_at": row.get("created_at"),
                    "event_type": row.get("event_type", "progress"),
                    "stage": row.get("stage", ""),
                    "payload": payload,
                }
            )
    except Exception as exc:
        return 0, str(exc)

    return len(rows), None


def main() -> None:
    migrated = skipped = failed = events_migrated = events_failed = 0

    runtime_files = sorted(RUNTIME_JOBS_DIR.glob("job_*.json"))
    if not runtime_files:
        print(f"No generic job files found in {RUNTIME_JOBS_DIR}")

    for file in runtime_files:
        status, message = _migrate_job_file(file)
        print(f"  {message}")
        if status == "migrated":
            migrated += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    video_status_files = sorted(VIDEO_JOBS_DIR.glob("*/job_status.json"))
    if not video_status_files:
        print(f"No video job status files found in {VIDEO_JOBS_DIR}")

    for file in video_status_files:
        status, message = _migrate_job_file(file)
        print(f"  {message}")
        if status == "migrated":
            migrated += 1
        elif status == "skipped":
            skipped += 1
            continue
        else:
            failed += 1
            continue

        job_id = file.parent.name
        imported, error = _migrate_events(job_id, file.parent / "events.jsonl")
        if error:
            events_failed += 1
            print(f"  FAILED (events) {job_id}: {error}")
        else:
            events_migrated += imported
            print(f"  EVENTS         {job_id}: {imported}")

    print()
    print(
        "Migrated jobs: "
        f"{migrated}  Skipped: {skipped}  Failed jobs: {failed}  "
        f"Migrated events: {events_migrated}  Event failures: {events_failed}"
    )


if __name__ == "__main__":
    main()
