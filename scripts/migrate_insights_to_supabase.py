#!/usr/bin/env python3
"""One-time migration: load all data/insights/insight_*.json → Supabase insights table.

Run from the repo root:
    python scripts/migrate_insights_to_supabase.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.paths import INSIGHTS_DIR
from app.db.repositories.insight_repository import InsightRepository


def main() -> None:
    files = sorted(INSIGHTS_DIR.glob("insight_*.json"))
    if not files:
        print("No insight files found — nothing to migrate.")
        return

    repo = InsightRepository()
    migrated = skipped = failed = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (bad JSON) {path.name}: {e}")
            failed += 1
            continue

        insight_id = data.get("insight_id") or path.stem
        if not insight_id:
            print(f"  SKIP (no id) {path.name}")
            failed += 1
            continue

        # Check if already present
        try:
            existing = repo.get(insight_id)
        except Exception:
            existing = None

        if existing:
            print(f"  SKIP (exists) {insight_id}")
            skipped += 1
            continue

        # Map insight_id → id for the DB column
        data["id"] = insight_id

        try:
            repo.create(data)
            print(f"  OK {insight_id}")
            migrated += 1
        except Exception as e:
            print(f"  FAIL {insight_id}: {e}")
            failed += 1

    print(f"\nDone. Migrated={migrated}  Skipped={skipped}  Failed={failed}")


if __name__ == "__main__":
    main()
