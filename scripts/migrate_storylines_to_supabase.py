#!/usr/bin/env python3
"""One-time migration: load all data/storylines/storyline_*.json → Supabase storylines table.

Run from the repo root:
    python scripts/migrate_storylines_to_supabase.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.paths import STORYLINES_DIR
from app.db.repositories.storyline_repository import StorylineRepository


def main() -> None:
    files = sorted(STORYLINES_DIR.glob("storyline_*.json"))
    if not files:
        print("No storyline files found — nothing to migrate.")
        return

    repo = StorylineRepository()
    migrated = skipped = failed = 0

    for path in files:
        storyline_id = path.stem  # e.g. storyline_20260623_124212_validated
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP (bad JSON) {path.name}: {e}")
            failed += 1
            continue

        # Check if already present
        try:
            existing = repo.get(storyline_id)
        except Exception:
            existing = None

        if existing:
            print(f"  SKIP (exists) {storyline_id}")
            skipped += 1
            continue

        # Inject the id and validated flag
        data["id"] = storyline_id
        data["is_validated"] = "_validated" in storyline_id

        # Pull nested storyline fields to top level for repository mapping
        story = data.get("storyline") or {}
        data.setdefault("title", story.get("title"))
        data.setdefault("logline", story.get("logline"))
        data.setdefault("sections", story.get("sections") or [])

        try:
            repo.create(data)
            print(f"  OK {storyline_id}")
            migrated += 1
        except Exception as e:
            print(f"  FAIL {storyline_id}: {e}")
            failed += 1

    print(f"\nDone. Migrated={migrated}  Skipped={skipped}  Failed={failed}")


if __name__ == "__main__":
    main()
