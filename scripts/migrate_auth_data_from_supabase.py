#!/usr/bin/env python3
"""Manual CLI for the one-time copy of the auth tables from the Supabase auth
project into Postgres. The migration itself now also runs automatically,
once, on API startup (see app.services.auth_data_migration.run_once, wired
into app.main's lifespan) — use this script for a dry run beforehand, or to
check/force it manually.

Requires AUTH_SUPABASE_URL and AUTH_SUPABASE_SERVICE_KEY (the Supabase auth
project, read-only) and DATABASE_URL (the destination Postgres).

Usage
-----
    # See what would be copied, change nothing (always do this first):
    python scripts/migrate_auth_data_from_supabase.py --dry-run

    # Perform the copy:
    python scripts/migrate_auth_data_from_supabase.py

    # Re-check counts on both sides after the fact:
    python scripts/migrate_auth_data_from_supabase.py --verify-only

    # Copy what changed in Supabase since the first copy (the API also runs
    # this on startup while AUTH_SUPABASE_CATCHUP is on):
    python scripts/migrate_auth_data_from_supabase.py --catch-up
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.settings import settings  # noqa: E402
from app.db.schema import ensure_auth_schema  # noqa: E402
from app.services.auth_data_migration import (  # noqa: E402
    already_completed,
    catch_up_once,
    mark_completed,
    migrate,
    verify,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("migrate_auth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Read and count only; write nothing to Postgres.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Compare row counts on both sides and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Run again even though the completion marker is present. "
                             "Safe (inserts are ON CONFLICT DO NOTHING) but normally unnecessary.")
    parser.add_argument("--catch-up", action="store_true",
                        help="Copy rows that changed in Supabase since the first copy "
                             "(the same pass the API runs on startup), then verify.")
    args = parser.parse_args()

    if not settings.auth_supabase_url or not settings.auth_supabase_service_key:
        log.error("AUTH_SUPABASE_URL and AUTH_SUPABASE_SERVICE_KEY must be set "
                  "(read-only access to the Supabase auth project).")
        return 2
    if not settings.database_url:
        log.error("DATABASE_URL must be set (destination Postgres).")
        return 2

    log.info("Source (read-only): %s", settings.auth_supabase_url)
    log.info("Destination:        %s", settings.database_url.rsplit("@", 1)[-1])

    # The destination tables must exist. This is the same call the API makes on
    # startup, so running the script first on a fresh cluster is also fine.
    ensure_auth_schema()

    if args.verify_only:
        return 0 if verify() else 1

    if args.catch_up:
        if catch_up_once("manual") != "done":
            return 1
        return 0 if verify() else 1

    marker = already_completed()
    if marker and not args.force and not args.dry_run:
        log.error("Migration already completed at %s. Re-run with --force only if you "
                  "know why; use --verify-only to compare counts.", marker.get("completed_at"))
        return 3

    if args.dry_run:
        log.info("DRY RUN — no writes will be made.")

    totals = migrate(dry_run=args.dry_run)

    log.info("-" * 60)
    for table, n in totals.items():
        log.info("%-16s %d rows", table, n)

    if args.dry_run:
        log.info("DRY RUN complete — nothing written.")
        return 0

    mark_completed(totals)
    log.info("Marker written to auth_migration_state.")
    log.info("-" * 60)

    if not verify():
        log.error("VERIFICATION FAILED — Postgres is missing rows. The Supabase project "
                  "is untouched; investigate before pointing traffic at Postgres.")
        return 1

    log.info("Migration complete and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
