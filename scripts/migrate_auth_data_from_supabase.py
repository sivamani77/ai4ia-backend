#!/usr/bin/env python3
"""One-time copy of the auth tables from the Supabase auth project into Postgres.

Run this ONCE, against the deployed cluster Postgres, after the schema half of
the migration has been applied (that happens automatically on API startup via
app.db.schema.ensure_auth_schema).

Design constraints, all deliberate:

  * READ-ONLY against Supabase. Nothing is deleted or updated there. The
    Supabase project is retained as the backup of record, so this script must
    remain safe to run against it forever.

  * No `supabase` SDK. That package was removed from the application's
    dependencies as part of this migration, so reads go through PostgREST over
    plain `requests` — which the project already depends on.

  * Idempotent and one-time. Every insert is ON CONFLICT (id) DO NOTHING, and a
    completion marker is written to the `auth_migration_state` table. A second
    run is refused unless --force is passed, and even --force cannot duplicate
    rows because primary keys are carried across verbatim.

  * Primary keys are preserved. app_users.id is what JWT `sub` claims and every
    foreign key already point at, so generating new UUIDs here would silently
    invalidate every issued token and orphan every job attributed to a user.

Order matters: app_users must land before the tables whose foreign keys
reference it, or those inserts are rejected.

Usage
-----
    # See what would be copied, change nothing (always do this first):
    python scripts/migrate_auth_data_from_supabase.py --dry-run

    # Perform the copy:
    python scripts/migrate_auth_data_from_supabase.py

    # Re-check counts on both sides after the fact:
    python scripts/migrate_auth_data_from_supabase.py --verify-only

Requires AUTH_SUPABASE_URL and AUTH_SUPABASE_SERVICE_KEY (the Supabase auth
project, read-only) and DATABASE_URL (the destination Postgres).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import requests  # noqa: E402

from app.core.settings import settings  # noqa: E402
from app.db.postgres import pg_execute, pg_fetch_one  # noqa: E402
from app.db.schema import ensure_auth_schema  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("migrate_auth")

PAGE_SIZE = 500
HTTP_TIMEOUT = 30

# Parent first — the rest carry foreign keys into app_users.
#
# user_sessions is absent on purpose: it never lived in Supabase. It has been
# in the pipeline Postgres all along (see SessionRepository); this migration
# only gives it a DDL definition it was previously missing.
TABLES: list[tuple[str, list[str]]] = [
    ("app_users", [
        "id", "username", "password_hash", "password", "first_name", "last_name",
        "email_id", "persona", "status", "email_verified", "failed_login_count",
        "locked_until", "created_at", "updated_at",
    ]),
    ("refresh_tokens", [
        "id", "user_id", "token_hash", "family_id", "expires_at",
        "created_at", "used_at", "revoked_at", "user_agent", "ip",
    ]),
    ("otp_codes", [
        "id", "email", "user_id", "purpose", "code_hash",
        "expires_at", "attempts", "used_at", "created_at",
    ]),
    ("auth_audit", [
        "id", "event", "success", "user_id", "ip", "user_agent", "created_at",
    ]),
]

MARKER_TABLE = """
create table if not exists auth_migration_state (
    id integer primary key default 1,
    completed_at timestamptz not null default now(),
    detail jsonb not null default '{}'::jsonb,
    constraint auth_migration_state_singleton check (id = 1)
)
"""


class MigrationError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Supabase reads (PostgREST, read-only)
# --------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    key = settings.auth_supabase_service_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def _rest_url(table: str) -> str:
    return f"{settings.auth_supabase_url.rstrip('/')}/rest/v1/{table}"


def fetch_table(table: str, columns: list[str]) -> Iterator[list[dict[str, Any]]]:
    """Yield pages of rows from the Supabase table.

    Ordered by id so paging is stable: without an explicit order PostgREST
    gives no guarantee between requests, and rows could be skipped or repeated
    across page boundaries.
    """
    offset = 0
    while True:
        resp = requests.get(
            _rest_url(table),
            headers=_headers(),
            params={"select": ",".join(columns), "order": "id.asc",
                    "limit": PAGE_SIZE, "offset": offset},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 404:
            log.warning("  table %s does not exist in Supabase — skipping", table)
            return
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return
        yield rows
        if len(rows) < PAGE_SIZE:
            return
        offset += PAGE_SIZE


def supabase_count(table: str) -> int | None:
    """Exact row count via PostgREST's Content-Range header."""
    try:
        resp = requests.get(
            _rest_url(table),
            headers={**_headers(), "Prefer": "count=exact", "Range": "0-0"},
            params={"select": "id"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        content_range = resp.headers.get("content-range", "")
        return int(content_range.split("/")[-1]) if "/" in content_range else None
    except Exception:
        log.warning("  could not count %s in Supabase", table, exc_info=True)
        return None


# --------------------------------------------------------------------------
# Postgres writes
# --------------------------------------------------------------------------

def postgres_count(table: str) -> int:
    row = pg_fetch_one(f"select count(*) as n from {table}")
    return int(row["n"]) if row else 0


def insert_page(table: str, columns: list[str], rows: list[dict[str, Any]]) -> int:
    """Insert one page. ON CONFLICT (id) DO NOTHING makes re-runs harmless:
    a row already copied is left exactly as it is, never overwritten."""
    column_sql = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join(f"%({c})s" for c in columns)
    inserted = 0
    for row in rows:
        params = {c: row.get(c) for c in columns}
        pg_execute(
            f"insert into {table} ({column_sql}) values ({placeholders}) "
            f"on conflict (id) do nothing",
            params,
        )
        inserted += 1
    return inserted


def already_completed() -> dict[str, Any] | None:
    pg_execute(MARKER_TABLE)
    return pg_fetch_one("select * from auth_migration_state where id = 1")


def mark_completed(detail: dict[str, Any]) -> None:
    import json
    pg_execute(
        """
        insert into auth_migration_state (id, detail) values (1, %(detail)s)
        on conflict (id) do update set completed_at = now(), detail = %(detail)s
        """,
        {"detail": json.dumps(detail)},
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def verify() -> bool:
    """Compare row counts on both sides. Postgres may legitimately hold MORE
    rows than Supabase (the app has been writing there since cutover), so the
    failure condition is Postgres holding FEWER."""
    ok = True
    log.info("%-16s %12s %12s   %s", "TABLE", "SUPABASE", "POSTGRES", "STATUS")
    for table, _ in TABLES:
        src = supabase_count(table)
        dst = postgres_count(table)
        if src is None:
            status = "SKIPPED (absent in Supabase)"
        elif dst >= src:
            status = "OK"
        else:
            status = f"MISSING {src - dst} ROWS"
            ok = False
        log.info("%-16s %12s %12d   %s", table, "-" if src is None else src, dst, status)
    return ok


def migrate(dry_run: bool) -> dict[str, int]:
    totals: dict[str, int] = {}
    for table, columns in TABLES:
        log.info("Copying %s ...", table)
        copied = 0
        for page in fetch_table(table, columns):
            if dry_run:
                copied += len(page)
            else:
                copied += insert_page(table, columns, page)
            log.info("  %s: %d rows processed", table, copied)
        totals[table] = copied
        if copied == 0:
            log.info("  %s: nothing to copy", table)
    return totals


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
