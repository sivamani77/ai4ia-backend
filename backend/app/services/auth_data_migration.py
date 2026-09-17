"""Copy the auth tables from the Supabase auth project into Postgres.

This is the data half of the Supabase -> Postgres migration; the schema half
(app.db.schema.ensure_auth_schema) already runs on every API startup. Design
constraints, all deliberate:

  * READ-ONLY against Supabase. Nothing is deleted or updated there. The
    Supabase project is retained as the backup of record, so this must
    remain safe to run against it forever.

  * No `supabase` SDK. Reads go through PostgREST over plain `requests`,
    which the project already depends on.

  * Idempotent and one-time. Every insert is ON CONFLICT (id) DO NOTHING, and
    a completion marker is written to the `auth_migration_state` table. Once
    that marker exists, run_once() is a no-op.

  * Primary keys are preserved. app_users.id is what JWT `sub` claims and
    every foreign key already point at, so generating new UUIDs here would
    silently invalidate every issued token and orphan every job attributed
    to a user.

Order matters: app_users must land before the tables whose foreign keys
reference it, or those inserts are rejected.

Used by scripts/migrate_auth_data_from_supabase.py (manual dry-run/verify/
force CLI) and by app.main's startup lifespan (automatic one-time run on
deploy, guarded by the completion marker).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import requests

from app.core.settings import settings
from app.db.postgres import pg_execute, pg_fetch_one
from app.db.schema import ensure_auth_schema

log = logging.getLogger("ai4ia.auth_migration")

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


def run_once() -> None:
    """Automatic, idempotent entry point for API startup.

    No-ops (with a log line) when Supabase source credentials aren't
    configured — that's the normal case for local dev, which never had a
    Supabase project to migrate from. Once the `auth_migration_state` marker
    exists, every subsequent call is a no-op too, so this is safe to run on
    every deploy/restart forever.
    """
    if not settings.auth_supabase_url or not settings.auth_supabase_service_key:
        log.info("AUTH_SUPABASE_URL/AUTH_SUPABASE_SERVICE_KEY not set — skipping "
                  "one-time Supabase auth data migration.")
        return

    ensure_auth_schema()

    if already_completed():
        log.info("Supabase auth data migration already completed — skipping.")
        return

    log.info("Running one-time Supabase auth data migration ...")
    totals = migrate(dry_run=False)
    mark_completed(totals)

    if verify():
        log.info("Supabase auth data migration complete and verified: %s", totals)
    else:
        log.error("Supabase auth data migration finished but verification found "
                   "Postgres missing rows relative to Supabase — investigate; "
                   "re-run scripts/migrate_auth_data_from_supabase.py --verify-only.")
