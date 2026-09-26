# Auth migration: Supabase → Postgres

One-time move of `app_users`, `otp_codes`, `refresh_tokens` and `auth_audit`
out of the separate Supabase auth project and into the pipeline Postgres
database (`DATABASE_URL`), plus removal of the `supabase` package from the
application entirely.

**The Supabase auth project is not deleted.** It is retained, read-only, as the
backup of record. Nothing in this migration writes to it or drops anything from
it, and the migration script is safe to point at it forever.

---

## What changed

| | Before | After |
|---|---|---|
| `app_users`, `otp_codes`, `refresh_tokens`, `auth_audit` | Supabase auth project | Postgres (`DATABASE_URL`) |
| `user_sessions` | Postgres, but **no DDL in the repo** | Postgres, DDL now in `0006` |
| Auth DB client | `supabase` SDK over PostgREST | `psycopg2` via `app.db.postgres` |
| `supabase` package | Runtime dependency | **Removed** from `requirements.txt` / `pyproject.toml` |
| Legacy pipeline Supabase read-fallback | Present but inert | Removed |
| `/api/health` key | `checks.supabase` | `checks.postgres` |

`user_sessions` was never in Supabase — it has always been in the pipeline
Postgres. What was missing was its schema definition: `session_repository.py`
and `session_service.py` both reference `db/migrations/0005_user_sessions.sql`,
which does not exist in this repository, so a fresh database could not be built
from `db/` alone. `0006` now defines it.

## Two halves, applied differently

**Schema — automatic, on deploy.** `ensure_auth_schema()` is called from
`ensure_pipeline_schema()`, which the API already runs at startup. Every
statement is `IF NOT EXISTS`, so deploying this build creates the tables on the
cluster Postgres with no manual step. `db/migrations/0006_auth_tables_to_postgres.sql`
is the reviewable copy of the same DDL — keep the two in sync.

**Data — manual, once.** `scripts/migrate_auth_data_from_supabase.py` copies the
rows. It is deliberately *not* automatic: it needs to run exactly once, at a
moment a human chooses, with the output checked.

## Cutover runbook

Run from a pod or job that can reach **both** the cluster Postgres and Supabase.

### 1. Deploy this build

Creates the tables. Auth will not work correctly against the empty tables yet —
keep the window between this step and step 3 short, or run the whole sequence
during a maintenance window.

### 2. Dry run — changes nothing

```bash
python scripts/migrate_auth_data_from_supabase.py --dry-run
```

Confirm the row counts look like what you expect in Supabase. Investigate a
count of `0` for `app_users` before continuing.

### 3. Copy

```bash
python scripts/migrate_auth_data_from_supabase.py
```

The script verifies counts itself and exits non-zero if Postgres ends up with
fewer rows than Supabase. It writes a marker row to `auth_migration_state`, so a
second accidental run is refused with exit code 3.

### 4. Verify independently

```bash
python scripts/migrate_auth_data_from_supabase.py --verify-only
```

Then confirm by hand:

```sql
select count(*) from app_users;
select count(*) from app_users where status = 'active';
select count(*) from refresh_tokens where revoked_at is null and expires_at > now();
```

### 5. Smoke-test auth

- Log in as an existing account — **password must work unchanged** (the Argon2id
  hash is copied verbatim, not re-derived).
- Refresh a token.
- Sign up a new account and verify the OTP email.
- Load an admin user list.

### Required environment

| Variable | Used by | Notes |
|---|---|---|
| `DATABASE_URL` | app + script | Destination. |
| `AUTH_SUPABASE_URL` | script only | Source, read-only. |
| `AUTH_SUPABASE_SERVICE_KEY` | script only | Source, read-only. |

Keep `AUTH_SUPABASE_*` in the deployment secret after cutover. The application
never reads them, but they are what lets anyone re-read the backup.

## Rollback

**Before step 3**, rollback is just redeploying the previous build — Supabase is
still the source of truth and is untouched.

**After step 3**, rollback is only clean while the application has not yet served
auth *writes* from Postgres. Once a user has signed up, logged in or rotated a
token against Postgres, that row exists only in Postgres; reverting to the
Supabase-backed build silently loses it. In that state, roll *forward*.

Rolling back also means restoring the application code, not just the data: this
build has no Supabase client and no `supabase` dependency.

The destructive `Down` section in `0006` is commented out on purpose.

## Design notes

**Primary keys are preserved verbatim.** `app_users.id` is what every JWT `sub`
claim and every foreign key already points at. Generating fresh UUIDs would
invalidate every issued token and orphan every job attributed to a user.

**Idempotent.** Every insert is `ON CONFLICT (id) DO NOTHING`, so a re-run cannot
duplicate or overwrite. `--force` exists to bypass the marker; it is still
non-destructive.

**No `supabase` SDK in the script.** The package was removed from the project, so
the script reads Supabase over PostgREST using `requests`, which the project
already depends on. That is what allows the runtime dependency to go while the
backup stays readable.

**Table order matters.** `app_users` is copied first; the other three carry
foreign keys into it.

**`refresh_tokens.family_id → user_sessions.id` is still not a foreign key.**
Both tables finally share a database, so it *could* be enforced now. It is
deliberately left unenforced because a refresh token is issued before its
session row exists in some flows, and adding the constraint would reject logins
that succeed today. Enforcing it needs the write order audited first.

**`app_users.password` is kept.** It is the dead legacy plaintext column —
`NOT NULL` in the original schema, blanked for new accounts, still populated for
rows predating the Argon2id cutover. Dropping it is a separate migration.

## Local testing

```bash
docker compose -f docker-compose.infra.yml up -d postgres   # localhost:15432
export DATABASE_URL="postgresql://ai4ia:ai4ia_local_dev@localhost:15432/ai4ia"

python -c "import sys; sys.path.insert(0,'backend'); \
  from app.db.schema import ensure_auth_schema; ensure_auth_schema()"

pytest tests/unit/test_auth_repositories_postgres.py tests/unit/test_user_repository.py -q
```

`docker-compose.infra.yml` is for local development only — it is not a
deployment path.
