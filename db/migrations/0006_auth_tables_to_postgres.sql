-- 0006_auth_tables_to_postgres.sql
--
-- Moves the auth tables off the separate Supabase auth project and into the
-- pipeline Postgres database (DATABASE_URL), alongside jobs/job_events.
--
-- This file is the reviewable record of the schema. The statements are
-- mirrored in backend/app/db/schema.py (_AUTH_STATEMENTS / ensure_auth_schema),
-- which runs on every API startup and is what actually applies them on deploy
-- — matching how 001_schema.sql mirrors ensure_pipeline_schema. Keep the two
-- in sync.
--
-- The one-time DATA copy is scripts/migrate_auth_data_from_supabase.py. The
-- Supabase project is NOT dropped: it is retained read-only as the backup of
-- record.
--
-- Note on user_sessions: it is created here but was never in Supabase. It has
-- lived in the pipeline Postgres since it was introduced, but its migration
-- file (referenced as 0005_user_sessions.sql by session_repository.py and
-- session_service.py) is absent from this repository — so a fresh database
-- could not be built from db/ alone. Defining it here closes that gap.

begin;

create extension if not exists pgcrypto;

-- app_users --------------------------------------------------------------
create table if not exists app_users (
    id uuid primary key default gen_random_uuid(),
    username text not null,
    password_hash text,
    -- Legacy plaintext column. Dead for new accounts (signup writes '') but
    -- NOT NULL in the original schema and still populated for rows predating
    -- the Argon2id cutover. Dropping it is a separate migration.
    password text not null default '',
    first_name text,
    last_name text,
    email_id text not null,
    persona text,
    status text default 'pending_verification',
    email_verified boolean default false,
    failed_login_count integer default 0,
    locked_until timestamptz,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    constraint app_users_status_check check (
        status in ('pending_verification', 'active', 'locked', 'deactivated')
    )
);

-- Case-insensitive uniqueness. UserRepository.username_exists/email_exists are
-- a TOCTOU pre-check only; these indexes are the real backstop and the
-- repository maps their 23505 onto DuplicateUserError.
create unique index if not exists app_users_username_lower_key on app_users (lower(username));
create unique index if not exists app_users_email_lower_key    on app_users (lower(email_id));

-- user_sessions ----------------------------------------------------------
-- id is the refresh-token family_id, not a separately generated identifier.
create table if not exists user_sessions (
    id text primary key,
    user_id uuid not null,
    created_at timestamptz default now(),
    last_seen_at timestamptz default now(),
    ended_at timestamptz,
    end_reason text,
    user_agent text,
    ip text,
    warmup_state text not null default 'idle',
    warmup_started_at timestamptz,
    warmup_ready_at timestamptz,
    warmup_error text,
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists user_sessions_user_id_idx    on user_sessions (user_id);
create index if not exists user_sessions_active_idx     on user_sessions (user_id) where ended_at is null;
create index if not exists user_sessions_created_at_idx on user_sessions (created_at desc);

-- otp_codes --------------------------------------------------------------
create table if not exists otp_codes (
    id uuid primary key default gen_random_uuid(),
    email text not null,
    user_id uuid references app_users (id) on delete cascade,
    purpose text not null,
    code_hash text not null,
    expires_at timestamptz not null,
    attempts integer default 0,
    used_at timestamptz,
    created_at timestamptz default now(),
    constraint otp_codes_purpose_check check (
        purpose in ('signup', 'password_reset', 'password_change',
                    'email_change', 'email_change_old')
    )
);

-- Serves get_latest_active(): filter (email, purpose, used_at is null),
-- order by created_at desc.
create index if not exists otp_codes_lookup_idx
    on otp_codes (email, purpose, created_at desc) where used_at is null;

-- refresh_tokens ---------------------------------------------------------
-- family_id is also user_sessions.id. Both tables now share a database, so
-- that could be a real foreign key — deliberately NOT added, because a refresh
-- token is issued before its session row exists in some flows and the FK would
-- reject logins that succeed today. Enforcing it needs the write order audited
-- first.
create table if not exists refresh_tokens (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references app_users (id) on delete cascade,
    token_hash text not null,
    family_id text not null,
    expires_at timestamptz not null,
    created_at timestamptz default now(),
    used_at timestamptz,
    revoked_at timestamptz,
    user_agent text,
    ip text
);

create unique index if not exists refresh_tokens_token_hash_key on refresh_tokens (token_hash);
create index        if not exists refresh_tokens_family_idx     on refresh_tokens (family_id);
create index        if not exists refresh_tokens_user_idx       on refresh_tokens (user_id);

-- auth_audit -------------------------------------------------------------
create table if not exists auth_audit (
    id uuid primary key default gen_random_uuid(),
    event text not null,
    success boolean not null,
    user_id uuid references app_users (id) on delete set null,
    ip text,
    user_agent text,
    created_at timestamptz default now()
);

create index if not exists auth_audit_user_idx on auth_audit (user_id, created_at desc);

-- Completion marker for the one-time data copy. Written by
-- scripts/migrate_auth_data_from_supabase.py so a second run is refused.
create table if not exists auth_migration_state (
    id integer primary key default 1,
    completed_at timestamptz not null default now(),
    detail jsonb not null default '{}'::jsonb,
    constraint auth_migration_state_singleton check (id = 1)
);

commit;


-- ==========================================================================
-- Down
-- ==========================================================================
-- DESTRUCTIVE: drops every migrated user account, token and audit row.
--
-- Only safe while the Supabase auth project is still the source of truth —
-- i.e. before the app has been serving auth writes from Postgres. After
-- cutover, rows created here do not exist in Supabase and this loses them.
--
-- Rolling back also requires restoring the application code (the repositories
-- and the `supabase` dependency), since the code in this commit can only talk
-- to Postgres.
--
-- begin;
--   drop table if exists auth_migration_state;
--   drop table if exists auth_audit;
--   drop table if exists refresh_tokens;
--   drop table if exists otp_codes;
--   drop table if exists user_sessions;
--   drop table if exists app_users;
-- commit;
