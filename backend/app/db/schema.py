"""Idempotent Postgres schema for the pipeline tables that used to live on
Supabase (fragments, storylines, insights, ingest_jobs, ingest_status,
ingest_job_events).

Run once at API startup (see main.py's lifespan) so an already-running
server-cluster Postgres — which never had these tables — gets them created
automatically, without anyone having to run a migration file by hand. The
same statements are mirrored into db/init/001_schema.sql for fresh
containers, but this function is the source of truth.
"""
from __future__ import annotations

import logging

import psycopg2

from app.core.settings import settings

logger = logging.getLogger("ai4ia.db.schema")

_STATEMENTS = [
    """
    create table if not exists fragments (
        fragment_id text primary key,
        source_video text,
        video_id text,
        video_title text,
        speaker text,
        speaker_attribution text,
        text text not null default '',
        summary text,
        start double precision,
        "end" double precision,
        duration double precision,
        primary_theme text,
        primary_chapter text,
        theme_tags jsonb not null default '[]'::jsonb,
        chapter_tags jsonb not null default '[]'::jsonb,
        clip_usage_candidates jsonb not null default '[]'::jsonb,
        quote_strength double precision,
        clarity_score double precision,
        year integer,
        edition text,
        session_type text,
        keywords jsonb not null default '[]'::jsonb,
        extra jsonb not null default '{}'::jsonb
    )
    """,
    "create index if not exists idx_fragments_speaker on fragments (speaker)",
    "create index if not exists idx_fragments_primary_chapter on fragments (primary_chapter)",
    "create index if not exists idx_fragments_source_video on fragments (source_video)",
    "create index if not exists idx_fragments_video_id on fragments (video_id)",
    """
    create table if not exists storylines (
        id text primary key,
        job_id text,
        title text,
        logline text,
        user_prompt text,
        target_duration_seconds integer,
        is_validated boolean not null default false,
        sections jsonb not null default '[]'::jsonb,
        selected_candidates jsonb not null default '[]'::jsonb,
        export_timeline jsonb not null default '[]'::jsonb,
        editorial_quality_gates jsonb not null default '{}'::jsonb,
        full_payload jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now()
    )
    """,
    "create index if not exists idx_storylines_job_id on storylines (job_id)",
    "create index if not exists idx_storylines_created_at on storylines (created_at desc)",
    """
    create table if not exists insights (
        id text primary key,
        query text not null default '',
        title text,
        summary text,
        filters jsonb not null default '{}'::jsonb,
        patterns jsonb not null default '[]'::jsonb,
        contrasts jsonb not null default '[]'::jsonb,
        recommended_clips jsonb not null default '[]'::jsonb,
        unsupported_claims jsonb not null default '[]'::jsonb,
        evidence_pool jsonb not null default '[]'::jsonb,
        evidence_coverage double precision not null default 0,
        models jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now()
    )
    """,
    "create index if not exists idx_insights_created_at on insights (created_at desc)",
    # Owner of each insight. Rows made before this column existed get their
    # owner from the decision record, but only when exactly one user has a
    # record for that id (old second-resolution ids can collide). Rows left
    # null are visible to admins only.
    "alter table insights add column if not exists user_id text",
    "create index if not exists idx_insights_user_created on insights (user_id, created_at desc)",
    """
    do $$
    begin
        if to_regclass('agent_decision_records') is not null then
            update insights i
               set user_id = d.user_id
              from (
                    select insight_id, min(user_id) as user_id
                      from agent_decision_records
                     where insight_id is not null
                       and user_id is not null
                     group by insight_id
                    having count(distinct user_id) = 1
                   ) d
             where d.insight_id = i.id
               and i.user_id is null;
        end if;
    end
    $$
    """,
    """
    create table if not exists ingest_jobs (
        id text primary key,
        status text,
        channel_url text,
        video_limit integer,
        progress integer not null default 0,
        message text,
        error jsonb,
        metadata jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now(),
        started_at timestamptz,
        completed_at timestamptz
    )
    """,
    "create index if not exists idx_ingest_jobs_status on ingest_jobs (status)",
    "create index if not exists idx_ingest_jobs_created_at on ingest_jobs (created_at desc)",
    """
    create table if not exists ingest_status (
        id bigserial primary key,
        video_id text not null,
        stage text not null,
        status text not null default 'pending',
        job_id text,
        error text,
        metadata jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now(),
        unique (video_id, stage)
    )
    """,
    "create index if not exists idx_ingest_status_video_id on ingest_status (video_id)",
    "create index if not exists idx_ingest_status_stage_status on ingest_status (stage, status)",
    """
    create table if not exists ingest_job_events (
        id bigserial primary key,
        job_id text not null,
        type text,
        data jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now()
    )
    """,
    "create index if not exists idx_ingest_job_events_job_id on ingest_job_events (job_id, id)",
    # ------------------------------------------------------------------
    # videos — canonical registry, one row per video. Lets list_videos()/
    # get_video() (archive_service.py) stop re-deriving video identity by
    # parsing the same filename string independently in three different
    # places in the codebase, and gives a real signal ("does this video
    # have a raw file in S3") for scripts/reconcile_archive_stores.py.
    # ------------------------------------------------------------------
    """
    create table if not exists videos (
        video_id text primary key,
        source_video text unique,
        title text,
        duration double precision,
        webpage_url text,
        s3_raw_key text,
        updated_at timestamptz not null default now()
    )
    """,
    "create index if not exists idx_videos_source_video on videos (source_video)",
    # ------------------------------------------------------------------
    # speakers / speaker_aliases — canonical speaker identity. fragments.
    # speaker stays a plain text column (no FK migration, no query
    # rewrites elsewhere); this is the growable source of truth that
    # backend/app/services/speaker_aliases.canonicalize() and
    # scripts/04b_extract_speakers.py's in-code alias map are seeded from
    # and should eventually read from directly.
    # ------------------------------------------------------------------
    """
    create table if not exists speakers (
        canonical_name text primary key
    )
    """,
    """
    create table if not exists speaker_aliases (
        alias text primary key,
        canonical_name text not null references speakers(canonical_name)
    )
    """,
    # Speaker registry columns (config/speaker_registry.json). Additive only:
    # uat and release share this database, so older code must keep working.
    "alter table speakers add column if not exists status text not null default 'confirmed'",
    "alter table speakers add column if not exists source text",
    "alter table speakers add column if not exists register_name text",
    "alter table speakers add column if not exists affiliation text",
    "alter table speakers add column if not exists updated_at timestamptz not null default now()",
    "alter table speaker_aliases add column if not exists normalized_alias text",
    "alter table speaker_aliases add column if not exists source text",
    "create index if not exists idx_speaker_aliases_normalized on speaker_aliases (normalized_alias)",
    """
    create table if not exists video_speakers (
        video_id text not null,
        canonical_name text not null,
        role text,
        ordinal integer not null default 0,
        primary key (video_id, canonical_name)
    )
    """,
    "alter table videos add column if not exists description text",
    # One-off data repairs (app/services/speaker_repair.py): a done marker per
    # repair, and the original values of every fragment it touches.
    """
    create table if not exists data_repairs (
        name text primary key,
        status text not null,
        mode text,
        attempts integer not null default 0,
        registry_version text,
        summary jsonb not null default '{}'::jsonb,
        updated_at timestamptz not null default now()
    )
    """,
    """
    create table if not exists fragment_speaker_backup (
        repair_name text not null,
        store text not null,
        fragment_id text not null,
        video_id text,
        speaker text,
        speaker_attribution text,
        speaker_roster text,
        captured_at timestamptz not null default now(),
        primary key (repair_name, store, fragment_id)
    )
    """,
    # One row per request: what was asked and what each layer decided.
    # See docs/intent-layer-design.md §10.
    """
    create table if not exists prompt_resolutions (
        id text primary key,
        job_id text,
        insight_id text,
        surface text not null,
        attempt integer not null default 1,
        user_id text,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now(),
        raw_prompt text not null,
        flags jsonb not null default '{}'::jsonb,
        intent text,
        intent_version text,
        template_hash text,
        resolution_method text,
        slots jsonb not null default '{}'::jsonb,
        dropped_slots jsonb not null default '[]'::jsonb,
        layer_outcome text,
        reason_code text,
        user_message text,
        suggestions jsonb not null default '[]'::jsonb,
        coverage jsonb not null default '{}'::jsonb,
        code_version text,
        corpus_version text,
        registry_version text,
        timings_ms jsonb not null default '{}'::jsonb,
        expires_at timestamptz
    )
    """,
    "create index if not exists idx_prompt_resolutions_job_id on prompt_resolutions (job_id)",
    "create index if not exists idx_prompt_resolutions_insight_id on prompt_resolutions (insight_id)",
    "create index if not exists idx_prompt_resolutions_user_id on prompt_resolutions (user_id)",
    "create index if not exists idx_prompt_resolutions_created_at on prompt_resolutions (created_at desc)",
    "create index if not exists idx_prompt_resolutions_outcome on prompt_resolutions (intent, layer_outcome)",
    # Exact text of every LLM request and response. Text, not jsonb: invalid
    # JSON responses must be storable.
    """
    create table if not exists llm_call_payloads (
        id text primary key,
        trace_id text,
        job_id text,
        insight_id text,
        user_id text,
        stage text,
        model text,
        request_text text not null,
        response_text text,
        error text,
        latency_ms integer,
        size_bytes integer not null,
        called_at timestamptz not null default now(),
        expires_at timestamptz
    )
    """,
    "create index if not exists idx_llm_call_payloads_job_id on llm_call_payloads (job_id)",
    "create index if not exists idx_llm_call_payloads_user_id on llm_call_payloads (user_id)",
    "create index if not exists idx_llm_call_payloads_trace_id on llm_call_payloads (trace_id)",
    "create index if not exists idx_llm_call_payloads_expires_at on llm_call_payloads (expires_at)",
    "create index if not exists idx_prompt_resolutions_expires_at on prompt_resolutions (expires_at)",
    # ------------------------------------------------------------------
    # feedbacks / notifications — the frontend's feedbackApi.js and
    # notificationsApi.js were switched from direct Supabase calls to this
    # backend's API during the security remediation (frontend commit
    # "minor fixes", 2026-09-21 — see docs/frontend-changes-for-security-
    # fixes.md F-5, which already lists both as needing the auth-refresh
    # interceptor), but the backend side was never actually built until now.
    # ------------------------------------------------------------------
    """
    create table if not exists feedbacks (
        id text primary key,
        user_id text not null,
        username text not null,
        email text not null,
        feedback text not null,
        status text not null default 'open',
        resolution_message text,
        resolved_at timestamptz,
        resolved_by text,
        created_at timestamptz not null default now(),
        constraint feedbacks_status_check check (status in ('open', 'resolved'))
    )
    """,
    "create index if not exists idx_feedbacks_status on feedbacks (status)",
    "create index if not exists idx_feedbacks_user_id on feedbacks (user_id)",
    """
    create table if not exists notifications (
        id text primary key,
        user_id text not null,
        message text not null,
        type text not null default 'info',
        redirect_page text,
        meta jsonb not null default '{}'::jsonb,
        read boolean not null default false,
        created_at timestamptz not null default now()
    )
    """,
    "create index if not exists idx_notifications_user_id on notifications (user_id, created_at desc)",
]


def _seed_speaker_aliases() -> None:
    """Upsert speakers and aliases from the speaker registry. Safe on every
    startup. Rows the registry no longer lists are removed only by the
    speaker repair (app/services/speaker_repair.py), not here."""
    from app.db.repositories.speaker_repository import SpeakerRepository
    from app.services.speaker_registry import get_registry

    SpeakerRepository().upsert_from_registry(get_registry())


# ---------------------------------------------------------------------------
# Auth tables — migrated off the separate auth Supabase project.
#
# These used to live in their own Supabase project reached via
# AUTH_SUPABASE_URL. They now live here, in the same database as the pipeline
# tables, which is what makes refresh_tokens.family_id -> user_sessions.id a
# same-database relationship instead of the unenforced cross-project link it
# used to be. See db/migrations/0006_auth_tables_to_postgres.sql for the same
# DDL as a reviewable migration, and scripts/migrate_auth_data_from_supabase.py
# for the one-time data copy.
#
# Two deliberate choices, both to preserve existing behaviour exactly:
#
#   * app_users.password is NOT NULL. It is the dead legacy plaintext column;
#     signup writes '' and upgrade_legacy_password blanks it. Dropping it is a
#     separate change, because rows predating the Argon2id cutover still carry
#     a value there and UserRecord still exposes it.
#   * refresh_tokens.family_id gets an index but NOT a foreign key to
#     user_sessions. A refresh token is issued before its session row exists in
#     some flows, so adding that FK now would reject logins that succeed today.
#     Enforcing it is a follow-up that needs the write order audited first.
# ---------------------------------------------------------------------------

_AUTH_STATEMENTS = [
    # gen_random_uuid() lives in pgcrypto on older servers; on PG13+ it is
    # built in. Creating the extension is a no-op where it is already present
    # and harmless where it is built in.
    "create extension if not exists pgcrypto",

    """
    create table if not exists app_users (
        id uuid primary key default gen_random_uuid(),
        username text not null,
        password_hash text,
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
    )
    """,
    # Bumped to revoke every access token the user holds (password, persona
    # or account changes); tokens carry the version they were minted with.
    "alter table app_users add column if not exists token_version integer not null default 0",
    # Case-insensitive uniqueness. UserRepository.username_exists/email_exists
    # are a TOCTOU pre-check only — these indexes are the real backstop, and
    # the repository maps their 23505 to DuplicateUserError.
    """
    create unique index if not exists app_users_username_lower_key
        on app_users (lower(username))
    """,
    """
    create unique index if not exists app_users_email_lower_key
        on app_users (lower(email_id))
    """,

    """
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
    )
    """,
    "create index if not exists user_sessions_user_id_idx on user_sessions (user_id)",
    """
    create index if not exists user_sessions_active_idx
        on user_sessions (user_id) where ended_at is null
    """,
    "create index if not exists user_sessions_created_at_idx on user_sessions (created_at desc)",

    """
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
            purpose in (
                'signup', 'password_reset', 'password_change',
                'email_change', 'email_change_old', 'persona_upgrade'
            )
        )
    )
    """,
    # The create above only shapes a *fresh* database — an existing otp_codes
    # keeps whatever constraint it was built with, so every new purpose value
    # has to be widened in explicitly here too. Guarded on the constraint
    # definition rather than run unconditionally: a bare drop/add would take
    # an ACCESS EXCLUSIVE lock and revalidate the table on every boot.
    """
    do $$
    begin
        if not exists (
            select 1 from pg_constraint
            where conrelid = 'otp_codes'::regclass
              and conname = 'otp_codes_purpose_check'
              and pg_get_constraintdef(oid) like '%persona_upgrade%'
        ) then
            alter table otp_codes drop constraint if exists otp_codes_purpose_check;
            alter table otp_codes add constraint otp_codes_purpose_check check (
                purpose in (
                    'signup', 'password_reset', 'password_change',
                    'email_change', 'email_change_old', 'persona_upgrade'
                )
            );
        end if;
    end $$;
    """,
    # get_latest_active() filters on (email, purpose, used_at is null) and
    # orders by created_at desc — this index serves that query directly.
    """
    create index if not exists otp_codes_lookup_idx
        on otp_codes (email, purpose, created_at desc) where used_at is null
    """,

    """
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
    )
    """,
    # get_by_hash() is the hot path on every refresh; the hash is effectively
    # unique, so a unique index both serves the lookup and blocks collisions.
    """
    create unique index if not exists refresh_tokens_token_hash_key
        on refresh_tokens (token_hash)
    """,
    "create index if not exists refresh_tokens_family_idx on refresh_tokens (family_id)",
    "create index if not exists refresh_tokens_user_idx on refresh_tokens (user_id)",

    """
    create table if not exists auth_audit (
        id uuid primary key default gen_random_uuid(),
        event text not null,
        success boolean not null,
        user_id uuid references app_users (id) on delete set null,
        ip text,
        user_agent text,
        created_at timestamptz default now()
    )
    """,
    "create index if not exists auth_audit_user_idx on auth_audit (user_id, created_at desc)",
]


# Serialises schema creation across API workers and pods.
_SCHEMA_LOCK_KEY = 4_341_830_001


def _run_locked(statements: list[str]) -> None:
    """Run DDL on one connection under an advisory lock. Concurrent
    CREATE ... IF NOT EXISTS on a new table otherwise fails with duplicate-key
    errors, which several workers starting at once would hit."""
    conn = psycopg2.connect(settings.database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("select pg_advisory_lock(%s)", (_SCHEMA_LOCK_KEY,))
            for statement in statements:
                cur.execute(statement)
    finally:
        conn.close()  # also releases the lock


def ensure_auth_schema() -> None:
    """Create the auth tables if they don't already exist. Safe to call on
    every startup — every statement is IF NOT EXISTS. This is what makes the
    schema half of the Supabase migration happen automatically on deploy; the
    data half is the one-time scripts/migrate_auth_data_from_supabase.py."""
    _run_locked(_AUTH_STATEMENTS)
    logger.info(
        "Auth Postgres schema ensured "
        "(app_users, user_sessions, otp_codes, refresh_tokens, auth_audit)"
    )


def ensure_pipeline_schema() -> None:
    """Create the pipeline tables/indexes if they don't already exist. Safe
    to call on every startup — every statement is IF NOT EXISTS."""
    _run_locked(_STATEMENTS)
    _seed_speaker_aliases()
    logger.info(
        "Pipeline Postgres schema ensured "
        "(fragments, storylines, insights, ingest_*, videos, speakers, speaker_aliases, "
        "prompt_resolutions, llm_call_payloads)"
    )
    ensure_auth_schema()
