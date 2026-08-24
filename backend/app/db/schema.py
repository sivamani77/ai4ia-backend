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

from app.db.postgres import pg_execute

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
]


def _seed_speaker_aliases() -> None:
    """Idempotent: inserts canonical names + aliases from the in-code map
    (backend/app/services/speaker_aliases.ALIASES) if not already present.
    Safe to call every startup — ON CONFLICT DO NOTHING, never overwrites
    a row someone has since edited directly in the table."""
    from app.services.speaker_aliases import ALIASES

    canonical_names = set(ALIASES.values())
    for name in canonical_names:
        pg_execute(
            "insert into speakers (canonical_name) values (%s) on conflict do nothing",
            (name,),
        )
    for alias, canonical_name in ALIASES.items():
        pg_execute(
            """
            insert into speaker_aliases (alias, canonical_name)
            values (%s, %s)
            on conflict (alias) do nothing
            """,
            (alias, canonical_name),
        )


def ensure_pipeline_schema() -> None:
    """Create the pipeline tables/indexes if they don't already exist. Safe
    to call on every startup — every statement is IF NOT EXISTS."""
    for statement in _STATEMENTS:
        pg_execute(statement)
    _seed_speaker_aliases()
    logger.info(
        "Pipeline Postgres schema ensured "
        "(fragments, storylines, insights, ingest_*, videos, speakers, speaker_aliases)"
    )
