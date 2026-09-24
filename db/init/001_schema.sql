-- Native Postgres schema for the video-job pipeline tables.
--
-- Replaces the Supabase-hosted versions of these tables for a direct
-- connection (see backend/app/db/postgres.py) — the REST/PostgREST layer
-- was the source of the "Server disconnected" errors seen during
-- long-running jobs, not the tables themselves, so the schema here is
-- unchanged from what the repositories already expect.
--
-- Auto-run by the official postgres image on first container start via
-- docker-entrypoint-initdb.d (see docker-compose.infra.yml). For a Postgres
-- instance that's already running (the server cluster), these same tables
-- are created idempotently on API startup by
-- backend/app/db/schema.py::ensure_pipeline_schema() — that function is the
-- source of truth; the fragments/storylines/insights/ingest_* tables below
-- are mirrored here only so a fresh container has them immediately too.

create table if not exists jobs (
    id text primary key,
    status text check (status in ('queued', 'running', 'completed', 'failed', 'cancelled')),
    source_type text,
    source_value text,
    progress integer not null default 0,
    user_id text,
    created_at timestamptz not null default now(),
    started_at timestamptz,
    completed_at timestamptz,
    error jsonb,
    metadata jsonb not null default '{}'::jsonb,
    job_type text,
    worker_id text,
    retry_count integer not null default 0
);

create index if not exists idx_jobs_user_id on jobs (user_id);
create index if not exists idx_jobs_status on jobs (status);
create index if not exists idx_jobs_created_at on jobs (created_at desc);

create table if not exists job_events (
    id bigserial primary key,
    job_id text not null,
    created_at timestamptz not null default now(),
    event_type text,
    payload jsonb not null default '{}'::jsonb
);

create index if not exists idx_job_events_job_id on job_events (job_id);
create index if not exists idx_job_events_job_id_id on job_events (job_id, id);

create table if not exists agent_decision_records (
    id text primary key,
    job_id text,
    insight_id text,
    user_id text,
    schema_version text not null default 'osi-1.0',
    subject text not null default '',
    user_prompt text,
    accountability text not null default 'ai4ia-pipeline-v2',
    pipeline_version text,
    agents_executed jsonb not null default '[]'::jsonb,
    decisions jsonb not null default '{}'::jsonb,
    models_used jsonb not null default '{}'::jsonb,
    outcome text,
    total_duration_ms integer,
    expires_at timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists idx_decision_records_job_id on agent_decision_records (job_id);
create index if not exists idx_decision_records_insight_id on agent_decision_records (insight_id);
create index if not exists idx_decision_records_user_id on agent_decision_records (user_id);

create table if not exists llm_call_traces (
    id text primary key,
    job_id text,
    insight_id text,
    user_id text,
    stage text,
    agent text not null default '',
    model text not null default '',
    call_type text not null default '',
    decision_label text,
    prompt_hash text,
    response_hash text,
    latency_ms integer,
    temperature double precision,
    retry_count integer not null default 0,
    error text,
    called_at timestamptz not null default now()
);

create index if not exists idx_llm_traces_job_id on llm_call_traces (job_id);
create index if not exists idx_llm_traces_insight_id on llm_call_traces (insight_id);
create index if not exists idx_llm_traces_agent on llm_call_traces (agent);

-- Atomic partial update for the jobs table — see the original note in
-- db/migrations/0004_atomic_job_partial_update.sql for why this exists
-- (avoids a read-then-merge-then-write race between concurrent writers).
-- p_columns is a JSON object of scalar top-level columns to set (only keys
-- present are touched — everything else is left alone). p_metadata_patch is
-- shallow-merged into the existing metadata jsonb blob via `||`.
create or replace function update_job_partial(
    p_id text,
    p_columns jsonb default null,
    p_metadata_patch jsonb default null
) returns void as $$
begin
    update jobs
    set
        status = coalesce(p_columns->>'status', status),
        source_type = coalesce(p_columns->>'source_type', source_type),
        source_value = coalesce(p_columns->>'source_value', source_value),
        progress = case when p_columns ? 'progress' then (p_columns->>'progress')::int else progress end,
        started_at = coalesce((p_columns->>'started_at')::timestamptz, started_at),
        completed_at = coalesce((p_columns->>'completed_at')::timestamptz, completed_at),
        error = case when p_columns ? 'error' then p_columns->'error' else error end,
        job_type = coalesce(p_columns->>'job_type', job_type),
        worker_id = coalesce(p_columns->>'worker_id', worker_id),
        retry_count = case when p_columns ? 'retry_count' then (p_columns->>'retry_count')::int else retry_count end,
        metadata = case
            when p_metadata_patch is not null then coalesce(metadata, '{}'::jsonb) || p_metadata_patch
            else metadata
        end
    where id = p_id;
end;
$$ language plpgsql;

-- ---------------------------------------------------------------------
-- Former Supabase pipeline tables (fragments, storylines, insights,
-- ingest_jobs, ingest_status, ingest_job_events). See
-- backend/app/db/schema.py::ensure_pipeline_schema() for the authoritative,
-- idempotent version of these statements — this block is a mirror for
-- fresh containers only.
-- ---------------------------------------------------------------------

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
);

create index if not exists idx_fragments_speaker on fragments (speaker);
create index if not exists idx_fragments_primary_chapter on fragments (primary_chapter);
create index if not exists idx_fragments_source_video on fragments (source_video);
create index if not exists idx_fragments_video_id on fragments (video_id);

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
);

create index if not exists idx_storylines_job_id on storylines (job_id);
create index if not exists idx_storylines_created_at on storylines (created_at desc);

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
);

create index if not exists idx_insights_created_at on insights (created_at desc);

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
);

create index if not exists idx_ingest_jobs_status on ingest_jobs (status);
create index if not exists idx_ingest_jobs_created_at on ingest_jobs (created_at desc);

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
);

create index if not exists idx_ingest_status_video_id on ingest_status (video_id);
create index if not exists idx_ingest_status_stage_status on ingest_status (stage, status);

create table if not exists ingest_job_events (
    id bigserial primary key,
    job_id text not null,
    type text,
    data jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_ingest_job_events_job_id on ingest_job_events (job_id, id);

create table if not exists videos (
    video_id text primary key,
    source_video text unique,
    title text,
    duration double precision,
    webpage_url text,
    s3_raw_key text,
    updated_at timestamptz not null default now()
);

create index if not exists idx_videos_source_video on videos (source_video);

create table if not exists speakers (
    canonical_name text primary key
);

create table if not exists speaker_aliases (
    alias text primary key,
    canonical_name text not null references speakers(canonical_name)
);


-- Request and LLM-exchange records (mirrors backend/app/db/schema.py).
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
    );
create index if not exists idx_prompt_resolutions_job_id on prompt_resolutions (job_id);
create index if not exists idx_prompt_resolutions_insight_id on prompt_resolutions (insight_id);
create index if not exists idx_prompt_resolutions_user_id on prompt_resolutions (user_id);
create index if not exists idx_prompt_resolutions_created_at on prompt_resolutions (created_at desc);
create index if not exists idx_prompt_resolutions_outcome on prompt_resolutions (intent, layer_outcome);
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
    );
create index if not exists idx_llm_call_payloads_job_id on llm_call_payloads (job_id);
create index if not exists idx_llm_call_payloads_user_id on llm_call_payloads (user_id);
create index if not exists idx_llm_call_payloads_trace_id on llm_call_payloads (trace_id);
create index if not exists idx_llm_call_payloads_expires_at on llm_call_payloads (expires_at);
create index if not exists idx_prompt_resolutions_expires_at on prompt_resolutions (expires_at);
