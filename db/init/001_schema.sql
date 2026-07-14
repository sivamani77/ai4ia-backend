-- Native Postgres schema for the video-job pipeline tables.
--
-- Replaces the Supabase-hosted versions of these 4 tables for a direct
-- connection (see backend/app/db/postgres.py) — the REST/PostgREST layer
-- was the source of the "Server disconnected" errors seen during
-- long-running jobs, not the tables themselves, so the schema here is
-- unchanged from what the repositories already expect.
--
-- Auto-run by the official postgres image on first container start via
-- docker-entrypoint-initdb.d (see docker-compose.infra.yml).

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
