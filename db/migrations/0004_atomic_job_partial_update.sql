-- Atomic partial update for the jobs table.
--
-- IMPORTANT — run this in the MAIN BACKEND project's SQL editor (the one
-- pointed at by SUPABASE_URL / SUPABASE_SERVICE_KEY, where the `jobs` table
-- lives), NOT the AUTH project (AUTH_SUPABASE_URL, app_users table). See
-- backend/app/db/repositories/job_repository.py, which calls
-- get_supabase_client() — the main-project client — not
-- get_auth_supabase_client().
--
-- backend/app/db/repositories/job_repository.py previously did a
-- read-then-merge-then-write from application code: fetch the row, merge
-- updates into a Python dict, write the whole row back. Two concurrent
-- writers (e.g. the pipeline runner updating progress while a user cancels
-- the job) can both read the same snapshot and the last writer silently
-- clobbers the other's change. This function does the merge inside a single
-- UPDATE statement so Postgres's row-level locking serializes concurrent
-- callers instead of losing writes.
--
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
        started_at = coalesce(p_columns->>'started_at', started_at),
        completed_at = coalesce(p_columns->>'completed_at', completed_at),
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
