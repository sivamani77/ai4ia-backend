from __future__ import annotations

import json
import logging
from typing import Any, Mapping, TypedDict

import psycopg2.errors
import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one, pg_transaction
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.jobs")


class JobRecord(TypedDict, total=False):
    id: str
    status: str
    source_type: str | None
    source_value: str | None
    progress: int
    user_id: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    error: Any
    metadata: dict[str, Any]
    job_type: str | None
    worker_id: str | None
    retry_count: int


# Serialises quota checks for video-job creation across every replica.
_JOB_QUOTA_LOCK_KEY = 0x6A6F6271  # "jobq"

_INSERT_JOB_SQL = """
    insert into jobs (id, status, source_type, source_value, progress, user_id,
                      created_at, started_at, completed_at, error, metadata,
                      job_type, worker_id, retry_count)
    values (%(id)s, %(status)s, %(source_type)s, %(source_value)s, %(progress)s, %(user_id)s,
            coalesce(%(created_at)s, now()), %(started_at)s, %(completed_at)s,
            %(error)s, %(metadata)s, %(job_type)s, %(worker_id)s, %(retry_count)s)
    returning *
"""


class JobRepository:
    table_name = "jobs"
    _schema_columns = {
        "id",
        "status",
        "source_type",
        "source_value",
        "progress",
        "user_id",
        "created_at",
        "started_at",
        "completed_at",
        "error",
        "metadata",
        "job_type",
        "worker_id",
        "retry_count",
    }

    def create_job(self, data: Mapping[str, Any]) -> JobRecord:
        payload = self._normalize_input(data)
        record_id = payload.get("id")
        self._log("insert", record_id)

        row = pg_fetch_one(_INSERT_JOB_SQL, self._to_row_params(payload))
        if row is not None:
            return self._normalize_output(row)
        return self.get_job(str(record_id))

    def create_job_within_quota(
        self, data: Mapping[str, Any], *, user_id: str, max_running: int, max_queued: int, max_active: int,
        dispatch_id: str | None = None, worker_queue: str | None = None,
    ) -> str | None:
        """Insert a job only while its owner and the platform are under their
        limits. The count and the insert share one transaction under an
        advisory lock, so parallel submissions cannot all pass the check.
        Returns None when the job was created, otherwise the limit that was hit:
        "user_running", "user_queued" or "active", or "busy" when the lock
        could not be taken in time."""
        payload = self._normalize_input(data)
        self._log("insert_within_quota", payload.get("id"))
        try:
            return self._create_within_quota(
                payload, user_id, max_running, max_queued, max_active, dispatch_id, worker_queue,
            )
        except psycopg2.errors.LockNotAvailable:
            return "busy"

    def _create_within_quota(
        self, payload: dict[str, Any], user_id: str, max_running: int, max_queued: int, max_active: int,
        dispatch_id: str | None, worker_queue: str | None,
    ) -> str | None:
        with pg_transaction() as cur:
            # Waiting requests each hold a pool connection; give up rather than
            # queue long enough to drain the pool.
            cur.execute("set local lock_timeout = '5s'")
            cur.execute("select pg_advisory_xact_lock(%s)", (_JOB_QUOTA_LOCK_KEY,))
            cur.execute(
                """
                select count(*) filter (where user_id = %(user_id)s and status = 'running') as user_running,
                       count(*) filter (where user_id = %(user_id)s and status = 'queued') as user_queued,
                       count(*) as active
                from jobs
                where status in ('queued', 'running')
                """,
                {"user_id": user_id},
            )
            counts = cur.fetchone()
            if counts["user_running"] >= max_running:
                return "user_running"
            if counts["user_queued"] >= max_queued:
                return "user_queued"
            if counts["active"] >= max_active:
                return "active"
            cur.execute(_INSERT_JOB_SQL, self._to_row_params(payload))
            if dispatch_id is not None:
                cur.execute(
                    "update jobs set dispatch_id = %s, worker_queue = %s, dispatched_at = now() where id = %s",
                    (dispatch_id, worker_queue, payload["id"]),
                )
        return None

    # -- leases (docs/design/job-recovery-plan.md) -----------------------------------

    def claim(self, record_id: str, *, dispatch_id: str | None, owner: str, worker_queue: str) -> bool:
        """Take the lease on a job that has never run. Only a queued job with
        no earlier claim and a matching dispatch id can be claimed, so any
        duplicate or late delivery loses."""
        row = pg_fetch_one(
            """
            update jobs
               set status = 'running', lease_owner = %(owner)s, heartbeat_at = now(),
                   attempts = attempts + 1, started_at = coalesce(started_at, now()),
                   worker_queue = coalesce(worker_queue, %(queue)s),
                   metadata = metadata || jsonb_build_object('status', 'running')
             where id = %(id)s
               and status = 'queued'
               and attempts = 0
               and dispatch_id is not distinct from %(dispatch_id)s
            returning id
            """,
            {"id": record_id, "owner": owner, "queue": worker_queue, "dispatch_id": dispatch_id},
        )
        return row is not None

    def lease_state(self, record_id: str) -> dict[str, Any] | None:
        """The fields a claim or heartbeat needs to explain why it lost, with
        the heartbeat's age measured on the database clock."""
        return pg_fetch_one(
            """
            select status, dispatch_id, lease_owner, attempts,
                   extract(epoch from now() - heartbeat_at) as heartbeat_age,
                   metadata->>'active_pid' as active_pid,
                   metadata->>'active_host' as active_host
              from jobs where id = %s
            """,
            (record_id,),
        )

    def heartbeat(self, record_id: str, owner: str) -> bool:
        row = pg_fetch_one(
            """
            update jobs set heartbeat_at = now()
             where id = %s and lease_owner = %s and status = 'running'
            returning id
            """,
            (record_id, owner),
        )
        return row is not None

    def finish(self, record_id: str, status: str, owner: str | None) -> bool:
        """Move a job to a terminal status if the caller still holds it (or,
        with owner None, if nobody does). Rows from before leases existed are
        finished by the snapshot save as before, so they report True."""
        row = pg_fetch_one(
            """
            update jobs
               set status = %(status)s, lease_owner = null,
                   completed_at = coalesce(completed_at, now()),
                   metadata = metadata || jsonb_build_object('status', %(status)s)
             where id = %(id)s
               and status in ('queued', 'running')
               and lease_owner is not distinct from %(owner)s
            returning id
            """,
            {"id": record_id, "status": status, "owner": owner},
        )
        if row is not None:
            return True
        state = pg_fetch_one("select dispatch_id from jobs where id = %s", (record_id,))
        return state is not None and state["dispatch_id"] is None

    def fail_if_lease(self, record_id: str, *, expected_owner: str | None, error: dict[str, Any]) -> bool:
        """Fail a running job whose lease holder is known to be gone."""
        row = pg_fetch_one(
            """
            update jobs
               set status = 'failed', lease_owner = null, completed_at = now(), error = %(error)s::jsonb,
                   metadata = metadata || jsonb_build_object(
                       'status', 'failed', 'error', %(error)s::jsonb, 'completed_at', now())
             where id = %(id)s and status = 'running'
               and lease_owner is not distinct from %(owner)s
            returning id
            """,
            {"id": record_id, "owner": expected_owner, "error": psycopg2.extras.Json(error)},
        )
        return row is not None

    # -- sweeper queries --------------------------------------------------------

    def fail_where(self, where_sql: str, params: dict[str, Any], *, error: dict[str, Any]) -> list[dict[str, Any]]:
        """Fail every job matching a sweeper rule in one statement; returns
        the ids and stages of the rows it changed. where_sql is a fixed
        fragment from the sweeper, never user input."""
        return pg_fetch_all(
            f"""
            update jobs
               set status = 'failed', lease_owner = null, completed_at = now(), error = %(error)s::jsonb,
                   metadata = metadata || jsonb_build_object(
                       'status', 'failed',
                       'error', %(error)s::jsonb || jsonb_build_object('stage', coalesce(metadata->>'current_stage', 'unknown')),
                       'completed_at', now())
             where {where_sql}
            returning id, user_id, coalesce(metadata->>'current_stage', 'unknown') as stage
            """,
            {**params, "error": psycopg2.extras.Json(error)},
        )

    def lost_queue_candidates(self, worker_queue: str, grace_seconds: int) -> list[dict[str, Any]]:
        return pg_fetch_all(
            """
            select id, dispatch_id, metadata->'request' as request from jobs
             where worker_queue = %s and status = 'queued' and attempts = 0
               and dispatched_at < now() - make_interval(secs => %s)
            """,
            (worker_queue, grace_seconds),
        )

    def redispatch(self, record_id: str, *, old_dispatch_id: str | None, new_dispatch_id: str) -> bool:
        row = pg_fetch_one(
            """
            update jobs
               set dispatch_id = %(new)s, dispatched_at = now(),
                   metadata = metadata || jsonb_build_object('dispatch_id', %(new)s)
             where id = %(id)s and status = 'queued' and attempts = 0
               and dispatch_id is not distinct from %(old)s
            returning id
            """,
            {"id": record_id, "old": old_dispatch_id, "new": new_dispatch_id},
        )
        return row is not None

    def list_by_user_page(self, user_id: str, limit: int, offset: int) -> tuple[list[JobRecord], int]:
        """One page of a user's jobs, newest first, with the user's total."""
        rows = pg_fetch_all(
            """
            select *, count(*) over () as total_count
            from jobs where user_id = %s
            order by created_at desc
            limit %s offset %s
            """,
            (user_id, limit, offset),
        )
        if rows:
            total = int(rows[0]["total_count"])
        else:
            count_row = pg_fetch_one("select count(*) as n from jobs where user_id = %s", (user_id,))
            total = int(count_row["n"]) if count_row else 0
        for row in rows:
            row.pop("total_count", None)
        return [self._normalize_output(row) for row in rows], total

    def queue_position(self, record_id: str) -> int:
        """How many queued jobs were created no later than this one (itself included)."""
        row = pg_fetch_one(
            """
            select count(*) as n from jobs
            where status = 'queued'
              and created_at <= (select created_at from jobs where id = %s)
            """,
            (record_id,),
        )
        return int(row["n"]) if row else 0

    def get_job(self, record_id: str) -> JobRecord | None:
        self._log("select_one", record_id)
        row = pg_fetch_one("select * from jobs where id = %s limit 1", (record_id,))
        return self._normalize_output(row) if row is not None else None

    def upsert_job(self, data: Mapping[str, Any]) -> None:
        """Write a full job snapshot, except that a terminal status is sticky.

        Callers build `data` from an earlier load_job() read, so this is a
        read-modify-write. Cancellation does NOT go through here — it lands via
        the atomic update_job_partial() — so a save whose snapshot predates a
        cancellation would otherwise write the stale status back and silently
        un-cancel the job. update_stage() runs on every progress tick, so that
        window is hit easily during a long render.

        Once the row is cancelled/completed/failed, no snapshot may move it out
        of that state: the decision is made inside this single statement against
        the row's committed value, so there is no window between check and write.
        The metadata blob is still replaced wholesale (job dicts legitimately
        drop keys, e.g. active_pid), but its status key is re-pinned to the
        preserved value so the blob cannot contradict the column.
        """
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("id"))

        pg_execute(
            """
            insert into jobs (id, status, source_type, source_value, progress, user_id,
                              created_at, started_at, completed_at, error, metadata,
                              job_type, worker_id, retry_count)
            values (%(id)s, %(status)s, %(source_type)s, %(source_value)s, %(progress)s, %(user_id)s,
                    coalesce(%(created_at)s, now()), %(started_at)s, %(completed_at)s,
                    %(error)s, %(metadata)s, %(job_type)s, %(worker_id)s, %(retry_count)s)
            on conflict (id) do update set
                status = case
                    when jobs.status in ('cancelled', 'completed', 'failed') then jobs.status
                    -- Leased jobs (dispatch_id set) change status only through
                    -- claim / finish / the sweeper, so a lagging snapshot can't
                    -- undo a requeue or failure. Cancellation is always allowed.
                    when jobs.dispatch_id is not null and excluded.status <> 'cancelled' then jobs.status
                    else excluded.status
                end,
                source_type = excluded.source_type,
                source_value = excluded.source_value,
                progress = excluded.progress,
                user_id = excluded.user_id,
                started_at = excluded.started_at,
                -- A finished leased job keeps the error and completion time it
                -- ended with; a late snapshot from its worker can't erase them.
                completed_at = case
                    when jobs.dispatch_id is not null and jobs.status in ('cancelled', 'completed', 'failed')
                        then jobs.completed_at
                    else excluded.completed_at
                end,
                error = case
                    when jobs.dispatch_id is not null and jobs.status in ('cancelled', 'completed', 'failed')
                        then jobs.error
                    else excluded.error
                end,
                metadata = case
                    when jobs.dispatch_id is not null and jobs.status in ('cancelled', 'completed', 'failed')
                        then excluded.metadata || jsonb_strip_nulls(jsonb_build_object(
                            'status', jobs.status,
                            'error', jobs.metadata->'error',
                            'completed_at', jobs.metadata->'completed_at'))
                    when jobs.status in ('cancelled', 'completed', 'failed')
                        then excluded.metadata || jsonb_build_object('status', jobs.status)
                    when jobs.dispatch_id is not null and excluded.status <> 'cancelled'
                        then excluded.metadata || jsonb_build_object('status', jobs.status)
                    else excluded.metadata
                end,
                job_type = excluded.job_type,
                worker_id = excluded.worker_id,
                retry_count = excluded.retry_count
            """,
            self._to_row_params(payload),
        )

    def update_job(
        self,
        record_id: str,
        data: Mapping[str, Any],
        *,
        merge_existing: bool = True,
    ) -> None:
        """Update only the fields present in `data`.

        When `merge_existing` is True (default), this is done atomically via
        the `update_job_partial` Postgres function (db/init/001_schema.sql)
        — a single UPDATE statement that only touches the columns/metadata
        keys actually present in `data`, so concurrent writers touching
        different fields can't clobber each other (no read-modify-write
        race). When False, `data` is written as a full-row overwrite —
        only use this for a full snapshot you know is current.
        """
        if merge_existing:
            self._update_job_atomic(record_id, data)
            return

        payload = self._normalize_input(data, require_id=False)
        payload.pop("id", None)
        self._log("update", record_id)

        pg_execute(
            """
            update jobs set
                status = %(status)s, source_type = %(source_type)s, source_value = %(source_value)s,
                progress = %(progress)s, user_id = %(user_id)s, started_at = %(started_at)s,
                completed_at = %(completed_at)s, error = %(error)s, metadata = %(metadata)s,
                job_type = %(job_type)s, worker_id = %(worker_id)s, retry_count = %(retry_count)s
            where id = %(id)s
            """,
            {**self._to_row_params(payload), "id": record_id},
        )

    def _update_job_atomic(self, record_id: str, data: Mapping[str, Any]) -> None:
        normalized = self._normalize_value(dict(data))
        normalized.pop("id", None)

        metadata_patch: dict[str, Any] = {}
        explicit_metadata = normalized.pop("metadata", None)
        if isinstance(explicit_metadata, Mapping):
            metadata_patch.update(explicit_metadata)

        columns: dict[str, Any] = {}
        for key, value in normalized.items():
            if key in self._schema_columns:
                columns[key] = value
            else:
                # Any non-schema key (e.g. legacy job_id/type/progress_percent,
                # or a caller-defined field) is folded into metadata, same as
                # _build_metadata does for create_job/upsert_job.
                metadata_patch[key] = value

        self._log("update_partial", record_id)

        pg_execute(
            "select update_job_partial(%s, %s, %s)",
            (
                record_id,
                psycopg2.extras.Json(columns) if columns else None,
                psycopg2.extras.Json(metadata_patch) if metadata_patch else None,
            ),
        )

    def list_jobs(self) -> list[JobRecord]:
        self._log("select_many")
        rows = pg_fetch_all("select * from jobs order by created_at desc")
        return [self._normalize_output(row) for row in rows]

    def list_by_user(self, user_id: str) -> list[JobRecord]:
        rows = pg_fetch_all("select * from jobs where user_id = %s", (user_id,))
        return [self._normalize_output(row) for row in rows]

    def delete_job(self, record_id: str) -> bool:
        self._log("delete", record_id)
        pg_execute("delete from jobs where id = %s", (record_id,))
        return True

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug(
            "Postgres repository operation: table=%s operation=%s id=%s",
            self.table_name,
            operation,
            record_id,
        )

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Wrap jsonb-bound fields with psycopg2.extras.Json so they
        serialize correctly; leave everything else as-is."""
        params = dict(payload)
        params.setdefault("id", None)
        for jsonb_field in ("metadata", "error"):
            if jsonb_field in params and params[jsonb_field] is not None:
                params[jsonb_field] = psycopg2.extras.Json(params[jsonb_field])
        return params

    def _normalize_input(
        self,
        data: Mapping[str, Any],
        *,
        require_id: bool = True,
    ) -> dict[str, Any]:
        normalized = self._normalize_value(dict(data))
        record_id = self._coerce_record_id(normalized)

        if require_id and not record_id:
            raise ValueError("JobRepository requires an 'id' value for job records")

        metadata = self._build_metadata(normalized)
        progress_value = self._coerce_progress(normalized)
        source_type = self._coerce_source_type(normalized, metadata)
        source_value = self._coerce_source_value(normalized, metadata)
        job_type = normalized.get("job_type")
        if job_type is None:
            job_type = normalized.get("type")

        payload: dict[str, Any] = {
            "status": normalized.get("status"),
            "source_type": source_type,
            "source_value": source_value,
            "progress": progress_value,
            "user_id": normalized.get("user_id"),
            "created_at": normalized.get("created_at"),
            "started_at": normalized.get("started_at"),
            "completed_at": normalized.get("completed_at"),
            "error": normalized.get("error"),
            "metadata": metadata,
            "job_type": job_type,
            "worker_id": normalized.get("worker_id"),
            "retry_count": self._coerce_retry_count(normalized.get("retry_count")),
        }

        if record_id:
            payload["id"] = record_id

        return payload

    def _normalize_output(self, data: Mapping[str, Any]) -> JobRecord:
        normalized = self._normalize_value(dict(data))
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        output: JobRecord = {
            "id": str(normalized.get("id")),
            "status": normalized.get("status"),
            "source_type": normalized.get("source_type"),
            "source_value": normalized.get("source_value"),
            "progress": self._coerce_progress(normalized),
            "user_id": normalized.get("user_id"),
            "created_at": normalized.get("created_at"),
            "started_at": normalized.get("started_at"),
            "completed_at": normalized.get("completed_at"),
            "error": normalized.get("error"),
            "metadata": metadata,
            "job_type": normalized.get("job_type"),
            "worker_id": normalized.get("worker_id"),
            "retry_count": self._coerce_retry_count(normalized.get("retry_count")),
        }
        return JobRecord(**output)

    def _build_metadata(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        existing_metadata = normalized.get("metadata")
        metadata: dict[str, Any] = {}
        if isinstance(existing_metadata, dict):
            metadata.update(existing_metadata)

        for key, value in normalized.items():
            if key in self._schema_columns or key == "type":
                continue
            if key == "id":
                continue
            if key == "job_id":
                continue
            if key == "progress_percent":
                metadata.setdefault("progress_percent", value)
                continue
            metadata[key] = value

        if "job_id" in normalized:
            metadata.setdefault("legacy_job_id", normalized["job_id"])
        if "type" in normalized:
            metadata.setdefault("legacy_type", normalized["type"])

        return metadata

    def _coerce_record_id(self, normalized: Mapping[str, Any]) -> str | None:
        record_id = normalized.get("id")
        if record_id in (None, ""):
            record_id = normalized.get("job_id")
        if record_id in (None, ""):
            return None
        return str(record_id)

    def _coerce_progress(self, normalized: Mapping[str, Any]) -> int:
        value = normalized.get("progress")
        if value is None:
            value = normalized.get("progress_percent")
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_retry_count(self, value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_source_type(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        explicit = normalized.get("source_type")
        if explicit not in (None, ""):
            return str(explicit)

        if isinstance(normalized.get("source_value"), str) and normalized.get("source_value"):
            return "explicit"

        prompt = self._extract_prompt(normalized, metadata)
        if prompt:
            return "prompt"

        return "metadata"

    def _coerce_source_value(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        explicit = normalized.get("source_value")
        if explicit not in (None, ""):
            return self._stringify_source_value(explicit)

        prompt = self._extract_prompt(normalized, metadata)
        if prompt:
            return prompt

        job_type = normalized.get("job_type")
        if job_type not in (None, ""):
            return str(job_type)

        legacy_type = normalized.get("type")
        if legacy_type not in (None, ""):
            return str(legacy_type)

        return None

    def _extract_prompt(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        for candidate in (
            normalized.get("prompt"),
            metadata.get("prompt"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate

        for container_name in ("request", "payload"):
            container = metadata.get(container_name)
            if isinstance(container, dict):
                prompt = container.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
        return None

    def _stringify_source_value(self, value: Any) -> str:
        normalized = self._normalize_value(value)
        if isinstance(normalized, str):
            return normalized
        if isinstance(normalized, (int, float, bool)):
            return str(normalized)
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)

    def _normalize_value(self, value: Any) -> Any:
        return normalize_value(value)
