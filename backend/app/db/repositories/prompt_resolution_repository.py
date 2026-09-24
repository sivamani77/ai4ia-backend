"""prompt_resolutions: one row per request, updated as each layer decides."""
from __future__ import annotations

import logging
from typing import Any, Mapping

import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one

logger = logging.getLogger("ai4ia.db.repositories.prompt_resolutions")

_JSONB_FIELDS = ("flags", "slots", "dropped_slots", "suggestions", "coverage", "timings_ms")
_SCALAR_FIELDS = (
    "insight_id", "attempt", "intent", "intent_version", "template_hash", "resolution_method",
    "layer_outcome", "reason_code", "user_message", "code_version", "corpus_version",
    "registry_version", "expires_at",
)
_INSERT_FIELDS = frozenset({"id", "job_id", "surface", "user_id", "raw_prompt", *_SCALAR_FIELDS, *_JSONB_FIELDS})


def _param(key: str, value: Any) -> Any:
    return psycopg2.extras.Json(value) if key in _JSONB_FIELDS else value


class PromptResolutionRepository:
    table_name = "prompt_resolutions"

    def create(self, data: Mapping[str, Any]) -> None:
        unknown = set(data) - _INSERT_FIELDS
        if unknown:
            raise ValueError(f"not a prompt_resolutions column: {sorted(unknown)}")
        row = {k: v for k, v in data.items() if v is not None}
        columns = ", ".join(row)
        values = ", ".join(f"%({k})s" for k in row)
        pg_execute(
            f"insert into prompt_resolutions ({columns}) values ({values}) on conflict (id) do nothing",
            {k: _param(k, v) for k, v in row.items()},
        )

    def update_by_job(
        self,
        job_id: str,
        *,
        fields: Mapping[str, Any] | None = None,
        merge: Mapping[str, Mapping[str, Any]] | None = None,
        only_if_outcome: str | None = None,
    ) -> None:
        """Set scalar `fields` and shallow-merge `merge` into jsonb columns.
        With `only_if_outcome`, rows whose layer_outcome differs are left alone."""
        sets, params = ["updated_at = now()"], {"job_id": job_id}
        for key, value in (fields or {}).items():
            if key not in _SCALAR_FIELDS:
                raise ValueError(f"not an updatable column: {key}")
            sets.append(f"{key} = %({key})s")
            params[key] = value
        for key, value in (merge or {}).items():
            if key not in _JSONB_FIELDS:
                raise ValueError(f"not a jsonb column: {key}")
            sets.append(f"{key} = {key} || %(m_{key})s")
            params[f"m_{key}"] = psycopg2.extras.Json(dict(value))
        where = "job_id = %(job_id)s"
        if only_if_outcome is not None:
            where += " and layer_outcome = %(only_if_outcome)s"
            params["only_if_outcome"] = only_if_outcome
        pg_execute(f"update prompt_resolutions set {', '.join(sets)} where {where}", params)

    def get_by_job(self, job_id: str) -> dict[str, Any] | None:
        return pg_fetch_one(
            "select * from prompt_resolutions where job_id = %s order by attempt desc limit 1",
            (job_id,),
        )

    def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        return pg_fetch_all(
            "select * from prompt_resolutions where user_id = %s order by created_at", (user_id,)
        )

    def delete_by_job(self, job_id: str) -> None:
        pg_execute("delete from prompt_resolutions where job_id = %s", (job_id,))

    def delete_by_user(self, user_id: str) -> None:
        pg_execute("delete from prompt_resolutions where user_id = %s", (user_id,))

    def disconnect_by_user(self, user_id: str) -> None:
        """Keep the records but remove the link to the account (privacy policy 8.2)."""
        pg_execute("update prompt_resolutions set user_id = null where user_id = %s", (user_id,))

    def purge_expired(self) -> int:
        row = pg_fetch_one(
            """
            with gone as (delete from prompt_resolutions where expires_at < now() returning 1)
            select count(*) as n from gone
            """
        )
        return int((row or {}).get("n") or 0)
