"""llm_call_payloads: the exact text of each LLM request and response."""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one

logger = logging.getLogger("ai4ia.db.repositories.llm_payloads")

_COLUMNS = (
    "id", "trace_id", "job_id", "insight_id", "user_id", "stage", "model", "request_text",
    "response_text", "error", "latency_ms", "size_bytes", "called_at", "expires_at",
)


class LLMPayloadRepository:
    table_name = "llm_call_payloads"

    def create(self, data: Mapping[str, Any]) -> None:
        row = {k: data.get(k) for k in _COLUMNS}
        row["size_bytes"] = len((row["request_text"] or "").encode()) + len((row["response_text"] or "").encode())
        values = ", ".join("coalesce(%(called_at)s, now())" if k == "called_at" else f"%({k})s"
                           for k in _COLUMNS)
        pg_execute(
            f"insert into llm_call_payloads ({', '.join(_COLUMNS)}) values ({values}) "
            "on conflict (id) do nothing",
            row,
        )

    def create_many(self, rows: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for row in rows:
            self.create(row)
            count += 1
        return count

    def list_by_job(self, job_id: str) -> list[dict[str, Any]]:
        return pg_fetch_all(
            "select * from llm_call_payloads where job_id = %s order by called_at", (job_id,)
        )

    def delete_by_job(self, job_id: str) -> None:
        pg_execute("delete from llm_call_payloads where job_id = %s", (job_id,))

    def delete_by_user(self, user_id: str) -> None:
        pg_execute("delete from llm_call_payloads where user_id = %s", (user_id,))

    def disconnect_by_user(self, user_id: str) -> None:
        """Keep the records but remove the link to the account (privacy policy 8.2)."""
        pg_execute("update llm_call_payloads set user_id = null where user_id = %s", (user_id,))

    def purge_expired(self) -> int:
        row = pg_fetch_one(
            """
            with gone as (delete from llm_call_payloads where expires_at < now() returning 1)
            select count(*) as n from gone
            """
        )
        return int((row or {}).get("n") or 0)
