from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.db.postgres import pg_execute, pg_fetch_one, pg_fetch_all
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.decision_records")


class DecisionRecord(TypedDict, total=False):
    id: str
    job_id: str | None
    insight_id: str | None
    user_id: str | None
    schema_version: str
    subject: str
    user_prompt: str | None
    accountability: str
    pipeline_version: str | None
    agents_executed: list
    decisions: dict
    models_used: dict
    outcome: str | None
    total_duration_ms: int | None
    expires_at: str | None
    created_at: str | None


class DecisionRecordRepository:
    table_name = "agent_decision_records"

    def create(self, data: Mapping[str, Any]) -> DecisionRecord:
        payload = self._to_row(data)
        self._log("upsert", payload.get("id"))
        row = pg_fetch_one(
            """
            insert into agent_decision_records
                (id, job_id, insight_id, user_id, schema_version, subject, user_prompt,
                 accountability, pipeline_version, agents_executed, decisions, models_used,
                 outcome, total_duration_ms, expires_at, created_at)
            values
                (%(id)s, %(job_id)s, %(insight_id)s, %(user_id)s, %(schema_version)s, %(subject)s,
                 %(user_prompt)s, %(accountability)s, %(pipeline_version)s, %(agents_executed)s,
                 %(decisions)s, %(models_used)s, %(outcome)s, %(total_duration_ms)s,
                 %(expires_at)s, coalesce(%(created_at)s, now()))
            on conflict (id) do update set
                job_id = excluded.job_id,
                insight_id = excluded.insight_id,
                user_id = excluded.user_id,
                schema_version = excluded.schema_version,
                subject = excluded.subject,
                user_prompt = excluded.user_prompt,
                accountability = excluded.accountability,
                pipeline_version = excluded.pipeline_version,
                agents_executed = excluded.agents_executed,
                decisions = excluded.decisions,
                models_used = excluded.models_used,
                outcome = excluded.outcome,
                total_duration_ms = excluded.total_duration_ms,
                expires_at = excluded.expires_at
            returning *
            """,
            self._to_row_params(payload),
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def get(self, record_id: str) -> DecisionRecord | None:
        self._log("select_one", record_id)
        row = pg_fetch_one(
            "select * from agent_decision_records where id = %s limit 1", (record_id,)
        )
        return self._from_row(row) if row is not None else None

    def delete_by_user(self, user_id: str) -> None:
        pg_execute("delete from agent_decision_records where user_id = %s", (user_id,))

    def get_by_job(self, job_id: str) -> DecisionRecord | None:
        self._log("select_by_job", job_id)
        row = pg_fetch_one(
            """
            select * from agent_decision_records
            where job_id = %s order by created_at desc limit 1
            """,
            (job_id,),
        )
        return self._from_row(row) if row is not None else None

    def get_by_insight(self, insight_id: str) -> DecisionRecord | None:
        self._log("select_by_insight", insight_id)
        row = pg_fetch_one(
            """
            select * from agent_decision_records
            where insight_id = %s order by created_at desc limit 1
            """,
            (insight_id,),
        )
        return self._from_row(row) if row is not None else None

    def list_all(self, limit: int = 100, offset: int = 0) -> list[DecisionRecord]:
        self._log("select_all")
        rows = pg_fetch_all(
            """
            select id, job_id, insight_id, user_id, subject, outcome, decisions,
                   models_used, agents_executed, total_duration_ms, created_at
            from agent_decision_records
            order by created_at desc
            limit %s offset %s
            """,
            (limit, offset),
        )
        return [self._from_row(row) for row in rows]

    def list_by_user(self, user_id: str, limit: int = 50) -> list[DecisionRecord]:
        self._log("select_by_user", user_id)
        rows = pg_fetch_all(
            """
            select id, job_id, insight_id, user_id, subject, outcome, decisions,
                   agents_executed, total_duration_ms, created_at
            from agent_decision_records
            where user_id = %s
            order by created_at desc
            limit %s
            """,
            (user_id, limit),
        )
        return [self._from_row(row) for row in rows]

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        return {
            "id": d.get("id"),
            "job_id": d.get("job_id"),
            "insight_id": d.get("insight_id"),
            "user_id": d.get("user_id"),
            "schema_version": d.get("schema_version", "osi-1.0"),
            "subject": d.get("subject", ""),
            "user_prompt": d.get("user_prompt"),
            "accountability": d.get("accountability", "ai4ia-pipeline-v2"),
            "pipeline_version": d.get("pipeline_version"),
            "agents_executed": d.get("agents_executed") or [],
            "decisions": d.get("decisions") or {},
            "models_used": d.get("models_used") or {},
            "outcome": d.get("outcome"),
            "total_duration_ms": d.get("total_duration_ms"),
            "expires_at": d.get("expires_at"),
            "created_at": d.get("created_at"),
        }

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(payload)
        for jsonb_field in ("agents_executed", "decisions", "models_used"):
            params[jsonb_field] = psycopg2.extras.Json(params.get(jsonb_field) or {})
        return params

    def _from_row(self, row: Mapping[str, Any]) -> DecisionRecord:
        d = self._normalize(dict(row))
        return DecisionRecord(
            id=d.get("id", ""),
            job_id=d.get("job_id"),
            insight_id=d.get("insight_id"),
            user_id=d.get("user_id"),
            schema_version=d.get("schema_version", "osi-1.0"),
            subject=d.get("subject", ""),
            user_prompt=d.get("user_prompt"),
            accountability=d.get("accountability", "ai4ia-pipeline-v2"),
            pipeline_version=d.get("pipeline_version"),
            agents_executed=d.get("agents_executed") or [],
            decisions=d.get("decisions") or {},
            models_used=d.get("models_used") or {},
            outcome=d.get("outcome"),
            total_duration_ms=d.get("total_duration_ms"),
            expires_at=d.get("expires_at"),
            created_at=d.get("created_at"),
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("decision_records op=%s id=%s", operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
