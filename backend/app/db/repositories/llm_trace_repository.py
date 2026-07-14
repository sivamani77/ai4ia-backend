from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.postgres import pg_execute, pg_fetch_one, pg_fetch_all
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.llm_traces")


class LLMTraceRecord(TypedDict, total=False):
    id: str
    job_id: str | None
    insight_id: str | None
    user_id: str | None
    stage: str | None
    agent: str
    model: str
    call_type: str
    decision_label: str | None
    prompt_hash: str | None
    response_hash: str | None
    latency_ms: int | None
    temperature: float | None
    retry_count: int
    error: str | None
    called_at: str | None


class LLMTraceRepository:
    table_name = "llm_call_traces"

    def create(self, data: Mapping[str, Any]) -> LLMTraceRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        row = pg_fetch_one(
            """
            insert into llm_call_traces
                (id, job_id, insight_id, user_id, stage, agent, model, call_type,
                 decision_label, prompt_hash, response_hash, latency_ms, temperature,
                 retry_count, error, called_at)
            values
                (%(id)s, %(job_id)s, %(insight_id)s, %(user_id)s, %(stage)s, %(agent)s,
                 %(model)s, %(call_type)s, %(decision_label)s, %(prompt_hash)s,
                 %(response_hash)s, %(latency_ms)s, %(temperature)s, %(retry_count)s,
                 %(error)s, coalesce(%(called_at)s, now()))
            returning *
            """,
            payload,
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def delete_by_job(self, job_id: str) -> None:
        pg_execute("delete from llm_call_traces where job_id = %s", (job_id,))

    def delete_by_user(self, user_id: str) -> None:
        pg_execute("delete from llm_call_traces where user_id = %s", (user_id,))

    def list_by_job(self, job_id: str) -> list[LLMTraceRecord]:
        self._log("select_by_job", job_id)
        rows = pg_fetch_all(
            "select * from llm_call_traces where job_id = %s order by called_at asc",
            (job_id,),
        )
        return [self._from_row(row) for row in rows]

    def list_by_insight(self, insight_id: str) -> list[LLMTraceRecord]:
        self._log("select_by_insight", insight_id)
        rows = pg_fetch_all(
            "select * from llm_call_traces where insight_id = %s order by called_at asc",
            (insight_id,),
        )
        return [self._from_row(row) for row in rows]

    def list_all(self, limit: int = 200, offset: int = 0) -> list[LLMTraceRecord]:
        self._log("select_all")
        rows = pg_fetch_all(
            "select * from llm_call_traces order by called_at desc limit %s offset %s",
            (limit, offset),
        )
        return [self._from_row(row) for row in rows]

    def list_by_agent(self, agent: str, limit: int = 100) -> list[LLMTraceRecord]:
        self._log("select_by_agent", agent)
        rows = pg_fetch_all(
            """
            select * from llm_call_traces
            where agent = %s order by called_at desc limit %s
            """,
            (agent, limit),
        )
        return [self._from_row(row) for row in rows]

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        return {
            "id": d.get("id"),
            "job_id": d.get("job_id"),
            "insight_id": d.get("insight_id"),
            "user_id": d.get("user_id"),
            "stage": d.get("stage"),
            "agent": d.get("agent", ""),
            "model": d.get("model", ""),
            "call_type": d.get("call_type", ""),
            "decision_label": d.get("decision_label"),
            "prompt_hash": d.get("prompt_hash"),
            "response_hash": d.get("response_hash"),
            "latency_ms": d.get("latency_ms"),
            "temperature": d.get("temperature"),
            "retry_count": d.get("retry_count", 0),
            "error": d.get("error"),
            "called_at": d.get("called_at"),
        }

    def _from_row(self, row: Mapping[str, Any]) -> LLMTraceRecord:
        d = self._normalize(dict(row))
        return LLMTraceRecord(
            id=d.get("id", ""),
            job_id=d.get("job_id"),
            insight_id=d.get("insight_id"),
            user_id=d.get("user_id"),
            stage=d.get("stage"),
            agent=d.get("agent", ""),
            model=d.get("model", ""),
            call_type=d.get("call_type", ""),
            decision_label=d.get("decision_label"),
            prompt_hash=d.get("prompt_hash"),
            response_hash=d.get("response_hash"),
            latency_ms=d.get("latency_ms"),
            temperature=d.get("temperature"),
            retry_count=d.get("retry_count", 0),
            error=d.get("error"),
            called_at=d.get("called_at"),
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("llm_traces op=%s id=%s", operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
