from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
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

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create(self, data: Mapping[str, Any]) -> LLMTraceRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception("llm_traces insert failed id=%s", payload.get("id"))
            raise
        return self._from_row(response.data[0]) if response.data else payload  # type: ignore[return-value]

    def delete_by_job(self, job_id: str) -> None:
        self._table.delete().eq("job_id", job_id).execute()

    def delete_by_user(self, user_id: str) -> None:
        self._table.delete().eq("user_id", user_id).execute()

    def list_by_job(self, job_id: str) -> list[LLMTraceRecord]:
        self._log("select_by_job", job_id)
        try:
            response = (
                self._table.select("*")
                .eq("job_id", job_id)
                .order("called_at", desc=False)
                .execute()
            )
        except Exception:
            logger.exception("llm_traces list_by_job failed job_id=%s", job_id)
            raise
        return [self._from_row(row) for row in response.data or []]

    def list_by_insight(self, insight_id: str) -> list[LLMTraceRecord]:
        self._log("select_by_insight", insight_id)
        try:
            response = (
                self._table.select("*")
                .eq("insight_id", insight_id)
                .order("called_at", desc=False)
                .execute()
            )
        except Exception:
            logger.exception("llm_traces list_by_insight failed insight_id=%s", insight_id)
            raise
        return [self._from_row(row) for row in response.data or []]

    def list_all(self, limit: int = 200, offset: int = 0) -> list[LLMTraceRecord]:
        self._log("select_all")
        try:
            response = (
                self._table.select("*")
                .order("called_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
        except Exception:
            logger.exception("llm_traces list_all failed")
            raise
        return [self._from_row(row) for row in response.data or []]

    def list_by_agent(self, agent: str, limit: int = 100) -> list[LLMTraceRecord]:
        self._log("select_by_agent", agent)
        try:
            response = (
                self._table.select("*")
                .eq("agent", agent)
                .order("called_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("llm_traces list_by_agent failed agent=%s", agent)
            raise
        return [self._from_row(row) for row in response.data or []]

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
