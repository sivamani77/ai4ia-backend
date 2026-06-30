from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
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

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create(self, data: Mapping[str, Any]) -> DecisionRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception("decision_records insert failed id=%s", payload.get("id"))
            raise
        return self._from_row(response.data[0]) if response.data else payload  # type: ignore[return-value]

    def get(self, record_id: str) -> DecisionRecord | None:
        self._log("select_one", record_id)
        try:
            response = self._table.select("*").eq("id", record_id).limit(1).execute()
        except Exception:
            logger.exception("decision_records select failed id=%s", record_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def get_by_job(self, job_id: str) -> DecisionRecord | None:
        self._log("select_by_job", job_id)
        try:
            response = (
                self._table.select("*")
                .eq("job_id", job_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception("decision_records select_by_job failed job_id=%s", job_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def get_by_insight(self, insight_id: str) -> DecisionRecord | None:
        self._log("select_by_insight", insight_id)
        try:
            response = (
                self._table.select("*")
                .eq("insight_id", insight_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception("decision_records select_by_insight failed insight_id=%s", insight_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def list_by_user(self, user_id: str, limit: int = 50) -> list[DecisionRecord]:
        self._log("select_by_user", user_id)
        try:
            response = (
                self._table.select(
                    "id,job_id,insight_id,user_id,subject,outcome,decisions,agents_executed,total_duration_ms,created_at"
                )
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("decision_records list_by_user failed user_id=%s", user_id)
            raise
        return [self._from_row(row) for row in response.data or []]

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
