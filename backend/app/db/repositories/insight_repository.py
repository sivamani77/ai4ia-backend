from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.insights")


class InsightRecord(TypedDict, total=False):
    id: str
    query: str
    title: str | None
    summary: str | None
    filters: dict
    patterns: list
    contrasts: list
    recommended_clips: list
    unsupported_claims: list
    evidence_pool: list
    evidence_coverage: float
    models: dict
    created_at: str | None


class InsightRepository:
    table_name = "insights"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create(self, data: Mapping[str, Any]) -> InsightRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception("insights insert failed id=%s", payload.get("id"))
            raise
        return self._from_row(response.data[0]) if response.data else payload  # type: ignore[return-value]

    def get(self, insight_id: str) -> InsightRecord | None:
        self._log("select_one", insight_id)
        try:
            response = self._table.select("*").eq("id", insight_id).limit(1).execute()
        except Exception:
            logger.exception("insights select failed id=%s", insight_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def list(self, limit: int = 100) -> list[InsightRecord]:
        self._log("select_many")
        try:
            response = (
                self._table.select("id,query,title,evidence_coverage,created_at")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("insights list failed")
            raise
        return [self._from_row(row) for row in response.data or []]

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        insight_id = d.get("id") or d.get("insight_id")
        return {
            "id": insight_id,
            "query": d.get("query", ""),
            "title": d.get("title"),
            "summary": d.get("summary"),
            "filters": d.get("filters") or {},
            "patterns": d.get("patterns") or [],
            "contrasts": d.get("contrasts") or [],
            "recommended_clips": d.get("recommended_clips") or [],
            "unsupported_claims": d.get("unsupported_claims") or [],
            "evidence_pool": d.get("evidence_pool") or [],
            "evidence_coverage": float(d.get("evidence_coverage") or 0.0),
            "models": d.get("models") or {},
            "created_at": d.get("created_at"),
        }

    def _from_row(self, row: Mapping[str, Any]) -> InsightRecord:
        d = self._normalize(dict(row))
        return InsightRecord(
            id=d.get("id", ""),
            query=d.get("query", ""),
            title=d.get("title"),
            summary=d.get("summary"),
            filters=d.get("filters") or {},
            patterns=d.get("patterns") or [],
            contrasts=d.get("contrasts") or [],
            recommended_clips=d.get("recommended_clips") or [],
            unsupported_claims=d.get("unsupported_claims") or [],
            evidence_pool=d.get("evidence_pool") or [],
            evidence_coverage=float(d.get("evidence_coverage") or 0.0),
            models=d.get("models") or {},
            created_at=d.get("created_at"),
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("insights op=%s id=%s", operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
