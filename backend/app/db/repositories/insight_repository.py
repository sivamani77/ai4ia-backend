from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.core.settings import settings
from app.db.postgres import pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.insights")

_JSONB_FIELDS = ("filters", "patterns", "contrasts", "recommended_clips", "unsupported_claims", "evidence_pool", "models")


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
    """Insights live in Postgres (DATABASE_URL). Reads and writes both go
    there; the legacy Supabase read fallback was removed once the pipeline
    project was fully backfilled."""

    table_name = "insights"

    def create(self, data: Mapping[str, Any]) -> InsightRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        row = pg_fetch_one(
            """
            insert into insights (id, query, title, summary, filters, patterns, contrasts,
                                  recommended_clips, unsupported_claims, evidence_pool,
                                  evidence_coverage, models, created_at)
            values (%(id)s, %(query)s, %(title)s, %(summary)s, %(filters)s, %(patterns)s,
                    %(contrasts)s, %(recommended_clips)s, %(unsupported_claims)s,
                    %(evidence_pool)s, %(evidence_coverage)s, %(models)s, coalesce(%(created_at)s, now()))
            returning *
            """,
            self._to_row_params(payload),
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def get(self, insight_id: str) -> InsightRecord | None:
        self._log("select_one", insight_id)
        row = pg_fetch_one("select * from insights where id = %s limit 1", (insight_id,))
        return self._from_row(row) if row is not None else None

    def list(self, limit: int = 100) -> list[InsightRecord]:
        self._log("select_many")
        rows = pg_fetch_all(
            "select id, query, title, evidence_coverage, created_at from insights order by created_at desc limit %s",
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    # -- row shaping ----------------------------------------------------------

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

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(payload)
        for field in _JSONB_FIELDS:
            if field in params:
                params[field] = psycopg2.extras.Json(params[field])
        return params

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("Postgres repository operation: table=%s operation=%s id=%s", self.table_name, operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
