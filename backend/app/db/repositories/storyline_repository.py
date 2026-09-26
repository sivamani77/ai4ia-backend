from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.core.settings import settings
from app.db.postgres import pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.storylines")

_JSONB_FIELDS = ("sections", "selected_candidates", "export_timeline", "editorial_quality_gates", "full_payload")


class StorylineRecord(TypedDict, total=False):
    id: str
    job_id: str | None
    title: str | None
    logline: str | None
    user_prompt: str | None
    target_duration_seconds: int | None
    is_validated: bool
    sections: list
    selected_candidates: list
    export_timeline: list
    editorial_quality_gates: dict
    full_payload: dict
    created_at: str | None


class StorylineRepository:
    """Storylines live in Postgres (DATABASE_URL). Reads and writes both go
    there; the legacy Supabase read fallback was removed once the pipeline
    project was fully backfilled."""

    table_name = "storylines"

    def create(self, data: Mapping[str, Any]) -> StorylineRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        row = pg_fetch_one(
            """
            insert into storylines (id, job_id, title, logline, user_prompt,
                                    target_duration_seconds, is_validated, sections,
                                    selected_candidates, export_timeline,
                                    editorial_quality_gates, full_payload, created_at)
            values (%(id)s, %(job_id)s, %(title)s, %(logline)s, %(user_prompt)s,
                    %(target_duration_seconds)s, %(is_validated)s, %(sections)s,
                    %(selected_candidates)s, %(export_timeline)s,
                    %(editorial_quality_gates)s, %(full_payload)s, coalesce(%(created_at)s, now()))
            returning *
            """,
            self._to_row_params(payload),
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def upsert(self, data: Mapping[str, Any]) -> StorylineRecord:
        payload = self._to_row(data)
        self._log("upsert", payload.get("id"))
        row = pg_fetch_one(
            """
            insert into storylines (id, job_id, title, logline, user_prompt,
                                    target_duration_seconds, is_validated, sections,
                                    selected_candidates, export_timeline,
                                    editorial_quality_gates, full_payload, created_at)
            values (%(id)s, %(job_id)s, %(title)s, %(logline)s, %(user_prompt)s,
                    %(target_duration_seconds)s, %(is_validated)s, %(sections)s,
                    %(selected_candidates)s, %(export_timeline)s,
                    %(editorial_quality_gates)s, %(full_payload)s, coalesce(%(created_at)s, now()))
            on conflict (id) do update set
                job_id = excluded.job_id,
                title = excluded.title,
                logline = excluded.logline,
                user_prompt = excluded.user_prompt,
                target_duration_seconds = excluded.target_duration_seconds,
                is_validated = excluded.is_validated,
                sections = excluded.sections,
                selected_candidates = excluded.selected_candidates,
                export_timeline = excluded.export_timeline,
                editorial_quality_gates = excluded.editorial_quality_gates,
                full_payload = excluded.full_payload
            returning *
            """,
            self._to_row_params(payload),
        )
        return self._from_row(row) if row is not None else payload  # type: ignore[return-value]

    def get(self, storyline_id: str) -> StorylineRecord | None:
        self._log("select_one", storyline_id)
        row = pg_fetch_one("select * from storylines where id = %s limit 1", (storyline_id,))
        return self._from_row(row) if row is not None else None

    def list(self, limit: int = 100) -> list[StorylineRecord]:
        self._log("select_many")
        rows = pg_fetch_all(
            "select id, job_id, title, logline, user_prompt, target_duration_seconds, "
            "is_validated, created_at from storylines order by created_at desc limit %s",
            (limit,),
        )
        return [self._from_row(row) for row in rows]

    def get_latest(self) -> StorylineRecord | None:
        self._log("select_latest")
        row = pg_fetch_one("select * from storylines order by created_at desc limit 1")
        return self._from_row(row) if row is not None else None

    # -- row shaping ----------------------------------------------------------

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        story = d.get("storyline") or {}
        return {
            "id": d.get("id") or d.get("storyline_id"),
            "job_id": d.get("job_id"),
            "title": d.get("title") or story.get("title"),
            "logline": d.get("logline") or story.get("logline"),
            "user_prompt": d.get("user_prompt"),
            "target_duration_seconds": d.get("target_duration_seconds"),
            "is_validated": bool(d.get("is_validated", False)),
            "sections": (d.get("sections") or story.get("sections")) or [],
            "selected_candidates": d.get("selected_candidates") or [],
            "export_timeline": d.get("export_timeline") or [],
            "editorial_quality_gates": d.get("editorial_quality_gates") or {},
            "full_payload": d,
            "created_at": d.get("created_at"),
        }

    def _from_row(self, row: Mapping[str, Any]) -> StorylineRecord:
        d = self._normalize(dict(row))
        return StorylineRecord(
            id=d.get("id", ""),
            job_id=d.get("job_id"),
            title=d.get("title"),
            logline=d.get("logline"),
            user_prompt=d.get("user_prompt"),
            target_duration_seconds=d.get("target_duration_seconds"),
            is_validated=bool(d.get("is_validated", False)),
            sections=d.get("sections") or [],
            selected_candidates=d.get("selected_candidates") or [],
            export_timeline=d.get("export_timeline") or [],
            editorial_quality_gates=d.get("editorial_quality_gates") or {},
            full_payload=d.get("full_payload") or {},
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
