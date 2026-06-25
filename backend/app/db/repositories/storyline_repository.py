from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.storylines")


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
    table_name = "storylines"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create(self, data: Mapping[str, Any]) -> StorylineRecord:
        payload = self._to_row(data)
        self._log("insert", payload.get("id"))
        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception("storylines insert failed id=%s", payload.get("id"))
            raise
        return self._from_row(response.data[0]) if response.data else payload  # type: ignore[return-value]

    def upsert(self, data: Mapping[str, Any]) -> StorylineRecord:
        payload = self._to_row(data)
        self._log("upsert", payload.get("id"))
        try:
            response = self._table.upsert(payload).execute()
        except Exception:
            logger.exception("storylines upsert failed id=%s", payload.get("id"))
            raise
        return self._from_row(response.data[0]) if response.data else payload  # type: ignore[return-value]

    def get(self, storyline_id: str) -> StorylineRecord | None:
        self._log("select_one", storyline_id)
        try:
            response = self._table.select("*").eq("id", storyline_id).limit(1).execute()
        except Exception:
            logger.exception("storylines select failed id=%s", storyline_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def list(self, limit: int = 100) -> list[StorylineRecord]:
        self._log("select_many")
        try:
            response = (
                self._table.select(
                    "id,job_id,title,logline,user_prompt,target_duration_seconds,is_validated,created_at"
                )
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("storylines list failed")
            raise
        return [self._from_row(row) for row in response.data or []]

    def get_latest(self) -> StorylineRecord | None:
        self._log("select_latest")
        try:
            response = (
                self._table.select("*")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception("storylines get_latest failed")
            raise
        return self._from_row(response.data[0]) if response.data else None

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        # Support both top-level and nested storyline.{title,logline,...}
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

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("storylines op=%s id=%s", operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
