from __future__ import annotations

import json
import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.jobs")


class JobRecord(TypedDict, total=False):
    id: str
    status: str
    source_type: str | None
    source_value: str | None
    progress: int
    user_id: str
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    error: Any
    metadata: dict[str, Any]
    job_type: str | None
    worker_id: str | None
    retry_count: int


class JobRepository:
    table_name = "jobs"
    _schema_columns = {
        "id",
        "status",
        "source_type",
        "source_value",
        "progress",
        "user_id",
        "created_at",
        "started_at",
        "completed_at",
        "error",
        "metadata",
        "job_type",
        "worker_id",
        "retry_count",
    }

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def create_job(self, data: Mapping[str, Any]) -> JobRecord:
        payload = self._normalize_input(data)
        record_id = payload.get("id")
        self._log("insert", record_id)

        try:
            response = self._table.insert(payload).execute()
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s id=%s",
                self.table_name,
                "insert",
                record_id,
            )
            raise

        created = self._extract_optional_single(response.data)
        if created is not None:
            return created
        return self.get_job(str(record_id))

    def get_job(self, record_id: str) -> JobRecord | None:
        self._log("select_one", record_id)

        try:
            response = (
                self._table.select("*")
                .eq("id", record_id)
                .limit(1)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s id=%s",
                self.table_name,
                "select_one",
                record_id,
            )
            raise

        return self._extract_optional_single(response.data)

    def upsert_job(self, data: Mapping[str, Any]) -> None:
        payload = self._normalize_input(data)
        self._log("upsert", payload.get("id"))

        try:
            self._table.upsert(payload, on_conflict="id").execute()
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s id=%s",
                self.table_name,
                "upsert",
                payload.get("id"),
            )
            raise

    def update_job(
        self,
        record_id: str,
        data: Mapping[str, Any],
        *,
        merge_existing: bool = True,
    ) -> None:
        source = self._merge_existing_for_update(record_id, data) if merge_existing else data
        payload = self._normalize_input(source, require_id=False)
        payload.pop("id", None)
        self._log("update", record_id)

        try:
            self._table.update(payload).eq("id", record_id).execute()
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s id=%s",
                self.table_name,
                "update",
                record_id,
            )
            raise

    def _merge_existing_for_update(
        self,
        record_id: str,
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing = self.get_job(record_id)
        if existing is None:
            return dict(updates)

        merged: dict[str, Any] = dict(existing.get("metadata") or {})
        merged.update(
            {
                "id": existing.get("id"),
                "status": existing.get("status"),
                "source_type": existing.get("source_type"),
                "source_value": existing.get("source_value"),
                "progress": existing.get("progress"),
                "created_at": existing.get("created_at"),
                "started_at": existing.get("started_at"),
                "completed_at": existing.get("completed_at"),
                "error": existing.get("error"),
                "job_type": existing.get("job_type"),
                "worker_id": existing.get("worker_id"),
                "retry_count": existing.get("retry_count"),
            }
        )

        normalized_updates = self._normalize_value(dict(updates))
        update_metadata = normalized_updates.pop("metadata", None)
        if isinstance(update_metadata, Mapping):
            merged.update(update_metadata)

        merged.update(normalized_updates)
        merged["id"] = record_id
        return merged

    def list_jobs(self) -> list[JobRecord]:
        self._log("select_many")

        try:
            response = (
                self._table.select("*")
                .order("created_at", desc=True)
                .execute()
            )
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s",
                self.table_name,
                "select_many",
            )
            raise

        return [self._normalize_output(row) for row in response.data or []]

    def delete_job(self, record_id: str) -> bool:
        self._log("delete", record_id)

        try:
            self._table.delete().eq("id", record_id).execute()
        except Exception:
            logger.exception(
                "Supabase jobs operation failed: table=%s operation=%s id=%s",
                self.table_name,
                "delete",
                record_id,
            )
            raise

        return True

    def _extract_optional_single(
        self,
        rows: list[dict[str, Any]] | None,
    ) -> JobRecord | None:
        if not rows:
            return None
        return self._normalize_output(rows[0])

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug(
            "Supabase repository operation: table=%s operation=%s id=%s",
            self.table_name,
            operation,
            record_id,
        )

    def _normalize_input(
        self,
        data: Mapping[str, Any],
        *,
        require_id: bool = True,
    ) -> dict[str, Any]:
        normalized = self._normalize_value(dict(data))
        record_id = self._coerce_record_id(normalized)

        if require_id and not record_id:
            raise ValueError("JobRepository requires an 'id' value for job records")

        metadata = self._build_metadata(normalized)
        progress_value = self._coerce_progress(normalized)
        source_type = self._coerce_source_type(normalized, metadata)
        source_value = self._coerce_source_value(normalized, metadata)
        job_type = normalized.get("job_type")
        if job_type is None:
            job_type = normalized.get("type")

        payload: dict[str, Any] = {
            "status": normalized.get("status"),
            "source_type": source_type,
            "source_value": source_value,
            "progress": progress_value,
            "user_id": normalized.get("user_id"),
            "created_at": normalized.get("created_at"),
            "started_at": normalized.get("started_at"),
            "completed_at": normalized.get("completed_at"),
            "error": normalized.get("error"),
            "metadata": metadata,
            "job_type": job_type,
            "worker_id": normalized.get("worker_id"),
            "retry_count": self._coerce_retry_count(normalized.get("retry_count")),
        }

        if record_id:
            payload["id"] = record_id

        return payload

    def _normalize_output(self, data: Mapping[str, Any]) -> JobRecord:
        normalized = self._normalize_value(dict(data))
        metadata = normalized.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        output: JobRecord = {
            "id": str(normalized.get("id")),
            "status": normalized.get("status"),
            "source_type": normalized.get("source_type"),
            "source_value": normalized.get("source_value"),
            "progress": self._coerce_progress(normalized),
            "user_id": normalized.get("user_id"),
            "created_at": normalized.get("created_at"),
            "started_at": normalized.get("started_at"),
            "completed_at": normalized.get("completed_at"),
            "error": normalized.get("error"),
            "metadata": metadata,
            "job_type": normalized.get("job_type"),
            "worker_id": normalized.get("worker_id"),
            "retry_count": self._coerce_retry_count(normalized.get("retry_count")),
        }
        return JobRecord(**output)

    def _build_metadata(self, normalized: Mapping[str, Any]) -> dict[str, Any]:
        existing_metadata = normalized.get("metadata")
        metadata: dict[str, Any] = {}
        if isinstance(existing_metadata, dict):
            metadata.update(existing_metadata)

        for key, value in normalized.items():
            if key in self._schema_columns or key == "type":
                continue
            if key == "id":
                continue
            if key == "job_id":
                continue
            if key == "progress_percent":
                metadata.setdefault("progress_percent", value)
                continue
            metadata[key] = value

        if "job_id" in normalized:
            metadata.setdefault("legacy_job_id", normalized["job_id"])
        if "type" in normalized:
            metadata.setdefault("legacy_type", normalized["type"])

        return metadata

    def _coerce_record_id(self, normalized: Mapping[str, Any]) -> str | None:
        record_id = normalized.get("id")
        if record_id in (None, ""):
            record_id = normalized.get("job_id")
        if record_id in (None, ""):
            return None
        return str(record_id)

    def _coerce_progress(self, normalized: Mapping[str, Any]) -> int:
        value = normalized.get("progress")
        if value is None:
            value = normalized.get("progress_percent")
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_retry_count(self, value: Any) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_source_type(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        explicit = normalized.get("source_type")
        if explicit not in (None, ""):
            return str(explicit)

        if isinstance(normalized.get("source_value"), str) and normalized.get("source_value"):
            return "explicit"

        prompt = self._extract_prompt(normalized, metadata)
        if prompt:
            return "prompt"

        return "metadata"

    def _coerce_source_value(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        explicit = normalized.get("source_value")
        if explicit not in (None, ""):
            return self._stringify_source_value(explicit)

        prompt = self._extract_prompt(normalized, metadata)
        if prompt:
            return prompt

        job_type = normalized.get("job_type")
        if job_type not in (None, ""):
            return str(job_type)

        legacy_type = normalized.get("type")
        if legacy_type not in (None, ""):
            return str(legacy_type)

        return None

    def _extract_prompt(
        self,
        normalized: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str | None:
        for candidate in (
            normalized.get("prompt"),
            metadata.get("prompt"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate

        for container_name in ("request", "payload"):
            container = metadata.get(container_name)
            if isinstance(container, dict):
                prompt = container.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
        return None

    def _stringify_source_value(self, value: Any) -> str:
        normalized = self._normalize_value(value)
        if isinstance(normalized, str):
            return normalized
        if isinstance(normalized, (int, float, bool)):
            return str(normalized)
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)

    def _normalize_value(self, value: Any) -> Any:
        return normalize_value(value)
