"""Request and LLM-exchange records (prompt_resolutions, llm_call_payloads).

Every function here is best-effort: a recording failure is logged and never
reaches the job. See docs/intent-layer-design.md §10.
"""
from __future__ import annotations

import json
import logging
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from app.core.code_version import code_version
from app.core.settings import settings
from app.pipeline.llm_exchange_log import ENV_VAR as EXCHANGE_LOG_ENV  # noqa: F401  (re-exported)

logger = logging.getLogger("ai4ia.services.processing_records")


def expires_at() -> str:
    """End of the retention day, per privacy policy §6."""
    return datetime.combine(settings.processing_record_retain_until, time.max, timezone.utc).isoformat()


def _resolutions():
    from app.db.repositories.prompt_resolution_repository import PromptResolutionRepository
    return PromptResolutionRepository()


def _payloads():
    from app.db.repositories.llm_payload_repository import LLMPayloadRepository
    return LLMPayloadRepository()


def start_request(job_id: str, user_id: str | None, prompt: str, surface: str = "video") -> None:
    """Open the record for a new job."""
    try:
        from app.services.speaker_registry import get_registry

        _resolutions().create({
            "id": f"pr_{uuid.uuid4().hex}",
            "job_id": job_id,
            "surface": surface,
            "user_id": user_id or None,
            "raw_prompt": prompt or "",
            "layer_outcome": "in_progress",
            "code_version": code_version(),
            "corpus_version": settings.storyline_corpus_version,
            "registry_version": get_registry().version or None,
            "expires_at": expires_at(),
        })
    except Exception:
        logger.warning("prompt_resolutions: could not open record for %s", job_id, exc_info=True)


def record_layer(
    job_id: str,
    *,
    outcome: str | None = None,
    reason_code: str | None = None,
    user_message: str | None = None,
    flags: Mapping[str, Any] | None = None,
    timings_ms: Mapping[str, int] | None = None,
) -> None:
    """Record one layer's result for a job."""
    fields = {k: v for k, v in (("layer_outcome", outcome), ("reason_code", reason_code),
                                ("user_message", user_message)) if v is not None}
    merge = {k: v for k, v in (("flags", flags), ("timings_ms", timings_ms)) if v}
    try:
        _resolutions().update_by_job(job_id, fields=fields, merge=merge)
    except Exception:
        logger.warning("prompt_resolutions: could not record layer for %s", job_id, exc_info=True)


def finish(job_id: str, outcome: str) -> None:
    """Set the final outcome unless a layer already decided one (e.g. l1_rejected)."""
    try:
        _resolutions().update_by_job(job_id, fields={"layer_outcome": outcome}, only_if_outcome="in_progress")
    except Exception:
        logger.warning("prompt_resolutions: could not finish %s", job_id, exc_info=True)


def store_llm_exchange(row: Mapping[str, Any]) -> None:
    """Store one LLM request/response, when payload recording is on."""
    if not settings.record_llm_payloads:
        return
    try:
        _payloads().create({**row, "expires_at": expires_at()})
    except Exception:
        logger.warning("llm_call_payloads: could not store %s", row.get("id"), exc_info=True)


def exchange_log_path(job_id: str, stage: str) -> Path:
    return Path(tempfile.gettempdir()) / f"ai4ia_llm_exchanges_{stage}_{job_id}.jsonl"


@contextmanager
def exchange_log(job_id: str, stage: str, env: dict[str, str]) -> Iterator[None]:
    """Give a stage script an exchange log via `env`; store its lines afterwards,
    whether or not the script succeeded."""
    if not settings.record_llm_payloads:
        yield
        return
    path = exchange_log_path(job_id, stage)
    path.unlink(missing_ok=True)
    env[EXCHANGE_LOG_ENV] = str(path)
    try:
        yield
    finally:
        try:
            from app.services.job_store import load_job
            user_id = load_job(job_id).get("user_id")
        except Exception:
            user_id = None
        import_exchange_log(path, job_id=job_id, user_id=user_id, stage=stage)


def import_exchange_log(path: Path, *, job_id: str, user_id: str | None, stage: str) -> int:
    """Store the exchanges a stage script logged, then delete the file."""
    stored = 0
    try:
        if not path.exists():
            return 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                exchange = json.loads(line)
            except json.JSONDecodeError:
                continue
            store_llm_exchange({**exchange, "job_id": job_id, "user_id": user_id or None, "stage": stage})
            stored += 1
    except Exception:
        logger.warning("llm_call_payloads: could not import %s", path, exc_info=True)
    finally:
        path.unlink(missing_ok=True)
    return stored


def delete_for_job(job_id: str) -> None:
    _resolutions().delete_by_job(job_id)
    _payloads().delete_by_job(job_id)


def delete_for_user(user_id: str) -> None:
    _resolutions().delete_by_user(user_id)
    _payloads().delete_by_user(user_id)


def purge_expired() -> dict[str, int]:
    return {"prompt_resolutions": _resolutions().purge_expired(),
            "llm_call_payloads": _payloads().purge_expired()}
