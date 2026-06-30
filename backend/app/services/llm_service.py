"""Unified Ollama client (embeddings + generation).

Lifts the working /api/embed -> /api/embeddings fallback out of
scripts/07_generate_storyline.py into one reusable place.

Provenance tracing
------------------
Pass a `_trace` dict to embed_query / generate_text / generate_json to emit a
row to llm_call_traces. Required keys: agent, call_type. Optional: job_id,
insight_id, user_id, stage, decision_label.
"""
import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

from app.core.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    STRONG_LLM_MODEL,
    FAST_LLM_MODEL,
    SPEAKER_EXTRACTION_MODEL,
)

logger = logging.getLogger("ai4ia.services.llm")

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _post_with_retries(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[dict, int]:
    """Return (response_dict, retry_count). Raises on exhausted retries."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json(), attempt - 1
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"Ollama request failed after {MAX_RETRIES} retries: {last_error}")


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _emit_trace(trace: dict) -> None:
    """Fire-and-forget: write one llm_call_traces row. Never raises."""
    def _insert() -> None:
        try:
            from app.db.repositories.llm_trace_repository import LLMTraceRepository
            LLMTraceRepository().create(trace)
        except Exception:  # noqa: BLE001
            logger.debug("llm trace emit failed (non-fatal): %s", trace.get("id"))

    threading.Thread(target=_insert, daemon=True).start()


def _build_trace(
    _trace: dict | None,
    call_type: str,
    model: str,
    prompt: str,
    response_text: str,
    latency_ms: int,
    retry_count: int,
    temperature: float | None,
    error: str | None,
) -> dict | None:
    if _trace is None:
        return None
    return {
        "id": f"lt_{uuid.uuid4().hex}",
        "job_id": _trace.get("job_id"),
        "insight_id": _trace.get("insight_id"),
        "user_id": _trace.get("user_id"),
        "stage": _trace.get("stage"),
        "agent": _trace.get("agent", "unknown"),
        "model": model,
        "call_type": call_type,
        "decision_label": _trace.get("decision_label"),
        "prompt_hash": _sha256(prompt),
        "response_hash": _sha256(response_text) if response_text else None,
        "latency_ms": latency_ms,
        "temperature": temperature,
        "retry_count": retry_count,
        "error": error,
        "called_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_query(query: str, _trace: dict | None = None) -> list[float]:
    """Embed text for retrieval. Tries /api/embed then /api/embeddings.

    nomic-embed-text expects a 'search_query:' prefix for query-side embeddings.
    """
    normalized = query.strip()
    if not normalized.startswith("search_query:"):
        normalized = "search_query: " + normalized

    errors: list[str] = []
    t0 = time.time()
    retry_count = 0
    result: list[float] = []
    error_str: str | None = None

    try:
        # Newer endpoint
        try:
            data, retry_count = _post_with_retries(
                f"{OLLAMA_BASE_URL}/api/embed",
                {"model": EMBEDDING_MODEL, "input": normalized},
                timeout=120,
            )
            if "embedding" in data:
                result = data["embedding"]
            elif data.get("embeddings"):
                result = data["embeddings"][0]
            else:
                errors.append(f"/api/embed unexpected keys: {list(data.keys())}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"/api/embed: {exc}")

        # Older endpoint fallback
        if not result:
            try:
                data, retry_count = _post_with_retries(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    {"model": EMBEDDING_MODEL, "prompt": normalized},
                    timeout=120,
                )
                if "embedding" in data:
                    result = data["embedding"]
                elif data.get("embeddings"):
                    result = data["embeddings"][0]
                else:
                    errors.append(f"/api/embeddings unexpected keys: {list(data.keys())}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"/api/embeddings: {exc}")

        if not result:
            raise RuntimeError("Embedding failed. " + " | ".join(errors))

        return result

    except Exception as exc:
        error_str = str(exc)
        raise
    finally:
        if _trace is not None:
            latency_ms = int((time.time() - t0) * 1000)
            trace = _build_trace(
                _trace, "embed", EMBEDDING_MODEL, normalized,
                str(len(result)) if result else "",
                latency_ms, retry_count, None, error_str,
            )
            if trace:
                _emit_trace(trace)


def generate_text(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    _trace: dict | None = None,
) -> str:
    resolved_model = model or STRONG_LLM_MODEL
    t0 = time.time()
    retry_count = 0
    response_text = ""
    error_str: str | None = None

    try:
        data, retry_count = _post_with_retries(
            f"{OLLAMA_BASE_URL}/api/generate",
            {
                "model": resolved_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response_text = data.get("response", "")
        return response_text
    except Exception as exc:
        error_str = str(exc)
        raise
    finally:
        if _trace is not None:
            latency_ms = int((time.time() - t0) * 1000)
            trace = _build_trace(
                _trace, "generate_text", resolved_model, prompt,
                response_text, latency_ms, retry_count, temperature, error_str,
            )
            if trace:
                _emit_trace(trace)


def generate_json(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    _trace: dict | None = None,
) -> Any:
    """Generate and parse JSON. Uses Ollama format=json, with a salvage fallback."""
    resolved_model = model or FAST_LLM_MODEL
    t0 = time.time()
    retry_count = 0
    response_text = ""
    error_str: str | None = None

    try:
        data, retry_count = _post_with_retries(
            f"{OLLAMA_BASE_URL}/api/generate",
            {
                "model": resolved_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            },
        )
        raw = (data.get("response") or "").strip()
        response_text = raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
            raise
    except Exception as exc:
        error_str = str(exc)
        raise
    finally:
        if _trace is not None:
            latency_ms = int((time.time() - t0) * 1000)
            trace = _build_trace(
                _trace, "generate_json", resolved_model, prompt,
                response_text, latency_ms, retry_count, temperature, error_str,
            )
            if trace:
                _emit_trace(trace)


__all__ = [
    "embed_query",
    "generate_text",
    "generate_json",
    "EMBEDDING_MODEL",
    "STRONG_LLM_MODEL",
    "FAST_LLM_MODEL",
    "SPEAKER_EXTRACTION_MODEL",
]
