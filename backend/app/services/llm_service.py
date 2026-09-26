"""LLM client — embeddings (Ollama) + generation (vLLM for strong, Ollama for fast).

Provenance tracing
------------------
Pass a `_trace` dict to embed_query / generate_text / generate_json to emit a
row to llm_call_traces. Required keys: agent, call_type. Optional: job_id,
insight_id, user_id, stage, decision_label.
"""
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

from app.core import dependency_state
from app.core import metrics as _metrics
from app.core.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    STRONG_LLM_MODEL,
    FAST_LLM_MODEL,
    SPEAKER_EXTRACTION_MODEL,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_CHAT_ENDPOINT,
    STORYLINE_VLLM_BASE_URL,
    STORYLINE_VLLM_MODEL,
)

logger = logging.getLogger("ai4ia.services.llm")

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 4

# Model name -> vLLM host. Two separate vLLM deployments exist (20b at
# VLLM_BASE_URL, 120b at STORYLINE_VLLM_BASE_URL) — resolve by model name so
# callers that pass INSIGHT_MODEL/SPEAKER_EXTRACTION_MODEL (120b) reach the
# right host instead of falling through to Ollama with a model name Ollama
# doesn't have.
_VLLM_BASE_URL_BY_MODEL = {
    VLLM_MODEL: VLLM_BASE_URL,
    STORYLINE_VLLM_MODEL: STORYLINE_VLLM_BASE_URL,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _post_with_retries(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[dict, int]:
    """Return (response_dict, retry_count). Raises on exhausted retries.

    Every vLLM and Ollama call in the backend funnels through here, which is
    why the model-host health record is taken here too: one instrumentation
    point covers embeddings, the guardrail, insights, comparisons and
    speaker extraction, and it costs no extra requests to the GPU. See
    app/core/dependency_state.py for why health is recorded rather than
    probed.
    """
    host = dependency_state.host_from_url(url)
    last_error = None
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            elapsed = time.time() - t0
            dependency_state.record(host, True, latency_s=elapsed)
            _metrics.llm_calls_total.labels(host=host, outcome="success").inc()
            _metrics.llm_call_duration_seconds.labels(host=host).observe(elapsed)
            return result, attempt - 1
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    elapsed = time.time() - t0
    dependency_state.record(host, False, latency_s=elapsed, error=str(last_error))
    _metrics.llm_calls_total.labels(host=host, outcome="failure").inc()
    _metrics.llm_call_duration_seconds.labels(host=host).observe(elapsed)
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} retries: {last_error}")


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_DUP_OPEN_RE = re.compile(r"^\{\s*\{")
_DUP_CLOSE_RE = re.compile(r"\}\s*\}\s*$")


def _strip_json_wrapping(raw: str) -> str:
    """Best-effort repair of common LLM JSON malformations: markdown fences,
    leading/trailing prose, and a duplicated opening/closing brace."""
    text = _FENCE_RE.sub("", raw.strip()).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # Duplicated brace, e.g. "{\n{\"key\": ...}\n}"
    if _DUP_OPEN_RE.match(text) and _DUP_CLOSE_RE.search(text):
        text = _DUP_OPEN_RE.sub("{", text, count=1)
        text = _DUP_CLOSE_RE.sub("}", text, count=1)

    return text.strip()


def _emit_trace(trace: dict, request_body: dict | None = None, response_text: str | None = None) -> None:
    """Fire-and-forget: write one llm_call_traces row, plus its llm_call_payloads
    row when the request body is given. Never raises."""
    def _insert() -> None:
        try:
            from app.db.repositories.llm_trace_repository import LLMTraceRepository
            LLMTraceRepository().create(trace)
        except Exception:  # noqa: BLE001
            logger.debug("llm trace emit failed (non-fatal): %s", trace.get("id"))
        if request_body is None:
            return
        try:
            from app.services import processing_records
            processing_records.store_llm_exchange({
                "id": trace["id"], "trace_id": trace["id"], "job_id": trace.get("job_id"),
                "insight_id": trace.get("insight_id"), "user_id": trace.get("user_id"),
                "stage": trace.get("stage"), "model": trace.get("model"),
                "request_text": json.dumps(request_body), "response_text": response_text,
                "error": trace.get("error"), "latency_ms": trace.get("latency_ms"),
                "called_at": trace.get("called_at"),
            })
        except Exception:  # noqa: BLE001
            logger.debug("llm payload emit failed (non-fatal): %s", trace.get("id"))

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
    system: str | None = None,
) -> str:
    resolved_model = model or VLLM_MODEL
    base_url = _VLLM_BASE_URL_BY_MODEL.get(resolved_model, VLLM_BASE_URL)
    t0 = time.time()
    retry_count = 0
    response_text = ""
    error_str: str | None = None

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": 4096,
        "stream": False,
    }

    try:
        data, retry_count = _post_with_retries(f"{base_url}{VLLM_CHAT_ENDPOINT}", body)
        response_text = data["choices"][0]["message"]["content"]
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
                _emit_trace(trace, body, response_text)


def generate_json(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    _trace: dict | None = None,
    system: str | None = None,
) -> Any:
    """Generate and parse JSON.

    Routing:
    - Fast model (llama3.1:8b / default) → Ollama /api/generate with format=json
    - Strong model (any model in _VLLM_BASE_URL_BY_MODEL, or explicit
      override) → vLLM /v1/chat/completions, on whichever vLLM host that
      model actually lives on

    `system`, when given, is sent as a real system-role message on the vLLM
    path and as Ollama's native top-level "system" field on the Ollama path —
    kept separate from `prompt` at the transport layer either way.
    """
    resolved_model = model or FAST_LLM_MODEL
    use_vllm = resolved_model in _VLLM_BASE_URL_BY_MODEL
    t0 = time.time()
    retry_count = 0
    response_text = ""
    error_str: str | None = None
    body: dict[str, Any] | None = None

    try:
        if use_vllm:
            messages = [{"role": "system", "content": system}] if system else []
            messages.append({"role": "user", "content": prompt})
            body = {
                "model": resolved_model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": 4096,
                "stream": False,
            }
            data, retry_count = _post_with_retries(
                f"{_VLLM_BASE_URL_BY_MODEL[resolved_model]}{VLLM_CHAT_ENDPOINT}", body,
            )
            raw = (data["choices"][0]["message"]["content"] or "").strip()
        else:
            ollama_payload: dict[str, Any] = {
                "model": resolved_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            }
            if system:
                ollama_payload["system"] = system
            body = ollama_payload
            data, retry_count = _post_with_retries(
                f"{OLLAMA_BASE_URL}/api/generate",
                ollama_payload,
            )
            raw = (data.get("response") or "").strip()

        response_text = raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = _strip_json_wrapping(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "generate_json: failed to parse LLM output as JSON (model=%s). Raw output: %r",
                    resolved_model, raw,
                )
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
                _emit_trace(trace, body, response_text)


__all__ = [
    "embed_query",
    "generate_text",
    "generate_json",
    "EMBEDDING_MODEL",
    "STRONG_LLM_MODEL",
    "FAST_LLM_MODEL",
    "SPEAKER_EXTRACTION_MODEL",
    "VLLM_MODEL",
    "VLLM_BASE_URL",
    "VLLM_CHAT_ENDPOINT",
]
