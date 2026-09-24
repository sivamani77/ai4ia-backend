"""LLM client for Step 07b — backend translation (Ollama/vLLM), retries, logging."""
import asyncio
import time
from datetime import datetime
from typing import Any

import httpx

from app.core.settings import settings
from app.pipeline import llm_exchange_log

OLLAMA_BASE_URL = settings.ollama_base_url
GENERATE_ENDPOINT = "/api/generate"
VLLM_BASE_URL = settings.vllm_base_url
VLLM_MODEL = settings.vllm_model
CHAT_ENDPOINT = settings.vllm_chat_endpoint
STRONG_LLM_MODEL = settings.storyline_strong_model
FAST_LLM_MODEL = settings.fast_llm_model

# Backend for STRONG_LLM_MODEL calls in this module: "ollama" (default,
# /api/generate) or "vllm" (dedicated gpt-oss-120b deployment, OpenAI-
# compatible /v1/chat/completions — see storyline_vllm_* settings). Every
# call site already builds an OpenAI-chat-shaped payload (model/messages/
# temperature/max_completion_tokens), so switching backend needs no
# call-site changes — only post_with_retries/post_with_retries_async below.
STORYLINE_LLM_BACKEND = settings.storyline_llm_backend
STORYLINE_VLLM_BASE_URL = settings.storyline_vllm_base_url
STORYLINE_VLLM_MODEL = settings.storyline_vllm_model

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


# --- model-host health, recorded not probed -------------------------------
# Mirrors the shim in 07_generate_storyline.py. Both scripts are short-lived
# subprocesses that issue most of the 120b traffic, so they are the main
# contributors to the shared health record /api/health reads
# (app/core/dependency_state.py). Best-effort throughout: telemetry must
# never be able to fail a validation run.

def _dep_host(url: str) -> str:
    try:
        from app.core import dependency_state
        return dependency_state.host_from_url(url)
    except Exception:
        return ""


def _record_dep(host: str, ok: bool, latency_s: float, error: str | None = None) -> None:
    if not host:
        return
    try:
        from app.core import dependency_state
        dependency_state.record(host, ok, latency_s=latency_s, error=error)
    except Exception:
        pass


def _to_ollama_generate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI-chat-shaped payload (model/messages/temperature/
    max_completion_tokens) into Ollama's /api/generate shape (model/prompt/
    options.temperature/options.num_ctx). Role-separated: any role="system"
    messages are joined into Ollama's native top-level "system" field; every
    other message (role="user", etc.) is joined into "prompt" — preserving
    real system/user separation instead of flattening both into one string.

    Deliberately does NOT set the top-level "format": "json" flag: verified live
    against this Ollama host that gpt-oss:120b returns an EMPTY response when
    that flag is set (confirmed via direct /api/generate call — "response": ""
    with done_reason="stop" even on a trivial prompt with ample context budget),
    while the identical prompt without the flag returns clean JSON immediately.
    This looks like an Ollama/gpt-oss "harmony" channel-format incompatibility,
    not a prompt issue. Every caller here already parses the free-form response
    with extract_json_object/extract_json_array, so plain prompting works fine —
    prefer that over a "format" flag that silently breaks this model."""
    messages = payload.get("messages") or []
    system_text = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system" and m.get("content")
    )
    prompt = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") != "system" and m.get("content")
    )
    result: dict[str, Any] = {
        "model": payload.get("model"),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": payload.get("temperature", 0.3),
            "num_ctx": payload.get("max_completion_tokens", 4096),
        },
    }
    if system_text:
        result["system"] = system_text
    return result


def _to_vllm_chat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload here is already OpenAI-chat-shaped — just swap in the
    vLLM-side model name (Ollama and vLLM name the same model differently,
    e.g. "gpt-oss:120b" vs "openai/gpt-oss-120b") and ensure a token cap is
    set (vLLM has no equivalent of Ollama's num_ctx-implied unlimited
    generation).

    reasoning_effort="low": gpt-oss is a "harmony"/reasoning model — it
    writes chain-of-thought to a separate reasoning_content field before the
    real answer goes into content. Verified live against the vLLM endpoint:
    at low token budgets the reasoning alone can consume the whole budget,
    leaving content=null with finish_reason="length". "low" roughly halves
    reasoning-token spend (measured ~23 vs ~55 completion tokens on a trivial
    prompt), making truncation far less likely on real prompts.
    """
    return {
        **payload,
        "model": STORYLINE_VLLM_MODEL,
        "max_completion_tokens": payload.get("max_completion_tokens", 8192),
        "stream": False,
        "reasoning_effort": "low",
    }


def post_with_retries(payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT_SECONDS) -> str:
    """POST to the strong LLM backend (Ollama or vLLM, see STORYLINE_LLM_BACKEND),
    return response text. Raises on failure."""
    import requests

    use_vllm = STORYLINE_LLM_BACKEND == "vllm"
    if use_vllm:
        url = STORYLINE_VLLM_BASE_URL.rstrip("/") + CHAT_ENDPOINT
        req_payload = _to_vllm_chat_payload(payload)
    else:
        url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT
        req_payload = _to_ollama_generate_payload(payload)

    _host = _dep_host(url)
    _call_started = time.time()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _t0 = time.time()
            resp = requests.post(url, json=req_payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if use_vllm:
                _choice = data["choices"][0]
                _content = _choice["message"].get("content") or ""
                if not _content:
                    log(f"  [llm] WARNING: vLLM returned empty content (finish_reason={_choice.get('finish_reason')!r}) "
                        f"— likely truncated in the reasoning channel before the answer; will be treated as a parse failure upstream")
            else:
                _content = data.get("response", "")
            log(f"  [llm-timing] model={req_payload.get('model','?')} latency={int((time.time()-_t0)*1000)}ms")
            _record_dep(_host, True, time.time() - _call_started)
            llm_exchange_log.record(req_payload.get("model"), req_payload, _content or "",
                                    int((time.time() - _call_started) * 1000))
            return _content or ""

        except Exception as exc:
            last_error = exc
            log(f"  [llm] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))
    _record_dep(_host, False, time.time() - _call_started, str(last_error))
    llm_exchange_log.record(req_payload.get("model"), req_payload, None,
                            int((time.time() - _call_started) * 1000), error=str(last_error))
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} retries: {last_error}")


async def post_with_retries_async(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> str:
    """Async twin of post_with_retries — same retry/backoff behavior, non-blocking I/O."""
    use_vllm = STORYLINE_LLM_BACKEND == "vllm"
    if use_vllm:
        url = STORYLINE_VLLM_BASE_URL.rstrip("/") + CHAT_ENDPOINT
        req_payload = _to_vllm_chat_payload(payload)
    else:
        url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT
        req_payload = _to_ollama_generate_payload(payload)

    _host = _dep_host(url)
    _call_started = time.time()
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _t0 = time.time()
            resp = await client.post(url, json=req_payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if use_vllm:
                _choice = data["choices"][0]
                _content = _choice["message"].get("content") or ""
                if not _content:
                    log(f"  [llm] WARNING: vLLM returned empty content (finish_reason={_choice.get('finish_reason')!r}) "
                        f"— likely truncated in the reasoning channel before the answer; will be treated as a parse failure upstream")
            else:
                _content = data.get("response", "")
            log(f"  [llm-timing] model={req_payload.get('model','?')} latency={int((time.time()-_t0)*1000)}ms")
            _record_dep(_host, True, time.time() - _call_started)
            llm_exchange_log.record(req_payload.get("model"), req_payload, _content or "",
                                    int((time.time() - _call_started) * 1000))
            return _content or ""

        except Exception as exc:
            last_error = exc
            log(f"  [llm] Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))
    _record_dep(_host, False, time.time() - _call_started, str(last_error))
    llm_exchange_log.record(req_payload.get("model"), req_payload, None,
                            int((time.time() - _call_started) * 1000), error=str(last_error))
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} retries: {last_error}")
