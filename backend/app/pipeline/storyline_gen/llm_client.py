import time
from datetime import datetime
from typing import Any

import requests

from app.core.settings import settings


OLLAMA_BASE_URL = settings.ollama_base_url
VLLM_MODEL = settings.vllm_model
VLLM_BASE_URL = settings.vllm_base_url

CHAT_ENDPOINT = settings.vllm_chat_endpoint

EMBED_MODEL = settings.embedding_model
FAST_LLM_MODEL = settings.fast_llm_model
STRONG_LLM_MODEL = settings.storyline_strong_model

# Backend for STRONG_LLM_MODEL calls: "ollama" (default, /api/generate) or
# "vllm" (dedicated gpt-oss-120b deployment, OpenAI-compatible
# /v1/chat/completions) — see call_strong_llm() below.
STORYLINE_LLM_BACKEND = settings.storyline_llm_backend
STORYLINE_VLLM_BASE_URL = settings.storyline_vllm_base_url
STORYLINE_VLLM_MODEL = settings.storyline_vllm_model

EMBED_ENDPOINT = "/api/embed"
GENERATE_ENDPOINT = "/api/generate"

REQUEST_TIMEOUT_SECONDS = 300
EMBED_TIMEOUT_SECONDS = 300       # nomic can be slow to cold-load on a busy shared server
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 5           # base for exponential backoff (5, 10, 20, ...)


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}", flush=True)


# --- model-host health, recorded not probed -------------------------------
# This script is a short-lived subprocess but issues most of the 120b
# traffic, so it is the main contributor to the shared health record that
# /api/health reads (app/core/dependency_state.py). Everything below is
# best-effort: telemetry must never be able to fail a storyline run.

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


def post_with_retries(url: str, payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    _host = _dep_host(url)
    last_error = None
    _call_started = time.time()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _t0 = time.time()
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            _result = response.json()
            log(f"  [llm-timing] model={payload.get('model','?')} latency={int((time.time()-_t0)*1000)}ms")
            _record_dep(_host, True, time.time() - _call_started)
            return _result
        except Exception as exc:
            last_error = exc
            log(f"Request failed attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                # Exponential backoff: gives a cold/overloaded Ollama time to recover.
                time.sleep(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))

    _record_dep(_host, False, time.time() - _call_started, str(last_error))
    raise RuntimeError(f"Request failed after retries: {last_error}")


def call_strong_llm(
    prompt: str,
    timeout: int,
    temperature: float = 0.4,
    top_p: float | None = None,
    max_tokens: int = 8192,
    system: str | None = None,
) -> str:
    """Call STRONG_LLM_MODEL via whichever backend is configured
    (STORYLINE_LLM_BACKEND), returning the raw response text.

    Single point of translation between the two request/response shapes so
    every call site below just deals with a prompt string, regardless of
    backend — matches the pattern already used in
    scripts/07b_validate_storyline_with_llm.py.

    `system`, when given, is sent as a real system-role message on the vLLM
    path and as Ollama's native top-level "system" field on the Ollama
    path — kept separate from `prompt` at the transport layer either way.

    max_tokens caps the vLLM response only (Ollama's /api/generate has no
    equivalent cap here — it already runs until natural completion). Callers
    generating a large JSON timeline need a much higher value than the
    default: verified live that the full storyline JSON response needs more
    than 8192 tokens and gets cut off mid-object at that cap
    (finish_reason="length").
    """
    if STORYLINE_LLM_BACKEND == "vllm":
        url = STORYLINE_VLLM_BASE_URL.rstrip("/") + CHAT_ENDPOINT
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": STORYLINE_VLLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
            # gpt-oss is a "harmony"/reasoning model: it writes its chain-of-
            # thought to a separate reasoning_content field before the real
            # answer goes into content. Verified live against this endpoint:
            # at low max_tokens the reasoning alone can eat the whole budget,
            # leaving content=null with finish_reason="length". reasoning_effort
            # "low" cuts reasoning-token spend roughly in half (measured ~23 vs
            # ~55 completion tokens on a trivial prompt) and makes truncation
            # far less likely on our much larger real prompts.
            "reasoning_effort": "low",
        }
        data = post_with_retries(url, payload, timeout=timeout)
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError(
                f"vLLM returned no content (finish_reason={data['choices'][0].get('finish_reason')!r}) "
                f"— likely truncated before the answer channel; consider raising max_completion_tokens"
            )
        return content

    url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT
    options: dict[str, Any] = {"temperature": temperature, "num_ctx": 32000}
    if top_p is not None:
        options["top_p"] = top_p
    payload: dict[str, Any] = {
        "model": STRONG_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system
    data = post_with_retries(url, payload, timeout=timeout)
    return (data.get("response") or "").strip()


def warm_up_ollama() -> None:
    """Force-load STRONG_LLM_MODEL into Ollama before the real storyline call.

    A GET /api/tags only checks the server is reachable — it does NOT load a
    model into memory. gpt-oss:120b is large enough that its cold load time
    (reading weights off disk) can be minutes, and without this warm-up that
    load happens inside generate_llm_storyline()'s timed call. Sending a
    trivial /api/generate here pays that cost up front instead. Best-effort:
    failures here are non-fatal, the real call will just eat the cold-load cost.

    No-op on the vLLM backend: vLLM serves the model as an already-resident
    process, there is no per-request cold-load to hide.
    """
    if STORYLINE_LLM_BACKEND == "vllm":
        return
    try:
        _t0 = time.time()
        requests.post(
            OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT,
            json={
                "model": STRONG_LLM_MODEL,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=600,
        )
        log(f"Ollama warm-up: {STRONG_LLM_MODEL} loaded in {time.time() - _t0:.1f}s")
    except Exception as exc:
        log(f"Ollama warm-up failed (continuing, real call will pay cold-load cost): {exc}")
