"""Log each LLM exchange of a stage script to a JSON-lines file.

Stage scripts cannot write to Postgres. The stage module sets
AI4IA_LLM_EXCHANGE_LOG, and stores the file's lines in llm_call_payloads after
the script exits. Logging never raises.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

ENV_VAR = "AI4IA_LLM_EXCHANGE_LOG"
_lock = threading.Lock()


def response_content(data: dict[str, Any]) -> str:
    """The generated text from a vLLM chat or Ollama generate response."""
    if data.get("choices"):
        return (data["choices"][0].get("message") or {}).get("content") or ""
    return data.get("response") or ""


def record(model: str | None, request_body: dict[str, Any], response_text: str | None,
           latency_ms: int, error: str | None = None) -> None:
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    try:
        line = json.dumps({
            "id": f"lx_{uuid.uuid4().hex}",
            "called_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "request_text": json.dumps(request_body),  # as sent on the wire
            "response_text": response_text,
            "latency_ms": latency_ms,
            "error": error,
        })
        with _lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)  # holds prompt text
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        pass
