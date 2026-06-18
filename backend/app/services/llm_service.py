"""Unified Ollama client (embeddings + generation).

Lifts the working /api/embed -> /api/embeddings fallback out of
scripts/07_generate_storyline.py into one reusable place.
"""
import json
import time
from typing import Any

import requests

from app.core.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    STRONG_LLM_MODEL,
    FAST_LLM_MODEL,
    SPEAKER_EXTRACTION_MODEL,
)

REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 4


def _post_with_retries(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"Ollama request failed after {MAX_RETRIES} retries: {last_error}")


def embed_query(query: str) -> list[float]:
    """Embed text for retrieval. Tries /api/embed then /api/embeddings.

    nomic-embed-text expects a 'search_query:' prefix for query-side embeddings.
    """
    normalized = query.strip()
    if not normalized.startswith("search_query:"):
        normalized = "search_query: " + normalized

    errors: list[str] = []

    # Newer endpoint
    try:
        data = _post_with_retries(
            f"{OLLAMA_BASE_URL}/api/embed",
            {"model": EMBEDDING_MODEL, "input": normalized},
            timeout=120,
        )
        if "embedding" in data:
            return data["embedding"]
        if data.get("embeddings"):
            return data["embeddings"][0]
        errors.append(f"/api/embed unexpected keys: {list(data.keys())}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"/api/embed: {exc}")

    # Older endpoint
    try:
        data = _post_with_retries(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            {"model": EMBEDDING_MODEL, "prompt": normalized},
            timeout=120,
        )
        if "embedding" in data:
            return data["embedding"]
        if data.get("embeddings"):
            return data["embeddings"][0]
        errors.append(f"/api/embeddings unexpected keys: {list(data.keys())}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"/api/embeddings: {exc}")

    raise RuntimeError("Embedding failed. " + " | ".join(errors))


def generate_text(prompt: str, model: str | None = None, temperature: float = 0.3) -> str:
    data = _post_with_retries(
        f"{OLLAMA_BASE_URL}/api/generate",
        {
            "model": model or STRONG_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    return data.get("response", "")


def generate_json(prompt: str, model: str | None = None, temperature: float = 0.2) -> Any:
    """Generate and parse JSON. Uses Ollama format=json, with a salvage fallback."""
    data = _post_with_retries(
        f"{OLLAMA_BASE_URL}/api/generate",
        {
            "model": model or FAST_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
    )
    raw = (data.get("response") or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


__all__ = [
    "embed_query",
    "generate_text",
    "generate_json",
    "EMBEDDING_MODEL",
    "STRONG_LLM_MODEL",
    "FAST_LLM_MODEL",
    "SPEAKER_EXTRACTION_MODEL",
]
