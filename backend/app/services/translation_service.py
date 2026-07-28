"""Batch text translation with pluggable providers and a Redis cache.

Primary provider is the vLLM strong model (gpt-oss-120b); the interface is
provider-agnostic so an MT API (DeepL, LibreTranslate, ...) can be swapped
in via settings.translation_provider without touching callers.

Contract: translate_batch() never raises — on any provider/cache failure
the original strings come back unchanged (fail-open), so a broken
translator can never break an API response.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from app.core.config import VLLM_BASE_URL, VLLM_CHAT_ENDPOINT, VLLM_MODEL
from app.core import settings as _settings_module
from app.services.llm_service import _post_with_retries, _strip_json_wrapping

logger = logging.getLogger("ai4ia.services.translation")

_LANGUAGE_NAMES = {
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "hi": "Hindi", "ja": "Japanese",
    "zh": "Chinese (Simplified)",
}

_REGION_HINTS = {
    "pt-BR": "Brazilian Portuguese", "pt-PT": "European Portuguese",
    "fr-CA": "Canadian French", "zh-TW": "Traditional Chinese (Taiwan)",
    "es-MX": "Mexican Spanish", "es-AR": "Argentine Spanish",
}

# Keep prompts (and single vLLM calls) bounded: chunk very large batches.
_MAX_CHARS_PER_CALL = 8_000
_MAX_ITEMS_PER_CALL = 60


def _locale_display_name(locale: str) -> str:
    if locale in _REGION_HINTS:
        return _REGION_HINTS[locale]
    base = locale.split("-")[0]
    name = _LANGUAGE_NAMES.get(base, base)
    if "-" in locale:
        return f"{name} as used in {locale.split('-')[1]}"
    return name


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

_TRANSLATE_PROMPT = """You are a professional translator. Translate every string in the JSON array below from English to {language}.

Rules:
- Return ONLY a JSON array of the translated strings, same length ({count}) and same order as the input.
- Do not translate proper nouns, people's names, organization names, or acronyms (e.g. AI4IA, S3, GPU).
- Preserve any markup, placeholders, numbers, and punctuation style.
- No explanations, no markdown fences.

Input:
{payload}"""


def _vllm_translate(texts: list[str], locale: str) -> list[str]:
    """Translate via the vLLM chat endpoint. Raises on any failure."""
    prompt = _TRANSLATE_PROMPT.format(
        language=_locale_display_name(locale),
        count=len(texts),
        payload=json.dumps(texts, ensure_ascii=False),
    )
    data, _ = _post_with_retries(
        f"{VLLM_BASE_URL}{VLLM_CHAT_ENDPOINT}",
        {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_completion_tokens": 8192,
            "stream": False,
            # gpt-oss is a reasoning model; low effort avoids paying for
            # chain-of-thought tokens on a mechanical translation task.
            "reasoning_effort": "low",
        },
    )
    raw = (data["choices"][0]["message"]["content"] or "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Reuse the JSON-repair helper, adapted for arrays.
        cleaned = _strip_json_wrapping(raw)
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)

    if not isinstance(parsed, list) or len(parsed) != len(texts):
        raise ValueError(
            f"translation output shape mismatch: expected {len(texts)} strings, "
            f"got {type(parsed).__name__} of {len(parsed) if isinstance(parsed, list) else 'n/a'}"
        )
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError("translation output contains non-string items")
    return parsed


def _resolve_provider(name: str):
    # Late-bound module-global lookup so tests can patch _vllm_translate.
    if name == "vllm":
        return _vllm_translate
    return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_redis_client = None
_redis_failed_at: float = 0.0
_REDIS_RETRY_SECONDS = 30.0


def _get_redis():
    """Lazily connect to Redis; back off for a while after a failure."""
    global _redis_client, _redis_failed_at
    if _redis_client is not None:
        return _redis_client
    if time.monotonic() - _redis_failed_at < _REDIS_RETRY_SECONDS:
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            _settings_module.settings.redis_url, socket_connect_timeout=0.5, socket_timeout=1.0
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:  # noqa: BLE001
        _redis_failed_at = time.monotonic()
        logger.debug("translation cache: Redis unavailable, running uncached")
        return None


def _cache_key(text: str, locale: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return f"tr:{locale}:{digest}"


def _cache_get_many(texts: list[str], locale: str) -> list[str | None]:
    client = _get_redis()
    if client is None:
        return [None] * len(texts)
    try:
        values = client.mget([_cache_key(t, locale) for t in texts])
        return [v.decode("utf-8") if v is not None else None for v in values]
    except Exception:  # noqa: BLE001
        return [None] * len(texts)


def _cache_set_many(pairs: list[tuple[str, str]], locale: str) -> None:
    client = _get_redis()
    if client is None or not pairs:
        return
    try:
        pipe = client.pipeline(transaction=False)
        for source, translated in pairs:
            pipe.setex(
                _cache_key(source, locale),
                _settings_module.settings.translation_cache_ttl_seconds,
                translated.encode("utf-8"),
            )
        pipe.execute()
    except Exception:  # noqa: BLE001
        logger.debug("translation cache: write failed (non-fatal)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_batch(texts: list[str], locale: str) -> tuple[list[str], bool]:
    """Translate texts to `locale`. Returns (results, fully_translated).

    Duplicate and cached strings are only translated once. On provider
    failure the affected strings are returned unchanged and the second
    element of the tuple is False.
    """
    if not texts:
        return [], True

    results: list[str | None] = list(_cache_get_many(texts, locale))

    # Unique uncached strings, preserving first-seen order.
    pending: dict[str, str | None] = {}
    for text, cached in zip(texts, results):
        if cached is None and text.strip():
            pending.setdefault(text, None)

    provider = _resolve_provider(_settings_module.settings.translation_provider)
    fully_translated = True

    if pending and provider is None:
        logger.warning(
            "translation provider %r not configured; passing text through",
            _settings_module.settings.translation_provider,
        )
        fully_translated = False
    elif pending:
        # Chunk to keep each provider call bounded.
        unique = list(pending.keys())
        chunks: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for text in unique:
            if current and (
                current_chars + len(text) > _MAX_CHARS_PER_CALL
                or len(current) >= _MAX_ITEMS_PER_CALL
            ):
                chunks.append(current)
                current, current_chars = [], 0
            current.append(text)
            current_chars += len(text)
        if current:
            chunks.append(current)

        for chunk in chunks:
            try:
                translated = provider(chunk, locale)
                for source, target in zip(chunk, translated):
                    pending[source] = target
                _cache_set_many(list(zip(chunk, translated)), locale)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "translation failed for %d strings (locale=%s): %s",
                    len(chunk), locale, exc,
                )
                fully_translated = False

    final: list[str] = []
    for text, cached in zip(texts, results):
        if cached is not None:
            final.append(cached)
        else:
            translated = pending.get(text)
            final.append(translated if translated is not None else text)
    return final, fully_translated


__all__ = ["translate_batch"]
