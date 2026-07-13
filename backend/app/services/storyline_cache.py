"""Storyline cache — skips 07_storyline + 07b_validate_storyline on a hit.

07b's output (the validated storyline) is what 08_assets/09_export actually
consume, so this caches the COMBINED result of both stages, keyed by the
request parameters that affect generation content. A hit means the pipeline
jumps straight to 08_assets; a miss runs both stages normally and (if the
result passed its quality gates) writes the result to cache afterward.

Two layers, both toggled independently via settings:

L1 — exact match on (normalized prompt, target_duration, style,
validation_mode, visual_mode, max_validation_iterations, corpus_version).
Zero false-positive risk. Controlled by `storyline_cache_enabled`.

L2 — semantic/paraphrase match: embed the incoming prompt (same
nomic-embed-text model the pipeline already uses for retrieval), compare
against embeddings of previously-cached prompts (scoped to the same
non-prompt params) via cosine similarity, and require an LLM confirmation
("do these two prompts ask for the same video?") on any candidate above the
similarity threshold before treating it as a hit — a wrong topic served fast
is worse than a right topic served slow. Off by default
(`storyline_cache_l2_enabled`); only turn on with real traffic data to tune
the threshold against.

Storage: Redis (same instance already used for Celery broker/results),
lazy-connected with backoff on failure — mirrors app/services/
translation_service.py's cache pattern. Fail-open: any Redis error is a
cache miss, never a hard failure of the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any

from app.core import settings as _settings_module
from app.core import metrics as _metrics

logger = logging.getLogger("ai4ia.services.storyline_cache")

_L1_PREFIX = "storyline_cache:l1:"
_L2_HASH_PREFIX = "storyline_cache:l2:"  # + params_key -> Redis HASH

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
            _settings_module.settings.redis_url, socket_connect_timeout=0.5, socket_timeout=2.0
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:  # noqa: BLE001
        _redis_failed_at = time.monotonic()
        logger.debug("storyline cache: Redis unavailable, running uncached")
        return None


# ---------------------------------------------------------------------------
# Key composition
# ---------------------------------------------------------------------------

def _normalize_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().lower().split())


def _relevant_params(request: dict[str, Any]) -> dict[str, Any]:
    """Fields that actually affect 07/07b output. Deliberately excludes
    subtitle_mode/burn_subtitles (09-only) so unrelated request differences
    don't lower the cache hit rate."""
    return {
        "target_duration": request.get("target_duration"),
        "style": request.get("style"),
        "validation_mode": request.get("validation_mode"),
        "visual_mode": request.get("visual_mode"),
        "max_validation_iterations": request.get("max_validation_iterations"),
        "corpus_version": _settings_module.settings.storyline_cache_corpus_version,
    }


def _params_key(request: dict[str, Any]) -> str:
    """Hash of the non-prompt params — scopes L2 candidate search to requests
    that are otherwise configured the same way."""
    canonical = json.dumps(_relevant_params(request), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _l1_key(request: dict[str, Any]) -> str:
    payload = {**_relevant_params(request), "prompt": _normalize_prompt(request.get("prompt", ""))}
    canonical = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_L1_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Embeddings (L2 only) — minimal standalone client, mirrors embed_query() in
# scripts/07_generate_storyline.py without importing that script as a module.
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> list[float] | None:
    """Embed via Ollama. Tries both /api/embed (newer) and /api/embeddings
    (older) since deployments vary in which one they expose — same fallback
    embed_query() in scripts/07_generate_storyline.py uses.

    Uses the "classification: " prefix, not "search_query: " (which the
    retrieval code in 07_generate_storyline.py uses). search_query is meant
    for asymmetric query-vs-document matching and gave poor separation when
    tested query-vs-query here: a genuine paraphrase and a same-theme-but-
    different-topic prompt scored almost identically (0.707 vs 0.694).
    classification: gave much better separation in the same test (0.87-0.91
    for true paraphrases vs 0.62-0.65 for clearly unrelated prompts) — still
    not enough alone to safely tell "paraphrase" from "same theme, different
    topic" (both score ~0.87-0.91), which is exactly why the LLM
    confirmation gate in _llm_confirms_same_topic is mandatory, not optional.
    """
    import requests

    settings = _settings_module.settings
    query = f"classification: {text}"
    base = settings.ollama_base_url.rstrip("/")

    try:
        resp = requests.post(
            f"{base}/api/embed",
            json={"model": settings.embedding_model, "input": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" in data:
            return list(data["embedding"])
        if "embeddings" in data and data["embeddings"]:
            return list(data["embeddings"][0])
    except Exception as exc:  # noqa: BLE001
        logger.debug("storyline cache: /api/embed failed, trying /api/embeddings: %s", exc)

    try:
        resp = requests.post(
            f"{base}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "embedding" in data:
            return list(data["embedding"])
        if "embeddings" in data and data["embeddings"]:
            return list(data["embeddings"][0])
    except Exception as exc:  # noqa: BLE001
        logger.debug("storyline cache: /api/embeddings also failed (non-fatal): %s", exc)

    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_CONFIRMATION_PROMPT_TEMPLATE = """You check whether two documentary video requests are asking for a video on the SAME specific subject, or merely share a general theme/keywords without being the same subject.

CRITICAL RULE: sharing topic words (e.g. "AI", "misinformation") is NOT enough — the specific subject/domain/institution must match too.

Examples:
- A: "AI and misinformation in elections"  B: "AI and misinformation in journalism"
  -> DIFFERENT (same theme — AI, misinformation — but different domain: elections vs journalism)
- A: "AI and misinformation in elections"  B: "How artificial intelligence spreads misinformation during elections"
  -> SAME (same domain: elections — just reworded)
- A: "Youth empowerment through AI education"  B: "AI and healthcare access in rural communities"
  -> DIFFERENT (different domain: education vs healthcare)
- A: "Climate change and community resilience"  B: "How communities are building resilience to climate change"
  -> SAME (same domain: climate/community resilience — just reworded)

Now classify this pair:
A: "{prompt_a}"
B: "{prompt_b}"

First, in one short phrase each, name the specific domain/subject of A and of B.
Then on its own final line, write exactly "ANSWER: SAME" or "ANSWER: DIFFERENT"."""


def _llm_confirms_same_topic(prompt_a: str, prompt_b: str) -> bool:
    """Cheap confirmation gate on an L2 candidate hit, using the fast model
    (not the 120b storyline model) — guards against the "same theme,
    different topic" false-positive an embedding-only match can produce.

    Uses few-shot examples and asks the model to name each prompt's specific
    domain before answering — a bare YES/NO prompt tested worse (it said
    "AI in elections" and "AI in journalism" were the same topic just
    because both mention AI and misinformation)."""
    try:
        import requests

        settings = _settings_module.settings
        question = _CONFIRMATION_PROMPT_TEMPLATE.format(prompt_a=prompt_a, prompt_b=prompt_b)
        resp = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": settings.fast_llm_model,
                "prompt": question,
                "stream": False,
                # Enough room for the two short domain phrases plus the
                # final ANSWER line — bare YES/NO (num_predict=5) tested
                # unreliably; naming the domain first anchors the judgment.
                "options": {"temperature": 0.0, "num_predict": 80},
            },
            # Longer than the embedding calls' timeout: this shared Ollama host
            # has shown 30s+ latency spikes under load even for the fast model,
            # and this fails CLOSED (treated as "no match") on any timeout —
            # so a too-short timeout here costs cache hits, not correctness.
            timeout=60,
        )
        resp.raise_for_status()
        answer = (resp.json().get("response") or "").strip().upper()
        # Look for the final answer line specifically rather than a bare
        # prefix check — the model emits reasoning text before it now.
        if "ANSWER: SAME" in answer:
            return True
        if "ANSWER: DIFFERENT" in answer:
            return False
        # Malformed output (didn't follow the format) — fail closed.
        logger.debug("storyline cache: L2 confirmation gave unparseable output: %r", answer[:200])
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("storyline cache: L2 confirmation call failed (treating as no match): %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached_storyline(request: dict[str, Any]) -> dict[str, Any] | None:
    """Look up a cached validated storyline for this request. Returns the
    parsed JSON dict on a hit, None on a miss. Never raises."""
    settings = _settings_module.settings
    if not settings.storyline_cache_enabled:
        return None

    client = _get_redis()
    if client is None:
        return None

    prompt = request.get("prompt", "")

    # --- L1: exact match ---
    l1_key = _l1_key(request)
    try:
        raw = client.get(l1_key)
    except Exception:  # noqa: BLE001
        raw = None
    if raw is not None:
        try:
            storyline = json.loads(raw)
        except Exception:  # noqa: BLE001
            storyline = None
        if storyline is not None:
            _metrics.storyline_cache_hits_total.labels(layer="l1").inc()
            logger.info(
                "STORYLINE CACHE HIT (L1, exact match) — prompt=%r key=%s — skipping 07_storyline + 07b_validate_storyline",
                prompt[:80], l1_key,
            )
            return storyline

    # --- L2: semantic/paraphrase match (opt-in) ---
    if not settings.storyline_cache_l2_enabled:
        _metrics.storyline_cache_misses_total.inc()
        logger.info("STORYLINE CACHE MISS — prompt=%r (L2 disabled)", prompt[:80])
        return None

    query_embedding = _embed_text(prompt)
    if query_embedding is None:
        _metrics.storyline_cache_misses_total.inc()
        logger.info("STORYLINE CACHE MISS — prompt=%r (L2 embedding unavailable)", prompt[:80])
        return None

    params_key = _params_key(request)
    l2_hash_key = f"{_L2_HASH_PREFIX}{params_key}"
    try:
        entries = client.hgetall(l2_hash_key)
    except Exception:  # noqa: BLE001
        entries = {}

    best_similarity = 0.0
    best_entry: dict[str, Any] | None = None
    for _field, raw_value in entries.items():
        try:
            entry = json.loads(raw_value)
            similarity = _cosine_similarity(query_embedding, entry["embedding"])
        except Exception:  # noqa: BLE001
            continue
        if similarity > best_similarity:
            best_similarity = similarity
            best_entry = entry

    threshold = settings.storyline_cache_l2_similarity_threshold
    if best_entry is not None and best_similarity >= threshold:
        if _llm_confirms_same_topic(prompt, best_entry["prompt"]):
            try:
                raw = client.get(best_entry["l1_key"])
                storyline = json.loads(raw) if raw is not None else None
            except Exception:  # noqa: BLE001
                storyline = None
            if storyline is not None:
                _metrics.storyline_cache_hits_total.labels(layer="l2").inc()
                logger.info(
                    "STORYLINE CACHE HIT (L2, semantic match, similarity=%.3f, confirmed) — "
                    "prompt=%r matched cached prompt=%r — skipping 07_storyline + 07b_validate_storyline",
                    best_similarity, prompt[:80], best_entry["prompt"][:80],
                )
                return storyline
        else:
            logger.info(
                "storyline cache: L2 candidate found (similarity=%.3f) but LLM confirmation rejected it — "
                "prompt=%r vs cached=%r, treating as miss",
                best_similarity, prompt[:80], best_entry["prompt"][:80],
            )

    _metrics.storyline_cache_misses_total.inc()
    logger.info("STORYLINE CACHE MISS — prompt=%r (going through LLM generation)", prompt[:80])
    return None


def store_storyline(request: dict[str, Any], validated_storyline: dict[str, Any], *, quality_ok: bool) -> None:
    """Write a freshly-generated validated storyline to cache. No-op if
    caching is disabled or the result didn't pass its quality gates — no
    point serving a known-weak result fast to the next identical query."""
    settings = _settings_module.settings
    if not settings.storyline_cache_enabled:
        return
    if not quality_ok:
        logger.info("storyline cache: not caching — result did not pass quality gates")
        return

    client = _get_redis()
    if client is None:
        return

    prompt = request.get("prompt", "")
    l1_key = _l1_key(request)
    ttl = settings.storyline_cache_ttl_seconds
    try:
        client.setex(l1_key, ttl, json.dumps(validated_storyline).encode("utf-8"))
        logger.info("storyline cache: stored L1 entry key=%s ttl=%ds prompt=%r", l1_key, ttl, prompt[:80])
    except Exception as exc:  # noqa: BLE001
        logger.debug("storyline cache: L1 write failed (non-fatal): %s", exc)
        return

    if not settings.storyline_cache_l2_enabled:
        return

    embedding = _embed_text(prompt)
    if embedding is None:
        return

    params_key = _params_key(request)
    l2_hash_key = f"{_L2_HASH_PREFIX}{params_key}"
    prompt_field = hashlib.sha256(_normalize_prompt(prompt).encode("utf-8")).hexdigest()[:16]
    try:
        if client.hlen(l2_hash_key) >= settings.storyline_cache_l2_max_candidates:
            logger.debug("storyline cache: L2 candidate cap reached for params_key=%s, skipping index write", params_key)
            return
        client.hset(l2_hash_key, prompt_field, json.dumps({
            "prompt": prompt,
            "embedding": embedding,
            "l1_key": l1_key,
        }))
        client.expire(l2_hash_key, ttl)
        logger.info("storyline cache: stored L2 embedding for prompt=%r under params_key=%s", prompt[:80], params_key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("storyline cache: L2 write failed (non-fatal): %s", exc)
