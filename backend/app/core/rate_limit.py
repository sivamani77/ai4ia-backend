from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("ai4ia.core.rate_limit")

_redis_client = None
_redis_failed_at: float = 0.0
_REDIS_RETRY_SECONDS = 30.0


def _get_redis():
    """Lazily connect to Redis; back off for a while after a failure.

    Mirrors the pattern in core/dependency_state.py and services/model_gate.py —
    short socket timeouts so a wedged Redis can never add meaningful latency to
    the request that triggered the check.
    """
    global _redis_client, _redis_failed_at
    if _redis_client is not None:
        return _redis_client
    if time.monotonic() - _redis_failed_at < _REDIS_RETRY_SECONDS:
        return None
    try:
        import redis

        from app.core.settings import settings

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=2.0,
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:  # noqa: BLE001
        _redis_failed_at = time.monotonic()
        logger.debug("rate_limit: Redis unavailable, falling back to in-process bucket")
        return None


class TokenBucket:
    """Allow at most `rate` requests per `per` seconds per key.

    Backed by Redis (a fixed-window INCR+EXPIRE counter, shared across every
    API replica) so a security limiter like login/OTP can't be defeated just
    by the attacker's requests landing on different pods. Falls back to a
    local, in-process bucket — same behavior as before this existed — if
    Redis is unreachable, so the limiter degrades to per-process instead of
    disappearing.
    """

    def __init__(self, rate: int, per: float, name: str = "default") -> None:
        self._rate = rate
        self._per = per
        self._name = name
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        client = _get_redis()
        if client is not None:
            try:
                redis_key = f"ratelimit:{self._name}:{key}"
                pipe = client.pipeline(transaction=True)
                pipe.incr(redis_key)
                pipe.expire(redis_key, int(self._per) or 1, nx=True)
                count, _ = pipe.execute()
                return count <= self._rate
            except Exception:  # noqa: BLE001
                global _redis_client, _redis_failed_at
                _redis_client = None
                _redis_failed_at = time.monotonic()
                logger.debug("rate_limit: Redis call failed mid-request, falling back")
        return self._allow_local(key)

    def _allow_local(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key, now)
            tokens = self._tokens.get(key, float(self._rate))
            tokens = min(float(self._rate), tokens + (now - last) * (self._rate / self._per))
            self._last[key] = now
            if tokens >= 1:
                self._tokens[key] = tokens - 1
                return True
            self._tokens[key] = tokens
            return False
