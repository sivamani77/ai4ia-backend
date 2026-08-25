from __future__ import annotations

import threading
import time


class TokenBucket:
    """Allow at most `rate` requests per `per` seconds per key."""

    def __init__(self, rate: int, per: float) -> None:
        self._rate = rate
        self._per = per
        self._lock = threading.Lock()
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def allow(self, key: str) -> bool:
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
