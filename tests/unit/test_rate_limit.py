"""Unit tests for the shared TokenBucket rate limiter (app.core.rate_limit).

Extracted out of app.services.auth_service / app.api.video_jobs, where it
was duplicated verbatim — this is the refactor's safety net, confirming the
extracted class behaves identically to the two copies it replaced.
"""
from __future__ import annotations

from app.core.rate_limit import TokenBucket


class TestTokenBucket:
    def test_allows_up_to_rate_requests(self):
        bucket = TokenBucket(rate=3, per=60.0)
        assert bucket.allow("k") is True
        assert bucket.allow("k") is True
        assert bucket.allow("k") is True

    def test_denies_once_rate_is_exhausted(self):
        bucket = TokenBucket(rate=2, per=60.0)
        assert bucket.allow("k") is True
        assert bucket.allow("k") is True
        assert bucket.allow("k") is False

    def test_keys_are_isolated(self):
        bucket = TokenBucket(rate=1, per=60.0)
        assert bucket.allow("a") is True
        assert bucket.allow("b") is True
        assert bucket.allow("a") is False
        assert bucket.allow("b") is False

    def test_refills_over_time(self, monkeypatch):
        import app.core.rate_limit as rate_limit_module

        now = [1000.0]
        monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: now[0])

        bucket = TokenBucket(rate=1, per=10.0)
        assert bucket.allow("k") is True
        assert bucket.allow("k") is False

        now[0] += 10.0  # a full refill interval later
        assert bucket.allow("k") is True
