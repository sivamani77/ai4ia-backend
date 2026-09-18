"""Unit tests for the shared TokenBucket rate limiter (app.core.rate_limit).

Originally extracted out of app.services.auth_service / app.api.video_jobs,
where it was duplicated verbatim. TokenBucket is now Redis-backed (a fixed
window INCR+EXPIRE counter shared across replicas) with the original
in-process dict logic kept as a fallback for when Redis is unreachable —
`TestTokenBucketLocalFallback` patches `_get_redis` to return None so it
exercises exactly that fallback path, hermetically, with no real Redis.
`TestTokenBucketRedisBacked` exercises the Redis path against a fake client.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.core.rate_limit import TokenBucket


class TestTokenBucketLocalFallback:
    """Redis unreachable — same behavior as the original in-process bucket."""

    def setup_method(self):
        self._patcher = patch("app.core.rate_limit._get_redis", return_value=None)
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

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


def _fake_redis_client(counts):
    """A fake Redis client whose pipeline().execute() returns the next count
    from `counts` for the INCR call, paired with a no-op EXPIRE result."""
    client = MagicMock()
    pipe = MagicMock()
    pipe.execute.side_effect = [(c, True) for c in counts]
    client.pipeline.return_value = pipe
    return client


class TestTokenBucketRedisBacked:
    def test_allows_up_to_rate_then_denies(self):
        client = _fake_redis_client([1, 2, 3, 4])
        with patch("app.core.rate_limit._get_redis", return_value=client):
            bucket = TokenBucket(rate=3, per=60.0, name="t")
            assert bucket.allow("k") is True
            assert bucket.allow("k") is True
            assert bucket.allow("k") is True
            assert bucket.allow("k") is False

    def test_uses_namespaced_key(self):
        client = _fake_redis_client([1])
        with patch("app.core.rate_limit._get_redis", return_value=client):
            bucket = TokenBucket(rate=1, per=60.0, name="login")
            bucket.allow("user@example.com")

        client.pipeline.return_value.incr.assert_called_once_with(
            "ratelimit:login:user@example.com"
        )

    def test_falls_back_to_local_bucket_when_redis_call_raises(self):
        client = MagicMock()
        client.pipeline.return_value.execute.side_effect = ConnectionError("boom")
        with patch("app.core.rate_limit._get_redis", return_value=client):
            bucket = TokenBucket(rate=1, per=60.0, name="t")
            assert bucket.allow("k") is True
            assert bucket.allow("k") is False
