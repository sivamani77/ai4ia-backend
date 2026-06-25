"""Unit tests for app.core.auth — API key enforcement and session ID extraction."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock, patch


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# require_api_key
# ---------------------------------------------------------------------------

class TestRequireApiKey:
    def test_no_api_key_configured_allows_any(self):
        """When settings.api_key is blank, all requests are allowed."""
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = ""
            from app.core.auth import require_api_key
            _run(require_api_key(key="anything"))

    def test_no_api_key_configured_allows_none(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = ""
            from app.core.auth import require_api_key
            _run(require_api_key(key=None))  # type: ignore[arg-type]

    def test_correct_key_passes(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = "secret-key-123"
            from app.core.auth import require_api_key
            _run(require_api_key(key="secret-key-123"))

    def test_wrong_key_raises_403(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = "secret-key-123"
            from app.core.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(key="wrong-key"))
            assert exc_info.value.status_code == 403

    def test_missing_key_raises_403(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = "secret-key-123"
            from app.core.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(key=None))  # type: ignore[arg-type]
            assert exc_info.value.status_code == 403

    def test_empty_key_raises_403_when_configured(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = "secret"
            from app.core.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(key=""))
            assert exc_info.value.status_code == 403

    def test_403_detail_message(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.api_key = "real-key"
            from app.core.auth import require_api_key
            with pytest.raises(HTTPException) as exc_info:
                _run(require_api_key(key="bad-key"))
            assert "API key" in exc_info.value.detail


# ---------------------------------------------------------------------------
# extract_session_id
# ---------------------------------------------------------------------------

class TestExtractSessionId:
    def _make_request(self, session_id: str | None) -> MagicMock:
        req = MagicMock()
        headers = MagicMock()
        if session_id is not None:
            headers.get = lambda name, default="": session_id if name == "X-Session-ID" else default
        else:
            headers.get = lambda name, default="": default
        req.headers = headers
        return req

    def test_missing_header_returns_empty_string(self):
        from app.core.auth import extract_session_id
        req = self._make_request(None)
        assert extract_session_id(req) == ""

    def test_empty_header_returns_empty_string(self):
        from app.core.auth import extract_session_id
        req = self._make_request("")
        assert extract_session_id(req) == ""

    def test_valid_uuid4_accepted(self):
        from app.core.auth import extract_session_id
        valid = "550e8400-e29b-41d4-a716-446655440000"
        req = self._make_request(valid)
        assert extract_session_id(req) == valid

    def test_valid_uuid4_uppercase_accepted(self):
        from app.core.auth import extract_session_id
        valid = "550E8400-E29B-41D4-A716-446655440000"
        req = self._make_request(valid)
        assert extract_session_id(req) == valid

    def test_invalid_uuid_raises_400(self):
        from app.core.auth import extract_session_id
        req = self._make_request("not-a-uuid")
        with pytest.raises(HTTPException) as exc_info:
            extract_session_id(req)
        assert exc_info.value.status_code == 400

    def test_uuid1_raises_400(self):
        """UUID1 (version 1) must be rejected — only UUID4 is valid."""
        import uuid
        uuid1 = str(uuid.uuid1())
        from app.core.auth import extract_session_id
        req = self._make_request(uuid1)
        with pytest.raises(HTTPException) as exc_info:
            extract_session_id(req)
        assert exc_info.value.status_code == 400

    def test_uuid4_wrong_variant_raises_400(self):
        """Version bit must be 4, variant bits must be 8/9/a/b."""
        # A UUID with version=3 (not 4)
        bad = "550e8400-e29b-31d4-a716-446655440000"
        from app.core.auth import extract_session_id
        req = self._make_request(bad)
        with pytest.raises(HTTPException) as exc_info:
            extract_session_id(req)
        assert exc_info.value.status_code == 400

    def test_400_detail_mentions_uuid4(self):
        from app.core.auth import extract_session_id
        req = self._make_request("garbage")
        with pytest.raises(HTTPException) as exc_info:
            extract_session_id(req)
        assert "UUID4" in exc_info.value.detail

    @pytest.mark.parametrize("valid_uuid4", [
        "a8098c1a-f86e-4a74-8a8f-8e1c1d1b1234",  # v4, variant 8
        "550e8400-e29b-41d4-8716-446655440000",  # v4, variant 8
        "550e8400-e29b-41d4-9716-446655440000",  # v4, variant 9
        "550e8400-e29b-41d4-a716-446655440000",  # v4, variant a
        "550e8400-e29b-41d4-b716-446655440000",  # v4, variant b
    ])
    def test_valid_uuid4_variants(self, valid_uuid4):
        from app.core.auth import extract_session_id
        req = self._make_request(valid_uuid4)
        assert extract_session_id(req) == valid_uuid4
