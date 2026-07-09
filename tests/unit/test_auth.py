"""Unit tests for app.core.auth — API key enforcement."""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from unittest.mock import patch


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

