"""Unit tests for app.core.auth.get_current_user / require_role — the real
identity dependency that replaced extract_session_id. Fail-closed behavior
here is what fixes the original "no header = see everyone's jobs" hole."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import CurrentUser, get_current_user, require_role
from app.core.security import create_access_token


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class TestGetCurrentUser:
    @pytest.fixture(autouse=True)
    def _jwt_signing_key(self, monkeypatch):
        """create_access_token needs a non-empty HMAC key; settings.jwt_signing_key
        defaults to "" with no .env present in this environment. Pin one here —
        see the identical fixture in test_core_security.py."""
        from app.core import security as security_module

        monkeypatch.setattr(security_module.settings, "jwt_signing_key", "test-signing-key-that-is-at-least-32-bytes-long", raising=False)

    def test_valid_token_resolves_user(self):
        token = create_access_token(user_id="user-123", persona="general")
        user = get_current_user(credentials=_bearer(token))
        assert isinstance(user, CurrentUser)
        assert user.id == "user-123"
        assert user.persona == "general"

    def test_missing_credentials_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=None)
        assert exc_info.value.status_code == 401

    def test_empty_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=_bearer(""))
        assert exc_info.value.status_code == 401

    def test_garbage_token_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=_bearer("not-a-real-jwt"))
        assert exc_info.value.status_code == 401

    def test_expired_token_raises_401_with_token_expired_code(self, monkeypatch):
        import time
        from app.core import security as security_module

        monkeypatch.setattr(security_module.settings, "access_token_ttl_seconds", 0, raising=False)
        token = create_access_token(user_id="user-123", persona="general")
        time.sleep(1.1)

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=_bearer(token))
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["code"] == "token_expired"

    def test_tampered_token_raises_401(self):
        token = create_access_token(user_id="user-123", persona="general")
        # Change a character inside the signature: the last one carries unused
        # base64 padding bits, so swapping it can decode to the same bytes.
        head, _, signature = token.rpartition(".")
        mid = len(signature) // 2
        swapped = "A" if signature[mid] != "A" else "B"
        tampered = f"{head}.{signature[:mid]}{swapped}{signature[mid + 1:]}"
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=_bearer(tampered))
        assert exc_info.value.status_code == 401


class TestRequireRole:
    def test_allowed_persona_passes(self):
        dep = require_role("admin")
        user = CurrentUser(id="user-1", persona="admin")
        assert dep(user=user) is user

    def test_disallowed_persona_raises_403(self):
        dep = require_role("admin")
        user = CurrentUser(id="user-1", persona="general")
        with pytest.raises(HTTPException) as exc_info:
            dep(user=user)
        assert exc_info.value.status_code == 403

    def test_multiple_allowed_personas(self):
        dep = require_role("admin", "expert")
        assert dep(user=CurrentUser(id="u", persona="expert")) is not None
        with pytest.raises(HTTPException):
            dep(user=CurrentUser(id="u", persona="general"))
