"""Every 401 carries a WWW-Authenticate challenge (A-03); a short signing key
stops start-up (C-06)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _jwt_signing_key(monkeypatch):
    from app.core import security as security_module

    monkeypatch.setattr(
        security_module.settings, "jwt_signing_key",
        "test-signing-key-that-is-at-least-32-bytes-long", raising=False,
    )


def test_missing_token_gets_a_bearer_challenge(client):
    response = client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer realm="ai4ia"'


def test_invalid_token_challenge_names_the_error(client):
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})

    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["WWW-Authenticate"]


def test_expired_token_keeps_its_code_and_says_expired(client, monkeypatch):
    from app.core import security

    monkeypatch.setattr(security.settings, "access_token_ttl_seconds", -10)
    token = security.create_access_token(user_id="u1", persona="general")

    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "token_expired"
    assert "expired" in response.headers["WWW-Authenticate"]


def test_bare_401_from_a_route_gets_the_default_challenge(client):
    response = client.post("/api/auth/refresh")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Bearer realm="ai4ia"'


def test_non_401_errors_get_no_challenge(client):
    response = client.get("/api/__nope__")

    assert response.status_code == 404
    assert "WWW-Authenticate" not in response.headers


def test_short_signing_key_stops_start_up(api_app, monkeypatch):
    from app.core import security

    monkeypatch.setattr(security.settings, "jwt_signing_key", "too-short")
    with pytest.raises(RuntimeError, match="shorter than"):
        with TestClient(api_app):
            pass
