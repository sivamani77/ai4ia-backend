"""The dashboard is public only as landing-page counts (A-01)."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _jwt_signing_key(monkeypatch):
    from app.core import security as security_module

    monkeypatch.setattr(
        security_module.settings, "jwt_signing_key",
        "test-signing-key-that-is-at-least-32-bytes-long", raising=False,
    )


def _auth_headers(persona: str) -> dict:
    from app.core.security import create_access_token

    token = create_access_token(user_id="550e8400-e29b-41d4-a716-446655440000", persona=persona)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def counts(monkeypatch):
    import app.services.dashboard_service as dashboard_service

    monkeypatch.setattr(dashboard_service, "_public_summary_cache", (None, 0.0))
    monkeypatch.setattr(dashboard_service, "_raw_video_count", lambda: 310)
    monkeypatch.setattr(dashboard_service, "_classified_fragment_count", lambda: 5112)
    monkeypatch.setattr(dashboard_service, "_speaker_count", lambda: 307)
    monkeypatch.setattr(dashboard_service, "_theme_count", lambda: 7)


def test_anonymous_dashboard_returns_only_archive_counts(client, counts):
    response = client.get("/api/dashboard")

    assert response.status_code == 200
    assert response.json() == {
        "archive": {"raw_videos": 310, "classified_fragments": 5112, "speakers": 307, "themes": 7},
    }


def test_public_summary_is_cached(monkeypatch, counts):
    import app.services.dashboard_service as dashboard_service

    first = dashboard_service.get_public_summary()
    monkeypatch.setattr(dashboard_service, "_speaker_count", lambda: 999)

    assert dashboard_service.get_public_summary() is first


def test_failed_count_is_null_and_retried_soon(monkeypatch, counts):
    import app.services.dashboard_service as dashboard_service

    def _broken():
        raise RuntimeError("s3 down")

    monkeypatch.setattr(dashboard_service, "_raw_video_count", _broken)
    summary = dashboard_service.get_public_summary()
    assert summary["archive"]["raw_videos"] is None
    assert summary["archive"]["speakers"] == 307

    # Cached only for the short retry window, not the full five minutes.
    _, expires_at = dashboard_service._public_summary_cache
    assert expires_at - dashboard_service.time.monotonic() <= dashboard_service._PUBLIC_SUMMARY_RETRY_TTL


def test_admin_dashboard_requires_admin(client):
    assert client.get("/api/admin/dashboard").status_code == 401
    assert client.get("/api/admin/dashboard", headers=_auth_headers("general")).status_code == 403


def test_admin_dashboard_returns_full_details(client, monkeypatch):
    import app.api.dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module, "get_admin_dashboard", lambda: {"models": {"tts": "edge-tts"}})

    response = client.get("/api/admin/dashboard", headers=_auth_headers("admin"))

    assert response.status_code == 200
    assert response.json() == {"models": {"tts": "edge-tts"}}
