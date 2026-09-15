from __future__ import annotations


def test_health_ok(client, monkeypatch):
    import app.main as main
    import app.services.video_job_runner as video_job_runner

    monkeypatch.setattr(main, "_check_postgres", lambda: {"ok": True})
    monkeypatch.setattr(main, "_check_chroma", lambda: {"ok": True})
    monkeypatch.setattr(main, "_check_ollama", lambda: {"ok": True})
    monkeypatch.setattr(video_job_runner, "_redis_reachable", lambda: True)

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "AI4IA Demo Video Studio"
    assert data["worker_mode"] == "celery"
    assert data["checks"]["redis"]["ok"] is True


def test_health_degraded_when_dependencies_fail(client, monkeypatch):
    import app.main as main
    import app.services.video_job_runner as video_job_runner

    monkeypatch.setattr(main, "_check_postgres", lambda: {"ok": False, "error": "db down"})
    monkeypatch.setattr(main, "_check_chroma", lambda: {"ok": True})
    monkeypatch.setattr(main, "_check_ollama", lambda: {"ok": False, "error": "ollama down"})
    monkeypatch.setattr(video_job_runner, "_redis_reachable", lambda: False)

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["worker_mode"] == "thread_fallback"
    assert data["checks"]["redis"]["ok"] is False
    assert data["checks"]["postgres"]["ok"] is False
    assert data["checks"]["ollama"]["ok"] is False


def test_api_key_required_when_configured(client, monkeypatch):
    import app.core.auth as auth_module
    import app.services.dashboard_service as dashboard_service

    monkeypatch.setattr(auth_module.settings, "api_key", "super-secret", raising=False)
    monkeypatch.setattr(dashboard_service, "get_dashboard", lambda: {"ok": True})

    response = client.get("/api/dashboard")

    assert response.status_code == 403
    assert "API key" in response.json()["detail"]


def test_api_key_allows_authenticated_requests(client, monkeypatch):
    import app.core.auth as auth_module
    import app.services.dashboard_service as dashboard_service

    monkeypatch.setattr(auth_module.settings, "api_key", "super-secret", raising=False)
    monkeypatch.setattr(dashboard_service, "get_dashboard", lambda: {"ok": True})

    response = client.get("/api/dashboard", headers={"X-API-Key": "super-secret"})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
