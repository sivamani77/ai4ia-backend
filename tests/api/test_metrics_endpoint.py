from __future__ import annotations


def test_metrics_endpoint_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "jobs_created_total" in response.text


def test_metrics_endpoint_no_api_key_required(client, monkeypatch):
    from app.core.settings import settings

    monkeypatch.setattr(settings, "api_key", "some-secret-key")
    response = client.get("/metrics")
    assert response.status_code == 200
