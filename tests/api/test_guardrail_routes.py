"""A guardrail rejection must surface as 400, not as a 502 generation failure.

Both routers wrap their service call in a blanket `except Exception` that
reports 502; GuardrailRejection has to be caught ahead of it or a blocked
query looks like an outage. (GAP-003)
"""
from __future__ import annotations

import pytest

from app.services.guardrail_service import GuardrailRejection, QUERY_REJECTION_MESSAGE


def _raise(*args, **kwargs):
    raise GuardrailRejection(QUERY_REJECTION_MESSAGE, category="harmful", reason="weapons")


@pytest.fixture
def authed(client):
    import app.main as main
    from app.core.auth import CurrentUser, get_current_user

    main.app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        id="test-user", persona="general")
    try:
        yield client
    finally:
        main.app.dependency_overrides.pop(get_current_user, None)


def test_insight_rejection_is_400(authed, monkeypatch):
    import app.services.insight_service as insight_service
    monkeypatch.setattr(insight_service, "generate_insight", _raise)

    response = authed.post("/api/insights/generate", json={"query": "adversarial text"})
    assert response.status_code == 400
    assert response.json()["detail"] == QUERY_REJECTION_MESSAGE


@pytest.mark.parametrize("service_attr,path,payload", [
    ("compare_speakers", "/api/compare/speakers", {"speakers": ["Alice", "Bob"], "topic": "x"}),
    ("synthesize_theme", "/api/synthesis/theme", {"theme": "adversarial text"}),
    ("build_relay", "/api/relay", {"topic": "adversarial text"}),
])
def test_compare_rejections_are_400(authed, monkeypatch, service_attr, path, payload):
    import app.services.comparison_service as comparison_service
    monkeypatch.setattr(comparison_service, service_attr, _raise)

    response = authed.post(path, json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == QUERY_REJECTION_MESSAGE
