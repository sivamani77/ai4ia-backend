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


def test_insight_decline_is_200_not_422(authed, monkeypatch):
    """GAP-005: no relevant evidence is a result, not a validation error."""
    import app.services.insight_service as insight_service
    from app.services.insight_service import OUTCOME_INSUFFICIENT_EVIDENCE

    monkeypatch.setattr(
        insight_service, "generate_insight",
        lambda *a, **k: insight_service._declined_insight("insight_x", "q", None))

    response = authed.post("/api/insights/generate", json={"query": "bonsai care"})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == OUTCOME_INSUFFICIENT_EVIDENCE
    assert body["patterns"] == []
    assert body["decline_reason"]


def test_genuine_value_error_is_still_422(authed, monkeypatch):
    """The decline path must not swallow real validation failures."""
    import app.services.insight_service as insight_service

    def _boom(*a, **k):
        raise ValueError("something genuinely invalid")

    monkeypatch.setattr(insight_service, "generate_insight", _boom)
    response = authed.post("/api/insights/generate", json={"query": "anything"})
    assert response.status_code == 422
