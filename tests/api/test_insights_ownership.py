"""Insights are private to the user who generated them (C-01)."""
from __future__ import annotations

import pytest

OWNER_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_ID = "550e8400-e29b-41d4-a716-446655440001"
ADMIN_ID = "550e8400-e29b-41d4-a716-446655440002"


@pytest.fixture(autouse=True)
def _jwt_signing_key(monkeypatch):
    from app.core import security as security_module

    monkeypatch.setattr(
        security_module.settings, "jwt_signing_key",
        "test-signing-key-that-is-at-least-32-bytes-long", raising=False,
    )


def _auth_headers(user_id: str, persona: str = "general") -> dict:
    from app.core.security import create_access_token

    token = create_access_token(user_id=user_id, persona=persona)
    return {"Authorization": f"Bearer {token}"}


class _FakeInsightRepository:
    """In-memory stand-in with the same owner filtering as the real one."""

    rows = {
        "insight_owned": {"id": "insight_owned", "user_id": OWNER_ID, "query": "q1", "title": "t1"},
        "insight_other": {"id": "insight_other", "user_id": OTHER_ID, "query": "q2", "title": "t2"},
        "insight_legacy": {"id": "insight_legacy", "user_id": None, "query": "q3", "title": "t3"},
    }

    def get(self, insight_id, user_id=None):
        row = self.rows.get(insight_id)
        if row is None or (user_id is not None and row["user_id"] != user_id):
            return None
        return dict(row)

    def list(self, limit=100, user_id=None):
        return [dict(r) for r in self.rows.values() if user_id is None or r["user_id"] == user_id]


@pytest.fixture
def insights_client(client, monkeypatch):
    import app.services.insight_service as insight_service

    monkeypatch.setattr(insight_service, "InsightRepository", _FakeInsightRepository)
    return client


def test_list_returns_only_the_callers_insights(insights_client):
    response = insights_client.get("/api/insights", headers=_auth_headers(OWNER_ID))

    assert response.status_code == 200
    assert [i["insight_id"] for i in response.json()["insights"]] == ["insight_owned"]


@pytest.mark.parametrize("suffix", ["", "/sources", "/provenance"])
def test_another_users_insight_is_not_found(insights_client, suffix):
    response = insights_client.get(f"/api/insights/insight_other{suffix}", headers=_auth_headers(OWNER_ID))

    assert response.status_code == 404


def test_ownerless_insight_is_hidden_from_users(insights_client):
    response = insights_client.get("/api/insights/insight_legacy", headers=_auth_headers(OWNER_ID))

    assert response.status_code == 404


def test_owner_can_read_their_insight(insights_client):
    response = insights_client.get("/api/insights/insight_owned", headers=_auth_headers(OWNER_ID))

    assert response.status_code == 200
    assert response.json()["insight_id"] == "insight_owned"


def test_admin_sees_every_insight(insights_client, user_states):
    from app.core.auth import UserState

    user_states[ADMIN_ID] = UserState(token_version=0, persona="admin", active=True)
    headers = _auth_headers(ADMIN_ID, persona="admin")

    listed = insights_client.get("/api/insights", headers=headers)
    legacy = insights_client.get("/api/insights/insight_legacy", headers=headers)

    assert listed.status_code == 200
    assert listed.json()["count"] == 3
    assert legacy.status_code == 200


def test_generated_insight_records_its_owner_and_a_random_id(monkeypatch):
    import app.services.insight_service as insight_service

    created = {}

    class _Repo:
        def create(self, data):
            created.update(data)

    monkeypatch.setattr(insight_service, "InsightRepository", _Repo)
    monkeypatch.setattr(insight_service.guardrail_service, "assert_query_allowed", lambda *a, **k: None)
    monkeypatch.setattr(
        insight_service, "build_evidence_pool",
        lambda *a, **k: [{"ref": 1, "fragment_id": "f1", "stream_url": "", "start": 0, "end": 1}],
    )
    monkeypatch.setattr(insight_service.evidence_gate, "evidence_is_sufficient", lambda *a, **k: (True, {}))
    monkeypatch.setattr(insight_service, "_build_user_message", lambda *a, **k: "")
    monkeypatch.setattr(insight_service, "generate_json", lambda *a, **k: {"patterns": [], "contrasts": []})
    monkeypatch.setattr(insight_service, "_emit_insight_decision_record", lambda *a, **k: None)

    result = insight_service.generate_insight("what is ai access", user_id=OWNER_ID)

    assert created["user_id"] == OWNER_ID
    assert result["insight_id"].startswith("insight_")
    assert len(result["insight_id"]) == len("insight_") + 32
