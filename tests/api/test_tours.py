"""Tour-seen state: the frontend's last remaining direct-Supabase call
(listSeenTours/markTourSeen), rebuilt per the frontend team's handover
spec — self-scoped, idempotent mark."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    yield
    api_app.dependency_overrides.clear()


def test_requires_auth(api_app, mock_settings):
    with TestClient(api_app) as client:
        response = client.get("/api/tours/seen")
    assert response.status_code == 401


def test_list_and_mark_seen(api_app, mock_settings, monkeypatch):
    import app.db.repositories.tour_repository as tour_repository

    seen: dict[str, list[str]] = {"u-1": []}
    monkeypatch.setattr(
        tour_repository.TourRepository, "list_seen_page_ids",
        lambda self, user_id: seen.get(user_id, []),
    )
    monkeypatch.setattr(
        tour_repository.TourRepository, "mark_seen",
        lambda self, tour_id, user_id, page_id: seen.setdefault(user_id, []).append(page_id),
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u-1", persona="general")
    with TestClient(api_app) as client:
        r = client.get("/api/tours/seen")
        assert r.status_code == 200 and r.json() == {"seen_tours": []}

        r = client.post("/api/tours/seen", json={"page_id": "dashboard"})
        assert r.status_code == 201
        assert r.json() == {"success": True, "page_id": "dashboard"}

        r = client.get("/api/tours/seen")
        assert r.json() == {"seen_tours": ["dashboard"]}


def test_mark_seen_is_scoped_per_user(api_app, mock_settings, monkeypatch):
    import app.db.repositories.tour_repository as tour_repository

    calls = []
    monkeypatch.setattr(tour_repository.TourRepository, "list_seen_page_ids", lambda self, user_id: [])
    monkeypatch.setattr(
        tour_repository.TourRepository, "mark_seen",
        lambda self, tour_id, user_id, page_id: calls.append((user_id, page_id)),
    )

    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="u-42", persona="general")
    with TestClient(api_app) as client:
        client.post("/api/tours/seen", json={"page_id": "create-video"})

    assert calls == [("u-42", "create-video")]
