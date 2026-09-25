from __future__ import annotations

import uuid

from app.db.repositories.tour_repository import TourRepository


def list_seen_tours(user_id: str) -> list[str]:
    return TourRepository().list_seen_page_ids(user_id)


def mark_tour_seen(user_id: str, page_id: str) -> None:
    TourRepository().mark_seen(f"tour_{uuid.uuid4().hex}", user_id, page_id)
