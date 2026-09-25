from __future__ import annotations

import logging

from app.db.postgres import pg_execute, pg_fetch_all

logger = logging.getLogger("ai4ia.db.repositories.tours")


class TourRepository:
    table_name = "user_seen_tours"

    def list_seen_page_ids(self, user_id: str) -> list[str]:
        rows = pg_fetch_all(
            "select page_id from user_seen_tours where user_id = %s",
            (user_id,),
        )
        return [row["page_id"] for row in rows]

    def mark_seen(self, tour_id: str, user_id: str, page_id: str) -> None:
        # Idempotent: marking an already-seen tour is a no-op, not an error —
        # matches the frontend spec ("ON CONFLICT (username, page_id) DO NOTHING").
        pg_execute(
            """
            insert into user_seen_tours (id, user_id, page_id)
            values (%s, %s, %s)
            on conflict (user_id, page_id) do nothing
            """,
            (tour_id, user_id, page_id),
        )
