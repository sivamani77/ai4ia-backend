"""Read-path session/activity queries for the admin activity feed and the
self-service /auth/me/sessions view.

Unlike app/services/session_service.py — which is best-effort and swallows
every exception because it sits in the login/logout critical path — these
are ordinary read endpoints. A Postgres failure here should surface as a
real error to the caller, not silently return an empty list and lie to
whoever's looking at the feed.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.db.repositories import get_session_repository, get_user_repository


def _enrich_with_usernames(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """user_sessions (pipeline Postgres) and app_users (auth Supabase) are
    separate databases — no SQL JOIN is possible, so usernames are resolved
    with one batch lookup instead of a query per row."""
    user_ids = sorted({s["user_id"] for s in sessions if s.get("user_id")})
    if not user_ids:
        return sessions
    users = {u["id"]: u for u in get_user_repository().get_by_ids(user_ids)}
    for s in sessions:
        user = users.get(s.get("user_id"))
        s["username"] = user.get("username") if user else None
    return sessions


def list_all_sessions(
    *,
    limit: int,
    offset: int,
    user_id: str | None = None,
    active_only: bool | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Global admin activity feed — every user's sessions."""
    sessions, total = get_session_repository().list_all(
        limit=limit, offset=offset, user_id=user_id, active_only=active_only, since=since
    )
    return {
        "sessions": _enrich_with_usernames([dict(s) for s in sessions]),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def list_sessions_for_user(user_id: str, *, limit: int, offset: int) -> dict[str, Any]:
    """One user's full session history. Used by both the admin drill-down
    (GET /admin/users/{user_id}/sessions, arbitrary user_id — 404s if it
    doesn't exist) and the self-service view (GET /auth/me/sessions,
    user_id from a verified access token, so the 404 branch is
    unreachable in practice but harmless to check)."""
    if get_user_repository().get_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    sessions, total = get_session_repository().list_for_user(user_id, limit=limit, offset=offset)
    return {
        "sessions": [dict(s) for s in sessions],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


__all__ = ["list_all_sessions", "list_sessions_for_user"]
