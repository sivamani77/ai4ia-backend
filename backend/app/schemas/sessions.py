"""Session/activity-log schemas — admin global feed, admin per-user
drill-down, and the self-service /auth/me/sessions view. All three share
the same shape; see app/services/session_activity_service.py.
"""
from __future__ import annotations

from pydantic import BaseModel


class SessionRecord(BaseModel):
    id: str
    user_id: str
    # Enriched onto the admin global feed only (one batch lookup against the
    # separate auth-project UserRepository) — omitted on the per-user and
    # self-service views, where the caller already knows whose sessions
    # they're looking at.
    username: str | None = None
    created_at: str | None = None
    last_seen_at: str | None = None
    ended_at: str | None = None
    end_reason: str | None = None
    user_agent: str | None = None
    ip: str | None = None
    warmup_state: str
    warmup_started_at: str | None = None
    warmup_ready_at: str | None = None
    warmup_error: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionRecord]
    total: int
    limit: int
    offset: int


__all__ = ["SessionRecord", "SessionListResponse"]
