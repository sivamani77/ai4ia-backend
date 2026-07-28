"""Login-session tracking (db/migrations/0005_user_sessions.sql).

`session_id` is the refresh-token family_id (see auth_service._issue_refresh_token)
— one stable identifier per login that survives refresh rotation, not a newly
minted id. This module owns only the durable record of who logged in, when,
and (as a status variable — see app/services/model_gate.py) whether that
login's model warm-up finished. It never decides whether a model host is
reachable; the gate probes the hosts directly instead of trusting this state.

Every method is best-effort: a Postgres hiccup here must never fail a login,
a refresh, or a feature request, so exceptions are caught and logged, not
raised.
"""
from __future__ import annotations

import logging

from app.db.repositories import get_session_repository

logger = logging.getLogger("ai4ia.services.session")


def create_session(
    *,
    session_id: str,
    user_id: str,
    user_agent: str | None = None,
    ip: str | None = None,
) -> None:
    try:
        get_session_repository().create(
            session_id=session_id, user_id=user_id, user_agent=user_agent, ip=ip
        )
    except Exception:  # noqa: BLE001
        logger.warning("session_service.create_session failed for %s", session_id, exc_info=True)


def touch_session(session_id: str) -> None:
    try:
        get_session_repository().touch(session_id)
    except Exception:  # noqa: BLE001
        logger.debug("session_service.touch_session failed for %s", session_id, exc_info=True)


def end_session(session_id: str, *, reason: str) -> None:
    try:
        get_session_repository().end(session_id, reason=reason)
    except Exception:  # noqa: BLE001
        logger.warning("session_service.end_session failed for %s", session_id, exc_info=True)


def end_all_sessions_for_user(user_id: str, *, reason: str) -> None:
    try:
        get_session_repository().end_all_for_user(user_id, reason=reason)
    except Exception:  # noqa: BLE001
        logger.warning("session_service.end_all_sessions_for_user failed for %s", user_id, exc_info=True)


def mark_warmup_started(session_id: str) -> None:
    try:
        get_session_repository().set_warmup_state(session_id, state="warming", started_at=True)
    except Exception:  # noqa: BLE001
        logger.debug("session_service.mark_warmup_started failed for %s", session_id, exc_info=True)


def mark_warmup_ready(session_id: str) -> None:
    try:
        get_session_repository().set_warmup_state(session_id, state="ready", ready_at=True)
    except Exception:  # noqa: BLE001
        logger.debug("session_service.mark_warmup_ready failed for %s", session_id, exc_info=True)


def mark_warmup_error(session_id: str, error: str) -> None:
    try:
        get_session_repository().set_warmup_state(session_id, state="error", error=error[:500])
    except Exception:  # noqa: BLE001
        logger.debug("session_service.mark_warmup_error failed for %s", session_id, exc_info=True)


__all__ = [
    "create_session",
    "touch_session",
    "end_session",
    "end_all_sessions_for_user",
    "mark_warmup_started",
    "mark_warmup_ready",
    "mark_warmup_error",
]
