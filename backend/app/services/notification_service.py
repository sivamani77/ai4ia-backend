"""Notification creation and delivery.

notify_user() backs the public POST /notifications route (self-notify only —
see notificationsApi.js's createNotification, used for persistent toast-style
alerts like "your video is ready"). notify_admins() is internal-only, used by
other services (feedback_service) to alert every admin — there is no public
"notify someone else" endpoint, by design.
"""
from __future__ import annotations

import logging
import uuid

from app.db.repositories.notification_repository import NotificationRepository
from app.db.repositories.user_repository import UserRepository

logger = logging.getLogger("ai4ia.services.notifications")


def _new_id() -> str:
    return f"ntf_{uuid.uuid4().hex}"


def notify_user(
    user_id: str, message: str, *, type: str = "info",
    redirect_page: str | None = None, meta: dict | None = None,
) -> dict:
    return NotificationRepository().create({
        "id": _new_id(),
        "user_id": user_id,
        "message": message,
        "type": type,
        "redirect_page": redirect_page,
        "meta": meta or {},
    })


def notify_admins(
    message: str, *, type: str = "info",
    redirect_page: str | None = None, meta: dict | None = None,
) -> int:
    """Best-effort: a failure here must never break the caller's real
    action (e.g. submitting feedback), so exceptions are logged, not raised."""
    try:
        admins, _total = UserRepository().list_all(persona="admin", status="active", limit=1000)
    except Exception:
        logger.exception("notify_admins: failed to list admins")
        return 0
    rows = [
        {
            "id": _new_id(),
            "user_id": admin["id"],
            "message": message,
            "type": type,
            "redirect_page": redirect_page,
            "meta": meta or {},
        }
        for admin in admins
    ]
    try:
        return NotificationRepository().create_many(rows)
    except Exception:
        logger.exception("notify_admins: failed to create notifications")
        return 0
