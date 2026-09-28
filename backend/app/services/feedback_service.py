from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException

from app.core.settings import settings
from app.db.repositories.feedback_repository import FeedbackRepository
from app.services import feedback_auto_reply, notification_service

logger = logging.getLogger("ai4ia.services.feedback")


def _notify_user_best_effort(user_id: str, message: str, **kwargs) -> None:
    """Same fail-open contract as notify_admins: the feedback row and any
    auto-reply are already committed by the time this runs, so a transient
    notification failure (e.g. connection-pool exhaustion) must never 500 a
    request whose real work already succeeded."""
    try:
        notification_service.notify_user(user_id, message, **kwargs)
    except Exception:
        logger.warning("notify_user failed for feedback to %s", user_id, exc_info=True)


def submit_feedback(*, user_id: str, username: str, email: str, feedback: str) -> dict:
    repo = FeedbackRepository()
    if repo.count_recent_by_user(user_id) >= settings.feedback_submissions_per_user_per_hour:
        raise HTTPException(status_code=429, detail="Too much feedback submitted recently. Please try again later.")

    record = repo.create({
        "id": f"fb_{uuid.uuid4().hex}",
        "user_id": user_id,
        "username": username,
        "email": email,
        "feedback": feedback,
        "status": "open",
    })

    # Best-effort: an LLM failure here must never break feedback submission,
    # same contract as notify_admins below.
    auto_reply = None
    try:
        auto_reply = feedback_auto_reply.generate(user_id, feedback)
    except Exception:
        logger.warning("feedback_auto_reply.generate raised; falling back to canned reply", exc_info=True)

    if auto_reply is not None:
        record = repo.set_auto_reply(record["id"], category=auto_reply.category, reply=auto_reply.reply) or record
        _notify_user_best_effort(
            user_id, auto_reply.reply, type="info",
            meta={"feedback_id": record["id"], "category": auto_reply.category},
        )
        admin_note = f"New feedback from {username or email} ({auto_reply.category})"
    else:
        _notify_user_best_effort(
            user_id, feedback_auto_reply.CANNED_FALLBACK, type="info",
            meta={"feedback_id": record["id"]},
        )
        admin_note = f"New feedback from {username or email}"

    notification_service.notify_admins(
        admin_note,
        type="info",
        redirect_page="admin-settings",
        meta={"tab": "feedbacks", "feedback_id": record["id"]},
    )
    return record


def list_feedbacks(*, status: str | None = None, limit: int = 200) -> list[dict]:
    return FeedbackRepository().list(status=status, limit=limit)  # type: ignore[return-value]


def resolve_feedback(feedback_id: str, *, resolution_message: str | None, resolved_by: str) -> dict:
    repo = FeedbackRepository()
    existing = repo.get(feedback_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    if existing["status"] == "resolved":
        return existing  # type: ignore[return-value]

    updated = repo.resolve(feedback_id, resolution_message=resolution_message, resolved_by=resolved_by)
    if updated is None:
        raise HTTPException(status_code=404, detail="Feedback not found")

    notification_service.notify_user(
        updated["user_id"],
        "Your feedback has been seen and the problem is resolved. Thank you!",
        type="success",
        meta={"feedback_id": feedback_id, "resolution_message": resolution_message},
    )
    return updated  # type: ignore[return-value]
