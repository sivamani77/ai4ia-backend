from __future__ import annotations

import uuid

from fastapi import HTTPException

from app.db.repositories.feedback_repository import FeedbackRepository
from app.services import notification_service


def submit_feedback(*, user_id: str, username: str, email: str, feedback: str) -> dict:
    record = FeedbackRepository().create({
        "id": f"fb_{uuid.uuid4().hex}",
        "user_id": user_id,
        "username": username,
        "email": email,
        "feedback": feedback,
        "status": "open",
    })
    notification_service.notify_admins(
        f"New feedback from {username or email}",
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
