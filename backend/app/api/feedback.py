from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user, require_role
from app.db.repositories.user_repository import UserRepository
from app.services import feedback_service

router = APIRouter(prefix="/feedback", tags=["Feedback"])


class SubmitFeedbackRequest(BaseModel):
    feedback: str = Field(..., min_length=1, max_length=4000)
    # username/email are accepted for backward compatibility with the
    # current frontend payload but ignored — see submit_feedback below.
    username: str | None = None
    email: str | None = None


class ResolveFeedbackRequest(BaseModel):
    resolution_message: str | None = Field(None, max_length=2000)


@router.post("", status_code=201)
def submit_feedback(payload: SubmitFeedbackRequest, current_user: CurrentUser = Depends(get_current_user)):
    # username/email come from the caller's own account record, not the
    # request body — a client could otherwise submit feedback under any
    # name it likes.
    account = UserRepository().get_by_id(current_user.id)
    username = (account or {}).get("username") or current_user.id
    email = (account or {}).get("email_id") or ""
    return feedback_service.submit_feedback(
        user_id=current_user.id, username=username, email=email, feedback=payload.feedback,
    )


@router.get("")
def list_feedbacks(
    status: str | None = Query(None, description="open | resolved"),
    current_user: CurrentUser = Depends(require_role("admin")),
):
    return {"feedbacks": feedback_service.list_feedbacks(status=status)}


@router.post("/{feedback_id}/resolve")
def resolve_feedback(
    feedback_id: str, payload: ResolveFeedbackRequest,
    current_user: CurrentUser = Depends(require_role("admin")),
):
    return feedback_service.resolve_feedback(
        feedback_id, resolution_message=payload.resolution_message, resolved_by=current_user.id,
    )
