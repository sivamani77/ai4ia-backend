from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user
from app.db.repositories.notification_repository import NotificationRepository
from app.services import notification_service

router = APIRouter(prefix="/notifications", tags=["Notifications"])


class CreateNotificationRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    type: str = Field("info", max_length=50)
    redirect_page: str | None = None
    meta: dict | None = None


@router.get("")
def list_notifications(
    limit: int = Query(50, ge=1, le=200),
    current_user: CurrentUser = Depends(get_current_user),
):
    return {"notifications": NotificationRepository().list_by_user(current_user.id, limit=limit)}


@router.post("", status_code=201)
def create_notification(payload: CreateNotificationRequest, current_user: CurrentUser = Depends(get_current_user)):
    # Self-only: this creates a notification for the caller, not anyone
    # else — there is no way to target another user through this route.
    return notification_service.notify_user(
        current_user.id, payload.message,
        type=payload.type, redirect_page=payload.redirect_page, meta=payload.meta,
    )


@router.post("/{notification_id}/read")
def mark_notification_read(notification_id: str, current_user: CurrentUser = Depends(get_current_user)):
    ok = NotificationRepository().mark_read(notification_id, current_user.id)
    if not ok:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"marked_read": True}


@router.post("/mark-all-read")
def mark_all_notifications_read(current_user: CurrentUser = Depends(get_current_user)):
    NotificationRepository().mark_all_read(current_user.id)
    return {"marked_all_read": True}


@router.delete("")
def clear_all_notifications(current_user: CurrentUser = Depends(get_current_user)):
    NotificationRepository().delete_all_for_user(current_user.id)
    return {"cleared": True}
