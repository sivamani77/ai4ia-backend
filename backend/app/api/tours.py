from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user
from app.services import tour_service

router = APIRouter(prefix="/tours", tags=["Tours"])


class MarkTourSeenRequest(BaseModel):
    page_id: str = Field(..., min_length=1, max_length=100)


@router.get("/seen")
def list_seen_tours(current_user: CurrentUser = Depends(get_current_user)):
    return {"seen_tours": tour_service.list_seen_tours(current_user.id)}


@router.post("/seen", status_code=201)
def mark_tour_seen(payload: MarkTourSeenRequest, current_user: CurrentUser = Depends(get_current_user)):
    tour_service.mark_tour_seen(current_user.id, payload.page_id)
    return {"success": True, "page_id": payload.page_id}
