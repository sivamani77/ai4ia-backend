from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, get_current_user
from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket
from app.services.concept_mindmap_service import generate_concept_mindmap


router = APIRouter(prefix="/v1", tags=["Concept Mindmap"])

# 10 requests per user per 60 seconds
_mindmap_limiter = TokenBucket(rate=10, per=60.0)


class ConceptMindmapRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=300)
    storyline_id: str | None = Field(None, max_length=100)


@router.post("/concept-mindmap")
def create_concept_mindmap(
    request: ConceptMindmapRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    if not _mindmap_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="concept_mindmap").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before trying again.",
        )
    try:
        return generate_concept_mindmap(
            query=request.query,
            storyline_id=request.storyline_id,
            mode="legacy",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build concept mindmap: {exc}") from exc
