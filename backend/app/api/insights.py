from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import INSIGHT_DEFAULT_LIMIT
from app.services import insight_service

router = APIRouter(prefix="/insights", tags=["Insights"])


class GenerateInsightRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    filters: dict | None = None
    limit: int = Field(INSIGHT_DEFAULT_LIMIT, ge=5, le=100)


@router.post("/generate")
def api_generate_insight(req: GenerateInsightRequest):
    try:
        return insight_service.generate_insight(req.query, req.filters, req.limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Insight generation failed: {exc}")


@router.get("")
def api_list_insights():
    items = insight_service.list_insights()
    return {"count": len(items), "insights": items}


@router.get("/{insight_id}")
def api_get_insight(insight_id: str):
    insight = insight_service.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return insight


@router.get("/{insight_id}/sources")
def api_insight_sources(insight_id: str):
    insight = insight_service.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return insight_service.build_sources_map(insight)
