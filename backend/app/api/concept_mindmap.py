from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.concept_mindmap_service import generate_concept_mindmap


router = APIRouter(prefix="/v1", tags=["Concept Mindmap"])


class ConceptMindmapRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=300)
    storyline_id: str | None = Field(None, max_length=100)
    mode: str = Field(
        "video",
        pattern="^(video|legacy)$",
        description="'video' = NotebookLM-style map of what the video says; 'legacy' = theme/section-centric map",
    )


@router.post("/concept-mindmap")
def create_concept_mindmap(request: ConceptMindmapRequest):
    try:
        return generate_concept_mindmap(
            query=request.query,
            storyline_id=request.storyline_id,
            mode=request.mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build concept mindmap: {exc}") from exc
