from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import require_models
from app.services import model_gate
from app.services.concept_mindmap_service import generate_concept_mindmap


router = APIRouter(prefix="/v1", tags=["Concept Mindmap"])


class ConceptMindmapRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=300)
    storyline_id: str | None = Field(None, max_length=100)


@router.post(
    "/concept-mindmap",
    dependencies=[Depends(require_models(model_gate.LLM))],
)
def create_concept_mindmap(request: ConceptMindmapRequest):
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
