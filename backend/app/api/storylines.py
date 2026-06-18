import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.paths import STORYLINES_DIR, SCRIPTS_DIR
from app.services.job_service import create_job
from app.services.script_runner import run_script_background

router = APIRouter(prefix="/storylines", tags=["Storylines"])


class GenerateStorylineRequest(BaseModel):
    prompt: str = Field(..., min_length=5)
    duration: int = 180
    style: str = "National Geographic-style documentary explainer"
    top_k: int = 90
    final_candidates: int = 35
    fallback_only: bool = False


def latest_storyline_path() -> Path | None:
    if not STORYLINES_DIR.exists():
        return None

    files = sorted(STORYLINES_DIR.glob("storyline_*.json"))
    return files[-1] if files else None


@router.get("")
def list_storylines():
    items = []

    if not STORYLINES_DIR.exists():
        return items

    for path in sorted(STORYLINES_DIR.glob("storyline_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            story = payload.get("storyline", {})
            items.append(
                {
                    "id": path.stem,
                    "filename": path.name,
                    "path": str(path),
                    "title": story.get("title"),
                    "logline": story.get("logline"),
                    "sections": len(story.get("sections", [])),
                    "created_at": payload.get("created_at"),
                    "target_duration_seconds": payload.get("target_duration_seconds"),
                }
            )
        except Exception:
            items.append({"id": path.stem, "filename": path.name, "path": str(path)})

    return items


def _load_storyline_with_id(path: Path) -> dict:
    """Load a storyline JSON and inject its id/path so the FE can pass it to
    /assets/generate and /exports/generate (the file itself has no self-id)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["storyline_id"] = path.stem
    payload["storyline_path"] = str(path)
    return payload


@router.get("/latest")
def get_latest_storyline():
    path = latest_storyline_path()

    if not path:
        raise HTTPException(status_code=404, detail="No storyline found")

    return _load_storyline_with_id(path)


@router.get("/{storyline_id}")
def get_storyline(storyline_id: str):
    path = STORYLINES_DIR / f"{storyline_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Storyline not found")

    return _load_storyline_with_id(path)


@router.get("/{storyline_id}/evidence-map")
def get_storyline_evidence_map(storyline_id: str):
    from app.services.insight_service import storyline_evidence_map

    path = STORYLINES_DIR / f"{storyline_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Storyline not found")

    storyline = json.loads(path.read_text(encoding="utf-8"))
    return storyline_evidence_map(storyline)


@router.post("/generate")
def generate_storyline(request: GenerateStorylineRequest):
    job = create_job("generate_storyline", request.model_dump())

    command = [
        "python",
        str(SCRIPTS_DIR / "07_generate_storyline.py"),
        "--prompt",
        request.prompt,
        "--duration",
        str(request.duration),
        "--style",
        request.style,
        "--top-k",
        str(request.top_k),
        "--final-candidates",
        str(request.final_candidates),
    ]

    if request.fallback_only:
        command.append("--fallback-only")

    run_script_background(
        job_id=job["job_id"],
        command=command,
        result_resolver=latest_storyline_path,
    )

    return {
        "job_id": job["job_id"],
        "status": "running",
        "message": "Storyline generation started",
    }
