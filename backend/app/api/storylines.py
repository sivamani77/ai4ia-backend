# DEPRECATED: This router uses the legacy 3-step script-runner flow
# (job_service.create_job + run_script_background calling 07_generate_storyline.py).
# It is superseded by POST /api/video-jobs, which runs the full pipeline end-to-end
# through the app.pipeline stage modules. Do not add new features here.

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.paths import STORYLINES_DIR, SCRIPTS_DIR
from app.db.repositories.storyline_repository import StorylineRepository
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


def _row_to_list_item(row: dict) -> dict:
    sections = row.get("sections") or []
    return {
        "id": row.get("id"),
        "filename": f"{row.get('id')}.json",
        "path": str(STORYLINES_DIR / f"{row.get('id')}.json"),
        "title": row.get("title"),
        "logline": row.get("logline"),
        "sections": len(sections),
        "created_at": row.get("created_at"),
        "target_duration_seconds": row.get("target_duration_seconds"),
    }


def _row_to_full(row: dict) -> dict:
    """Return the full payload the frontend expects, with injected id fields."""
    full = dict(row.get("full_payload") or row)
    full["storyline_id"] = row.get("id")
    full["storyline_path"] = str(STORYLINES_DIR / f"{row.get('id')}.json")
    # Ensure top-level fields callers depend on are always present
    full.setdefault("title", row.get("title"))
    full.setdefault("logline", row.get("logline"))
    return full


@router.get("")
def list_storylines():
    try:
        rows = StorylineRepository().list()
        return [_row_to_list_item(dict(r)) for r in rows]
    except Exception:
        # Filesystem fallback if DB unavailable
        items = []
        if not STORYLINES_DIR.exists():
            return items
        for path in sorted(STORYLINES_DIR.glob("storyline_*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                story = payload.get("storyline", {})
                items.append({
                    "id": path.stem, "filename": path.name, "path": str(path),
                    "title": story.get("title"), "logline": story.get("logline"),
                    "sections": len(story.get("sections", [])),
                    "created_at": payload.get("created_at"),
                    "target_duration_seconds": payload.get("target_duration_seconds"),
                })
            except Exception:
                items.append({"id": path.stem, "filename": path.name, "path": str(path)})
        return items


@router.get("/latest")
def get_latest_storyline():
    try:
        row = StorylineRepository().get_latest()
        if row:
            return _row_to_full(dict(row))
    except Exception:
        pass
    # Filesystem fallback
    path = latest_storyline_path()
    if not path:
        raise HTTPException(status_code=404, detail="No storyline found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["storyline_id"] = path.stem
    payload["storyline_path"] = str(path)
    return payload


@router.get("/{storyline_id}")
def get_storyline(storyline_id: str):
    try:
        row = StorylineRepository().get(storyline_id)
        if row:
            return _row_to_full(dict(row))
    except Exception:
        pass
    # Filesystem fallback
    path = STORYLINES_DIR / f"{storyline_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Storyline not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["storyline_id"] = path.stem
    payload["storyline_path"] = str(path)
    return payload


@router.get("/{storyline_id}/evidence-map")
def get_storyline_evidence_map(storyline_id: str):
    from app.services.insight_service import storyline_evidence_map

    # Try DB first, then filesystem
    storyline = None
    try:
        row = StorylineRepository().get(storyline_id)
        if row:
            storyline = dict(row.get("full_payload") or row)
    except Exception:
        pass

    if storyline is None:
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
