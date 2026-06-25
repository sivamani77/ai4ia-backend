from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import comparison_service

router = APIRouter(tags=["Comparison & Relay"])


class CompareSpeakersRequest(BaseModel):
    speakers: list[str] = Field(..., min_length=2, max_length=10)
    topic: str | None = Field(None, max_length=300)
    limit_per_speaker: int = Field(6, ge=1, le=20)


class ThemeSynthesisRequest(BaseModel):
    theme: str = Field(..., min_length=2, max_length=200)
    limit: int = Field(20, ge=5, le=60)


class RelayRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    limit: int = Field(20, ge=5, le=60)


def _run(fn, *args):
    try:
        return fn(*args)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}")


@router.post("/compare/speakers")
def api_compare_speakers(req: CompareSpeakersRequest):
    return _run(comparison_service.compare_speakers, req.speakers, req.topic,
                req.limit_per_speaker)


@router.post("/synthesis/theme")
def api_synthesize_theme(req: ThemeSynthesisRequest):
    return _run(comparison_service.synthesize_theme, req.theme, req.limit)


@router.post("/relay")
def api_build_relay(req: RelayRequest):
    return _run(comparison_service.build_relay, req.topic, req.limit)
