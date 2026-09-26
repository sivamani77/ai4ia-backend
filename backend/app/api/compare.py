import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import require_models
from app.core.auth import CurrentUser, get_current_user
from app.core.error_messages import DEFAULT_MESSAGE
from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket
from app.services import comparison_service, model_gate
from app.services.guardrail_service import GuardrailRejection

_log = logging.getLogger("ai4ia.api.compare")

router = APIRouter(tags=["Comparison & Relay"])

_model_gate_dep = [Depends(require_models(model_gate.EMBEDDING, model_gate.LLM))]

# 10 requests per user per 60 seconds, shared across the 3 routes below —
# they are comparable-cost LLM operations.
_compare_limiter = TokenBucket(rate=10, per=60.0, name="compare")


def _check_rate_limit(current_user: CurrentUser) -> None:
    if not _compare_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="compare").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before trying again.",
        )


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


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except GuardrailRejection as exc:
        # Ahead of the blanket handlers below, which would report a policy
        # rejection as a 502 generation failure.
        raise HTTPException(status_code=400, detail=exc.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:  # noqa: BLE001
        _log.exception("Generation failed")
        raise HTTPException(status_code=502, detail=DEFAULT_MESSAGE)


@router.post("/compare/speakers", dependencies=_model_gate_dep)
def api_compare_speakers(req: CompareSpeakersRequest, current_user: CurrentUser = Depends(get_current_user)):
    _check_rate_limit(current_user)
    return _run(comparison_service.compare_speakers, req.speakers, req.topic,
                req.limit_per_speaker, user_id=current_user.id)


@router.post("/synthesis/theme", dependencies=_model_gate_dep)
def api_synthesize_theme(req: ThemeSynthesisRequest, current_user: CurrentUser = Depends(get_current_user)):
    _check_rate_limit(current_user)
    return _run(comparison_service.synthesize_theme, req.theme, req.limit,
                user_id=current_user.id)


@router.post("/relay", dependencies=_model_gate_dep)
def api_build_relay(req: RelayRequest, current_user: CurrentUser = Depends(get_current_user)):
    _check_rate_limit(current_user)
    return _run(comparison_service.build_relay, req.topic, req.limit,
                user_id=current_user.id)
