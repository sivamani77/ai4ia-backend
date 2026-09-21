from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import require_models
from app.core.auth import CurrentUser, get_current_user
from app.core.config import INSIGHT_DEFAULT_LIMIT
from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket
from app.services import insight_service, model_gate
from app.services.guardrail_service import GuardrailRejection

router = APIRouter(prefix="/insights", tags=["Insights"])

# 10 generation requests per user per 60 seconds
_insight_limiter = TokenBucket(rate=10, per=60.0, name="insights_generate")


class GenerateInsightRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    filters: dict | None = None
    limit: int = Field(INSIGHT_DEFAULT_LIMIT, ge=5, le=100)


@router.post(
    "/generate",
    dependencies=[Depends(require_models(model_gate.EMBEDDING, model_gate.LLM))],
)
def api_generate_insight(req: GenerateInsightRequest, current_user: CurrentUser = Depends(get_current_user)):
    if not _insight_limiter.allow(current_user.id):
        _metrics.rate_limit_rejections_total.labels(limiter="insights_generate").inc()
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before trying again.",
        )
    try:
        return insight_service.generate_insight(req.query, req.filters, req.limit, user_id=current_user.id)
    except GuardrailRejection as exc:
        # Ahead of the blanket handler below, which would report a policy
        # rejection as a 502 generation failure.
        raise HTTPException(status_code=400, detail=exc.message)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Insight generation failed: {exc}")


@router.get("")
def api_list_insights(current_user: CurrentUser = Depends(get_current_user)):
    items = insight_service.list_insights()
    return {"count": len(items), "insights": items}


@router.get("/{insight_id}")
def api_get_insight(insight_id: str, current_user: CurrentUser = Depends(get_current_user)):
    insight = insight_service.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return insight


@router.get("/{insight_id}/sources")
def api_insight_sources(insight_id: str, current_user: CurrentUser = Depends(get_current_user)):
    insight = insight_service.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return insight_service.build_sources_map(insight)


@router.get("/{insight_id}/provenance")
def api_insight_provenance(insight_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Full two-layer provenance for an insight generation run.

    Layer 1 — decision_record: osi-1.0 record (agents, models, evidence coverage).
    Layer 2 — llm_call_traces: per-call rows for the retrieval embedding + LLM synthesis.
    """
    insight = insight_service.get_insight(insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")

    # ── Layer 1: Decision record ──────────────────────────────────────────────
    decision_record = None
    try:
        from app.db.repositories.decision_record_repository import DecisionRecordRepository
        row = DecisionRecordRepository().get_by_insight(insight_id)
        decision_record = dict(row) if row else None
    except Exception:
        pass

    # ── Layer 2: LLM call traces ─────────────────────────────────────────────
    llm_traces = []
    try:
        from app.db.repositories.llm_trace_repository import LLMTraceRepository
        llm_traces = [dict(r) for r in LLMTraceRepository().list_by_insight(insight_id)]
    except Exception:
        pass

    # ── Build agents from available sources ───────────────────────────────────
    # _explain is only in memory at generation time (not a Supabase column).
    # Fall back to: models stored on the insight row + decision_record decisions.
    pool = insight.get("evidence_pool") or []
    models = insight.get("models") or {}
    dr_decisions = (decision_record or {}).get("decisions") or {}
    dr_models = (decision_record or {}).get("models_used") or {}

    supported = len(insight.get("patterns") or []) + len(insight.get("contrasts") or [])
    unsupported = len(insight.get("unsupported_claims") or [])
    total_claims = dr_decisions.get("total_claims_generated") or (supported + unsupported) or None

    return {
        "insight_id": insight_id,
        "provenance_version": "2.0",
        "decision_record": decision_record,
        "agents": {
            "retrieval": {
                "rule_based": False,
                "method": "semantic_cosine_similarity",
                "embedding_model": dr_models.get("embedding") or models.get("embedding"),
                "fragments_retrieved": dr_decisions.get("evidence_pool_size") or len(pool),
                "filters_applied": insight.get("filters") or {},
            },
            "insight": {
                "rule_based": False,
                "model": dr_models.get("insight") or models.get("llm"),
                "temperature": 0.2,
                "citation_contract": "LLM may only cite numbered fragments from the evidence pool",
                "total_claims_generated": total_claims,
                "supported_claims": dr_decisions.get("supported_claims") or supported or None,
                "unsupported_claims_dropped": dr_decisions.get("unsupported_claims_dropped", unsupported),
            },
        },
        "evidence_coverage": insight.get("evidence_coverage"),
        "llm_call_traces": llm_traces,
        "query": insight.get("query"),
        "models": models,
        "created_at": insight.get("created_at"),
    }
