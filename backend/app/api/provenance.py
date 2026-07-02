"""Admin provenance API — unified feed and detail endpoints for all job types."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("ai4ia.api.provenance")

router = APIRouter(prefix="/provenance", tags=["provenance"])

# Maps subject stored in agent_decision_records → human job_type label
_SUBJECT_TO_JOB_TYPE = {
    "documentary_generation": "video_job",
    "insight_generation":     "insight_job",
    "video_ingestion":        "ingest_job",
}


# ---------------------------------------------------------------------------
# Unified provenance feed  (newest first, all job types)
# ---------------------------------------------------------------------------

@router.get("/feed")
def provenance_feed(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    job_type: str | None = Query(
        default=None,
        description="Filter by job type: video_job | insight_job | ingest_job",
    ),
) -> dict[str, Any]:
    """
    Unified provenance feed — all video jobs, insight jobs, and ingest jobs
    combined and ordered newest-first.

    Each item includes the full decision record (Layer 1) and a summary of
    LLM call traces (Layer 2): total calls, agents used, models, error count.

    Use `job_type` to filter to a single job type.
    Use `limit` / `offset` for pagination.
    """
    try:
        from app.db.repositories.decision_record_repository import DecisionRecordRepository
        from app.db.repositories.llm_trace_repository import LLMTraceRepository

        # Pull enough records to serve this page after optional filtering.
        # We over-fetch then filter in Python because Supabase postgrest doesn't
        # support OR across enum values easily without RPC.
        fetch_limit = min(limit * 5, 500)  # at most 500 rows from DB
        records = DecisionRecordRepository().list_all(limit=fetch_limit, offset=0)

        # Apply job_type filter if given
        if job_type:
            target_subjects = {
                subj for subj, jt in _SUBJECT_TO_JOB_TYPE.items() if jt == job_type
            }
            records = [r for r in records if r.get("subject") in target_subjects]

        # Paginate after filtering
        total_filtered = len(records)
        page = records[offset: offset + limit]

        # Fetch traces for the records on this page only (avoid N+1)
        # Group by job_id and insight_id
        job_ids    = {r["job_id"]     for r in page if r.get("job_id")}
        insight_ids = {r["insight_id"] for r in page if r.get("insight_id")}

        traces_by_job: dict[str, list] = defaultdict(list)
        traces_by_insight: dict[str, list] = defaultdict(list)

        if job_ids or insight_ids:
            all_traces = LLMTraceRepository().list_all(limit=500, offset=0)
            for t in all_traces:
                if t.get("job_id") and t["job_id"] in job_ids:
                    traces_by_job[t["job_id"]].append(t)
                if t.get("insight_id") and t["insight_id"] in insight_ids:
                    traces_by_insight[t["insight_id"]].append(t)

    except Exception:
        logger.exception("provenance feed failed")
        raise HTTPException(status_code=500, detail="Failed to build provenance feed")

    items = []
    for r in page:
        job_id     = r.get("job_id")
        insight_id = r.get("insight_id")
        subject    = r.get("subject") or "unknown"
        jtype      = _SUBJECT_TO_JOB_TYPE.get(subject, subject)

        traces = (
            traces_by_job.get(job_id, [])
            if job_id
            else traces_by_insight.get(insight_id, [])
        )
        trace_errors  = sum(1 for t in traces if t.get("error"))
        agents_used   = list({t.get("agent") for t in traces if t.get("agent")})
        models_used_t = list({t.get("model") for t in traces if t.get("model")})

        items.append({
            # ── identity ──────────────────────────────────────────────────────
            "id":          r.get("id"),
            "job_type":    jtype,
            "job_id":      job_id,
            "insight_id":  insight_id,
            "user_id":     r.get("user_id"),
            "created_at":  r.get("created_at"),
            # ── outcome ───────────────────────────────────────────────────────
            "subject":     subject,
            "outcome":     r.get("outcome"),
            "total_duration_ms": r.get("total_duration_ms"),
            # ── layer 1: decision record ──────────────────────────────────────
            "decision_record": {
                "schema_version":   "osi-1.0",
                "agents_executed":  r.get("agents_executed") or [],
                "decisions":        r.get("decisions") or {},
                "models_used":      r.get("models_used") or {},
            },
            # ── layer 2: llm traces summary ───────────────────────────────────
            "llm_traces_summary": {
                "total_calls":  len(traces),
                "error_count":  trace_errors,
                "agents_used":  agents_used,
                "models_used":  models_used_t,
            },
        })

    return {
        "total_filtered": total_filtered,
        "total_returned": len(items),
        "limit":   limit,
        "offset":  offset,
        "job_type_filter": job_type,
        "items":   items,
    }

