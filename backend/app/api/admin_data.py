"""Read-only admin data explorer.

Lets an admin browse/inspect rows from an allow-listed set of tables (logs,
generated content, provenance records) without a dedicated endpoint per
table. Deliberately GET-only: no create/update/delete, and no dynamic
schema/DDL access — see the table registry below for what's excluded and
why. Every table here already has a repository backing it; this router
just gives it a uniform, paginated, searchable HTTP shape instead of
writing N near-identical raw-browse endpoints by hand.

Excluded on purpose (never add these): app_users (password hashes),
user_sessions / refresh_tokens (session/token material), otp_codes (OTP
secrets), auth_audit (no read path exists even internally, by design).
Those already have their own curated, redacted endpoints where an admin
genuinely needs them (see admin_users.py).

Search is applied in Python over a scanned batch of rows, not pushed down
as SQL WHERE — repositories don't expose arbitrary column filtering, and
building that per table would mean allow-listing filterable columns per
table to stay injection-safe. At this app's table sizes that's not worth
it: a case-insensitive substring match over up to _SCAN_LIMIT rows is
simple, safe by construction (never touches SQL), and fast enough for an
admin debugging tool. It is NOT a full-table search past that cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from app.db.repositories.decision_record_repository import DecisionRecordRepository
from app.db.repositories.event_repository import EventRepository
from app.db.repositories.fragment_repository import FragmentRepository
from app.db.repositories.ingest_job_repository import IngestJobRepository
from app.db.repositories.insight_repository import InsightRepository
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.llm_payload_repository import LLMPayloadRepository
from app.db.repositories.llm_trace_repository import LLMTraceRepository
from app.db.repositories.prompt_resolution_repository import PromptResolutionRepository
from app.db.repositories.storyline_repository import StorylineRepository
from app.db.repositories.video_repository import VideoRepository

router = APIRouter(prefix="/admin/data", tags=["Admin Data Explorer"])

# Max rows pulled from a table in one request before filtering/pagination —
# a safety cap, not a real page size. Kept small when no search is active
# (just enough to cover the requested page); widened to this cap only when
# a search is actually running, since that has to scan before it can filter.
_SCAN_LIMIT = 2000


@dataclass(frozen=True)
class _TableSpec:
    fetch_fn: Callable[[int], list[dict[str, Any]]]
    get_fn: Callable[[str], dict[str, Any] | None] | None = None


_TABLES: dict[str, _TableSpec] = {
    "fragments": _TableSpec(
        fetch_fn=lambda cap: FragmentRepository().list_all(limit=cap),
        get_fn=lambda id_: FragmentRepository().get(id_),
    ),
    "videos": _TableSpec(
        fetch_fn=lambda cap: VideoRepository().list_all()[:cap],
        get_fn=lambda id_: VideoRepository().get(id_),
    ),
    "storylines": _TableSpec(
        fetch_fn=lambda cap: StorylineRepository().list(limit=cap),
        get_fn=lambda id_: StorylineRepository().get(id_),
    ),
    "insights": _TableSpec(
        fetch_fn=lambda cap: InsightRepository().list(limit=cap),
        get_fn=lambda id_: InsightRepository().get(id_),
    ),
    "jobs": _TableSpec(
        fetch_fn=lambda cap: JobRepository().list_jobs()[:cap],
        get_fn=lambda id_: JobRepository().get_job(id_),
    ),
    "ingest_jobs": _TableSpec(
        fetch_fn=lambda cap: IngestJobRepository().list_jobs(limit=cap),
        get_fn=lambda id_: IngestJobRepository().get_job(id_),
    ),
    "decision_records": _TableSpec(
        fetch_fn=lambda cap: DecisionRecordRepository().list_all(limit=cap, offset=0),
        get_fn=lambda id_: DecisionRecordRepository().get(id_),
    ),
    "llm_call_traces": _TableSpec(
        fetch_fn=lambda cap: LLMTraceRepository().list_all(limit=cap, offset=0),
        # No single-row getter exists on this repository — list-only.
        get_fn=None,
    ),
}


def _spec(table: str) -> _TableSpec:
    spec = _TABLES.get(table)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown or unexposed table: {table!r}")
    return spec


def _apply_search(rows: list[dict[str, Any]], field: str, value: str) -> list[dict[str, Any]]:
    needle = value.lower()
    return [row for row in rows if needle in str(row.get(field, "")).lower()]


@router.get("/tables")
def list_tables() -> dict[str, Any]:
    return {"tables": sorted(_TABLES.keys())}


@router.get("/tables/{table}")
def list_rows(
    table: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search_field: str | None = Query(default=None, description="Column to search"),
    search_value: str | None = Query(default=None, description="Case-insensitive substring match"),
) -> dict[str, Any]:
    if bool(search_field) != bool(search_value):
        raise HTTPException(status_code=400, detail="search_field and search_value must be provided together")

    spec = _spec(table)
    searching = bool(search_field and search_value)
    rows = spec.fetch_fn(_SCAN_LIMIT if searching else offset + limit)
    if searching:
        rows = _apply_search(rows, search_field, search_value)

    page = rows[offset:offset + limit]
    return {
        "table": table, "limit": limit, "offset": offset,
        "matched": len(rows), "count": len(page), "rows": page,
    }


@router.get("/tables/{table}/{row_id}")
def get_row(table: str, row_id: str) -> dict[str, Any]:
    spec = _spec(table)
    if spec.get_fn is None:
        raise HTTPException(status_code=400, detail=f"Single-row lookup not supported for table {table!r}")
    row = spec.get_fn(row_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No row {row_id!r} in table {table!r}")
    return row


@router.get("/jobs/{job_id}/full")
def get_job_full(job_id: str) -> dict[str, Any]:
    """Everything about one video job in a single call — the job row, its
    events, decision record, LLM call summaries (latency/model/stage), the
    prompt_resolutions summary row, and the full request/response text of
    every LLM call — instead of six separate lookups every time someone
    debugs a failed job or wants to see exactly what a model saw and said."""
    job = JobRepository().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
    return {
        "job": job,
        "events": EventRepository().get_events(job_id),
        "decision_record": DecisionRecordRepository().get_by_job(job_id),
        "llm_traces": LLMTraceRepository().list_by_job(job_id),
        "prompt_resolution": PromptResolutionRepository().get_by_job(job_id),
        "llm_payloads": LLMPayloadRepository().list_by_job(job_id),
    }
