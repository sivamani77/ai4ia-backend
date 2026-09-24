"""Read-only admin data explorer.

Lets an admin browse/inspect rows from an allow-listed set of tables (logs,
generated content, provenance records) without a dedicated endpoint per
table. Deliberately GET-only: no create/update/delete, and no dynamic
schema/DDL access — see the table registry below for what's excluded and
why. Every table here already has a repository backing it; this router
just gives it a uniform, paginated HTTP shape instead of writing N
near-identical raw-browse endpoints by hand.

Excluded on purpose (never add these): app_users (password hashes),
user_sessions / refresh_tokens (session/token material), otp_codes (OTP
secrets), auth_audit (no read path exists even internally, by design).
Those already have their own curated, redacted endpoints where an admin
genuinely needs them (see admin_users.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from app.db.repositories.decision_record_repository import DecisionRecordRepository
from app.db.repositories.fragment_repository import FragmentRepository
from app.db.repositories.ingest_job_repository import IngestJobRepository
from app.db.repositories.insight_repository import InsightRepository
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.llm_trace_repository import LLMTraceRepository
from app.db.repositories.storyline_repository import StorylineRepository
from app.db.repositories.video_repository import VideoRepository

router = APIRouter(prefix="/admin/data", tags=["Admin Data Explorer"])


@dataclass(frozen=True)
class _TableSpec:
    list_fn: Callable[[int, int], list[dict[str, Any]]]
    get_fn: Callable[[str], dict[str, Any] | None] | None = None


def _paginate_unbounded(rows: list[dict[str, Any]], limit: int, offset: int) -> list[dict[str, Any]]:
    """For repositories whose list method has no limit/offset of its own —
    fetches everything and slices in Python. Fine at this app's table sizes
    for an admin debug tool; not meant for high-traffic use."""
    return rows[offset:offset + limit]


_TABLES: dict[str, _TableSpec] = {
    "fragments": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(
            FragmentRepository().list_all(limit=offset + limit), limit, offset,
        ),
        get_fn=lambda id_: FragmentRepository().get(id_),
    ),
    "videos": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(VideoRepository().list_all(), limit, offset),
        get_fn=lambda id_: VideoRepository().get(id_),
    ),
    "storylines": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(
            StorylineRepository().list(limit=offset + limit), limit, offset,
        ),
        get_fn=lambda id_: StorylineRepository().get(id_),
    ),
    "insights": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(
            InsightRepository().list(limit=offset + limit), limit, offset,
        ),
        get_fn=lambda id_: InsightRepository().get(id_),
    ),
    "jobs": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(JobRepository().list_jobs(), limit, offset),
        get_fn=lambda id_: JobRepository().get_job(id_),
    ),
    "ingest_jobs": _TableSpec(
        list_fn=lambda limit, offset: _paginate_unbounded(
            IngestJobRepository().list_jobs(limit=offset + limit), limit, offset,
        ),
        get_fn=lambda id_: IngestJobRepository().get_job(id_),
    ),
    "decision_records": _TableSpec(
        list_fn=lambda limit, offset: DecisionRecordRepository().list_all(limit=limit, offset=offset),
        get_fn=lambda id_: DecisionRecordRepository().get(id_),
    ),
    "llm_call_traces": _TableSpec(
        list_fn=lambda limit, offset: LLMTraceRepository().list_all(limit=limit, offset=offset),
        # No single-row getter exists on this repository — list-only.
        get_fn=None,
    ),
}


def _spec(table: str) -> _TableSpec:
    spec = _TABLES.get(table)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown or unexposed table: {table!r}")
    return spec


@router.get("/tables")
def list_tables() -> dict[str, Any]:
    return {"tables": sorted(_TABLES.keys())}


@router.get("/tables/{table}")
def list_rows(
    table: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    spec = _spec(table)
    rows = spec.list_fn(limit, offset)
    return {"table": table, "limit": limit, "offset": offset, "count": len(rows), "rows": rows}


@router.get("/tables/{table}/{row_id}")
def get_row(table: str, row_id: str) -> dict[str, Any]:
    spec = _spec(table)
    if spec.get_fn is None:
        raise HTTPException(status_code=400, detail=f"Single-row lookup not supported for table {table!r}")
    row = spec.get_fn(row_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No row {row_id!r} in table {table!r}")
    return row
