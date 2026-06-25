from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    current_stage: str | None = None
    progress_percent: int = 0
    message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: dict | None = None
    prompt: str | None = None
