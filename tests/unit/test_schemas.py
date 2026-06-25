"""Unit tests for app.schemas.pipeline.JobResponse."""
from __future__ import annotations

import pytest

from app.schemas.pipeline import JobResponse


class TestJobResponseSchema:
    def test_minimal_required_fields(self):
        j = JobResponse(job_id="j1", status="queued")
        assert j.job_id == "j1"
        assert j.status == "queued"

    def test_progress_defaults_to_zero(self):
        j = JobResponse(job_id="j1", status="queued")
        assert j.progress_percent == 0

    def test_optional_fields_default_none(self):
        j = JobResponse(job_id="j1", status="queued")
        assert j.current_stage is None
        assert j.message is None
        assert j.created_at is None
        assert j.started_at is None
        assert j.completed_at is None
        assert j.error is None
        assert j.prompt is None

    def test_all_fields_set(self):
        j = JobResponse(
            job_id="j2",
            status="running",
            current_stage="07_storyline",
            progress_percent=42,
            message="Running storyline stage",
            created_at="2026-01-01T00:00:00Z",
            started_at="2026-01-01T00:01:00Z",
            completed_at=None,
            error=None,
            prompt="AI in healthcare",
        )
        assert j.progress_percent == 42
        assert j.current_stage == "07_storyline"
        assert j.prompt == "AI in healthcare"

    def test_extra_fields_are_allowed(self):
        """JobResponse has extra='allow' — unknown fields pass through."""
        j = JobResponse(
            job_id="j3",
            status="completed",
            my_custom_field="custom_value",
        )
        assert j.my_custom_field == "custom_value"  # type: ignore[attr-defined]

    def test_model_validate_from_dict(self):
        data = {
            "job_id": "j4",
            "status": "failed",
            "progress_percent": 55,
            "error": {"type": "pipeline_error", "message": "oops"},
        }
        j = JobResponse.model_validate(data)
        assert j.job_id == "j4"
        assert j.status == "failed"
        assert j.error["type"] == "pipeline_error"

    def test_model_dump_contains_job_id(self):
        j = JobResponse(job_id="j5", status="queued")
        d = j.model_dump()
        assert "job_id" in d
        assert d["job_id"] == "j5"

    def test_error_field_accepts_dict(self):
        j = JobResponse(
            job_id="j6",
            status="failed",
            error={"type": "pipeline_error", "stage": "08_assets", "message": "ffmpeg crashed"},
        )
        assert j.error["stage"] == "08_assets"

    def test_progress_percent_can_be_100(self):
        j = JobResponse(job_id="j7", status="completed", progress_percent=100)
        assert j.progress_percent == 100

    def test_multiple_extra_fields(self):
        j = JobResponse(
            job_id="j8",
            status="running",
            stage_status={"07_storyline": {"status": "completed"}},
            warnings=[],
        )
        assert j.stage_status["07_storyline"]["status"] == "completed"  # type: ignore[attr-defined]
        assert j.warnings == []  # type: ignore[attr-defined]
