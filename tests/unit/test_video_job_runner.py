"""Unit tests for app.services.video_job_runner — progress computation,
stage helpers, and job creation."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.services.video_job_runner import (
    STAGE_WEIGHTS,
    STAGE_DEFINITIONS,
    compute_global_progress,
)


# ---------------------------------------------------------------------------
# Stage weight invariant
# ---------------------------------------------------------------------------

class TestStageWeights:
    def test_weights_sum_to_100(self):
        assert sum(STAGE_WEIGHTS.values()) == 100

    def test_all_defined_stages_have_weights(self):
        defined = {s["stage"] for s in STAGE_DEFINITIONS}
        assert set(STAGE_WEIGHTS.keys()) == defined

    def test_all_weights_positive(self):
        assert all(w > 0 for w in STAGE_WEIGHTS.values())


# ---------------------------------------------------------------------------
# compute_global_progress
# ---------------------------------------------------------------------------

class TestComputeGlobalProgress:
    def test_all_pending_is_zero(self):
        stage_status = {s: {"status": "pending", "progress_percent": 0} for s in STAGE_WEIGHTS}
        assert compute_global_progress(stage_status) == 0

    def test_all_completed_is_100(self):
        stage_status = {s: {"status": "completed", "progress_percent": 100} for s in STAGE_WEIGHTS}
        assert compute_global_progress(stage_status) == 100

    def test_empty_stage_status_is_zero(self):
        assert compute_global_progress({}) == 0

    def test_one_stage_completed(self):
        stage_status = {"input_validation": {"status": "completed", "progress_percent": 100}}
        weight = STAGE_WEIGHTS["input_validation"]
        assert compute_global_progress(stage_status) == weight

    def test_running_stage_contributes_proportionally(self):
        stage_status = {
            "input_validation": {"status": "completed", "progress_percent": 100},
            "07_storyline": {"status": "running", "progress_percent": 50},
        }
        expected = (
            STAGE_WEIGHTS["input_validation"]
            + STAGE_WEIGHTS["07_storyline"] * 50 // 100
        )
        assert compute_global_progress(stage_status) == expected

    def test_running_at_zero_contributes_nothing(self):
        stage_status = {"07_storyline": {"status": "running", "progress_percent": 0}}
        assert compute_global_progress(stage_status) == 0

    def test_running_at_100_gives_full_weight(self):
        stage_status = {"07_storyline": {"status": "running", "progress_percent": 100}}
        assert compute_global_progress(stage_status) == STAGE_WEIGHTS["07_storyline"]

    def test_failed_stage_contributes_zero(self):
        stage_status = {"07_storyline": {"status": "failed", "progress_percent": 100}}
        assert compute_global_progress(stage_status) == 0

    def test_capped_at_100(self):
        # Force values that would overshoot
        stage_status = {s: {"status": "completed", "progress_percent": 100} for s in STAGE_WEIGHTS}
        assert compute_global_progress(stage_status) <= 100

    def test_missing_status_treated_as_pending(self):
        stage_status = {"07_storyline": {"progress_percent": 50}}  # no 'status' key
        assert compute_global_progress(stage_status) == 0

    def test_multiple_stages_completed_and_one_running(self):
        stage_status = {
            "input_validation": {"status": "completed", "progress_percent": 100},
            "07_storyline": {"status": "completed", "progress_percent": 100},
            "07b_validate_storyline": {"status": "running", "progress_percent": 40},
        }
        expected = (
            STAGE_WEIGHTS["input_validation"]
            + STAGE_WEIGHTS["07_storyline"]
            + STAGE_WEIGHTS["07b_validate_storyline"] * 40 // 100
        )
        assert compute_global_progress(stage_status) == expected


# ---------------------------------------------------------------------------
# create_video_job
# ---------------------------------------------------------------------------

class TestCreateVideoJob:
    def test_returns_job_id_string(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test prompt", "target_duration": 150})

        assert isinstance(job_id, str)
        assert job_id.startswith("vjob_")

    def test_job_id_has_uuid_suffix(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        parts = job_id.split("_")
        # Format: vjob_YYYYMMDD_HHMMSS_<8hex>
        assert len(parts) >= 4
        assert len(parts[-1]) == 8  # 8-char hex suffix

    def test_job_saved_with_correct_initial_state(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "AI ethics", "target_duration": 150})

        job = mock_job_store["jobs"][job_id]
        assert job["status"] == "queued"
        assert job["progress_percent"] == 0
        assert job["current_stage"] == "queued"
        assert job["prompt"] == "AI ethics"
        assert job["error"] is None

    def test_job_has_all_path_slots(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        paths = mock_job_store["jobs"][job_id]["paths"]
        for key in ("storyline_path", "validated_storyline_path", "asset_manifest_path",
                    "export_manifest_path", "video_path", "srt_path", "vtt_path"):
            assert key in paths
            assert paths[key] is None

    def test_created_event_emitted(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        events = mock_job_store["events"].get(job_id, [])
        assert any(e["event_type"] == "created" for e in events)

    def test_user_id_stored(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test", "target_duration": 150}, user_id="user_abc")

        assert mock_job_store["jobs"][job_id]["user_id"] == "user_abc"

    def test_target_duration_required(self, mock_job_store, tmp_path):
        # target_duration has no default here — it's the API layer's
        # `VideoJobRequest.target_duration = Field(..., ge=30, le=600)`
        # (backend/app/api/video_jobs.py) that guarantees callers of
        # create_video_job() always supply one.
        from app.services.video_job_runner import create_video_job

        with pytest.raises(KeyError):
            create_video_job({"prompt": "test"})

    def test_target_duration_from_request(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job

        job_id = create_video_job({"prompt": "test", "target_duration": 300})

        assert mock_job_store["jobs"][job_id]["target_duration"] == 300


# ---------------------------------------------------------------------------
# update_stage helper
# ---------------------------------------------------------------------------

class TestUpdateStage:
    def test_stage_status_written(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job, update_stage

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        update_stage(job_id, "07_storyline", "running", 30, "Generating storyline")
        job = mock_job_store["jobs"][job_id]
        stage = job["stage_status"]["07_storyline"]
        assert stage["status"] == "running"
        assert stage["progress_percent"] == 30
        assert stage["summary"] == "Generating storyline"

    def test_global_progress_updated(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job, update_stage

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        update_stage(job_id, "input_validation", "completed", 100, "Done")
        job = mock_job_store["jobs"][job_id]
        assert job["progress_percent"] == STAGE_WEIGHTS["input_validation"]

    def test_progress_event_emitted(self, mock_job_store, tmp_path):
        from app.services.video_job_runner import create_video_job, update_stage

        job_id = create_video_job({"prompt": "test", "target_duration": 150})

        update_stage(job_id, "07_storyline", "running", 50, "halfway")
        events = mock_job_store["events"][job_id]
        progress_events = [e for e in events if e["event_type"] == "progress"]
        assert len(progress_events) >= 1


# ---------------------------------------------------------------------------
# Code version stamps
# ---------------------------------------------------------------------------

class TestCodeVersionStamps:
    def test_api_version_stamped_at_creation(self, mock_job_store, monkeypatch):
        from app.services import video_job_runner as runner

        monkeypatch.setattr(runner, "code_version", lambda: "git:api123")
        job_id = runner.create_video_job({"prompt": "AI ethics", "target_duration": 150})

        assert mock_job_store["jobs"][job_id]["api_code_version"] == "git:api123"

    def test_worker_version_stamped_when_run_starts(self, mock_job_store, monkeypatch):
        from app.services import video_job_runner as runner

        job_id = runner.create_video_job({"prompt": "AI ethics", "target_duration": 150})
        monkeypatch.setattr(runner, "code_version", lambda: "tree:worker456")

        def _stop(*args, **kwargs):
            raise RuntimeError("stop after the stamp")

        monkeypatch.setattr(runner.model_gate, "ensure_ready", _stop)
        runner.run_video_job(job_id, {"prompt": "AI ethics", "target_duration": 150})

        assert mock_job_store["jobs"][job_id]["code_version"] == "tree:worker456"
