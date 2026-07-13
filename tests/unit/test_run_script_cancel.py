"""Unit tests for mid-stage cancellation in app.pipeline.base.run_script.

These verify the fix for the "cancel does not stop a running render" bug: while
a stage subprocess runs, run_script polls the job's persisted status and, when it
flips to "cancelled", kills the subprocess group and raises JobCancelledException
so the pipeline runner can finalize the job as cancelled (not failed).
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.pipeline.base import JobCancelledException, run_script


def _patches(job_state):
    """Patch job_store functions used by run_script.

    load_job returns the shared job_state dict, so a test can flip
    job_state["status"] = "cancelled" to simulate the API's DELETE handler.
    """
    return (
        patch("app.services.job_store.load_job", side_effect=lambda _id: dict(job_state)),
        patch("app.services.job_store.save_job", lambda _id, data: None),
        patch("app.services.job_store.append_event", lambda *a, **k: None),
    )


class TestRunScriptCancel:
    def test_cancel_mid_stage_raises_and_kills(self, tmp_path):
        # A subprocess that would run for 60s if never killed.
        job_state = {"job_id": "cancel_job", "status": "running"}
        p_load, p_save, p_evt = _patches(job_state)

        # Flip to cancelled shortly after start, from another thread.
        import threading

        def _flip():
            time.sleep(1.0)
            job_state["status"] = "cancelled"

        with patch("app.pipeline.base.PROJECT_ROOT", tmp_path), p_load, p_save, p_evt:
            threading.Thread(target=_flip, daemon=True).start()
            start = time.monotonic()
            with pytest.raises(JobCancelledException):
                run_script("cancel_job", "mystage", ["sleep", "60"])
            elapsed = time.monotonic() - start

        # Should return well before the 60s sleep — cancellation kicked in.
        assert elapsed < 30, f"cancellation took too long: {elapsed:.1f}s"

    def test_no_cancel_completes_normally(self, tmp_path):
        job_state = {"job_id": "ok_job", "status": "running"}
        p_load, p_save, p_evt = _patches(job_state)

        with patch("app.pipeline.base.PROJECT_ROOT", tmp_path), p_load, p_save, p_evt:
            code, out, err = run_script("ok_job", "mystage", ["echo", "done"])

        assert code == 0
        assert "done" in out

    def test_cancel_exception_carries_context(self, tmp_path):
        job_state = {"job_id": "ctx_job", "status": "cancelled"}  # already cancelling
        p_load, p_save, p_evt = _patches(job_state)

        with patch("app.pipeline.base.PROJECT_ROOT", tmp_path), p_load, p_save, p_evt:
            with pytest.raises(JobCancelledException) as ei:
                run_script("ctx_job", "07_storyline", ["sleep", "30"])

        assert ei.value.job_id == "ctx_job"
        assert ei.value.stage_name == "07_storyline"


class TestTerminalGuard:
    """run_video_job must skip re-processing a redelivered task whose job is
    already terminal (the Celery acks_late redelivery loop)."""

    @pytest.mark.parametrize("status", ["cancelled", "completed", "failed"])
    def test_terminal_job_skips_pipeline(self, status):
        from app.services import video_job_runner as vjr

        calls = {"save": 0}

        def _fake_save(_id, _data):
            calls["save"] += 1

        with patch.object(vjr, "load_job", return_value={"job_id": "j", "status": status}), \
             patch.object(vjr, "save_job", side_effect=_fake_save):
            vjr.run_video_job("j", {"prompt": "x", "target_duration": 30})

        # Guard hit: it returned before flipping status to running / saving.
        assert calls["save"] == 0

    def test_queued_job_proceeds_past_guard(self):
        """A non-terminal job must NOT be skipped — it proceeds (and then fails
        fast here because stages aren't stubbed, which is fine: we only assert
        the guard let it through by checking status was written to running)."""
        from app.services import video_job_runner as vjr

        saved = {}

        def _fake_save(_id, data):
            saved.update(data)

        with patch.object(vjr, "load_job", return_value={"job_id": "j", "status": "queued",
                                                          "paths": {}}), \
             patch.object(vjr, "save_job", side_effect=_fake_save), \
             patch.object(vjr, "append_event", lambda *a, **k: None), \
             patch.object(vjr, "_fail_job", lambda *a, **k: None), \
             patch.object(vjr, "_cleanup_job_tmp_dirs", lambda *a, **k: None):
            vjr.run_video_job("j", {"prompt": "x", "target_duration": 30})

        # Guard let it through: it started the pipeline (status flipped to running).
        assert saved.get("status") == "running"
