"""Unit tests for app.pipeline.base — StageResult, PipelineContext, CONTRACTS,
and run_script."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.pipeline.base import (
    CONTRACTS,
    PipelineContext,
    StageResult,
    run_script,
)


# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------

class TestStageResult:
    def test_required_fields(self):
        r = StageResult("s1", "completed", "ok")
        assert r.stage == "s1"
        assert r.status == "completed"
        assert r.message == "ok"

    def test_data_defaults_to_empty_dict(self):
        r = StageResult("s1", "completed", "ok")
        assert r.data == {}

    def test_data_not_shared_between_instances(self):
        r1 = StageResult("s1", "completed", "ok")
        r2 = StageResult("s2", "failed", "bad")
        r1.data["key"] = "val"
        assert "key" not in r2.data

    def test_failed_status(self):
        r = StageResult("s1", "failed", "something broke")
        assert r.status == "failed"

    def test_skipped_status(self):
        r = StageResult("s1", "skipped", "optional stage")
        assert r.status == "skipped"

    def test_custom_data(self):
        r = StageResult("s1", "completed", "ok", data={"path": "/tmp/out.json"})
        assert r.data["path"] == "/tmp/out.json"


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_required_fields(self):
        ctx = PipelineContext("job123", {"prompt": "test"})
        assert ctx.job_id == "job123"
        assert ctx.request == {"prompt": "test"}

    def test_all_paths_none_by_default(self):
        ctx = PipelineContext("job1", {})
        assert ctx.storyline_path is None
        assert ctx.validated_storyline_path is None
        assert ctx.asset_manifest_path is None
        assert ctx.export_manifest_path is None
        assert ctx.video_path is None
        assert ctx.srt_path is None
        assert ctx.vtt_path is None

    def test_paths_can_be_set(self):
        ctx = PipelineContext("job1", {})
        ctx.storyline_path = "/data/storyline.json"
        ctx.video_path = "/data/out.mp4"
        assert ctx.storyline_path == "/data/storyline.json"
        assert ctx.video_path == "/data/out.mp4"

    def test_empty_request(self):
        ctx = PipelineContext("job1", {})
        assert ctx.request == {}


# ---------------------------------------------------------------------------
# CONTRACTS
# ---------------------------------------------------------------------------

class TestContracts:
    EXPECTED_STAGES = {
        "input_validation",
        "07_storyline",
        "07b_validate_storyline",
        "08_assets",
        "09_export",
        "10_validate_export",
        "11_subtitles",
    }

    def test_all_expected_stages_present(self):
        assert set(CONTRACTS.keys()) == self.EXPECTED_STAGES

    def test_each_contract_has_stage_name(self):
        for key, contract in CONTRACTS.items():
            assert contract.stage == key

    def test_each_contract_has_script_name(self):
        for key, contract in CONTRACTS.items():
            assert isinstance(contract.script_name, str)
            assert contract.script_name

    def test_each_contract_has_required_args(self):
        for key, contract in CONTRACTS.items():
            assert isinstance(contract.required_args, list)

    def test_each_contract_has_output_description(self):
        for key, contract in CONTRACTS.items():
            assert isinstance(contract.output_description, str)
            assert contract.output_description

    def test_07_storyline_requires_prompt(self):
        assert "--prompt" in CONTRACTS["07_storyline"].required_args

    def test_09_export_requires_manifest_and_output(self):
        args = CONTRACTS["09_export"].required_args
        assert "--manifest" in args
        assert "--output" in args

    def test_11_subtitles_requires_video(self):
        assert "--video" in CONTRACTS["11_subtitles"].required_args


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------

class TestRunScript:
    """run_script no longer writes log files to a per-job logs dir or a pid
    file on disk. It stores the subprocess pid on the job dict via
    job_store.load_job/save_job (so cancellation can find it) and reports
    stdout/stderr via job_store.append_event, all imported lazily inside the
    function body — so tests patch the functions on app.services.job_store
    directly rather than names on app.pipeline.base."""

    def _patches(self, job=None):
        job = job if job is not None else {}
        return (
            patch("app.services.job_store.load_job", side_effect=lambda jid: dict(job)),
            patch("app.services.job_store.save_job", MagicMock()),
            patch("app.services.job_store.append_event", MagicMock()),
        )

    def test_successful_script(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event:
            code, out, err = run_script("test_job", "mystage", ["echo", "hello"])

        assert code == 0
        assert "hello" in out

    def test_failing_script_returns_nonzero(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event:
            code, out, err = run_script("fail_job", "mystage", ["false"])

        assert code != 0

    def test_stdout_captured(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event:
            code, out, err = run_script("log_job", "mystage", ["echo", "hello world"])

        assert "hello world" in out

    def test_stderr_captured(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event:
            code, out, err = run_script(
                "err_job", "mystage", ["sh", "-c", "echo errline >&2"]
            )

        assert "errline" in err

    def test_active_pid_stored_then_cleared(self):
        job = {"id": "pid_job"}
        p_load, p_save, p_event = self._patches(job)
        with p_load, p_save as mock_save, p_event:
            run_script("pid_job", "mystage", ["true"])

        # First save sets active_pid, second (in the finally block) clears it.
        assert mock_save.call_count == 2
        first_call_job = mock_save.call_args_list[0].args[1]
        second_call_job = mock_save.call_args_list[1].args[1]
        assert "active_pid" in first_call_job
        assert "active_pid" not in second_call_job

    def test_returns_tuple_of_three(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event:
            result = run_script("t3_job", "mystage", ["true"])

        assert len(result) == 3
        code, out, err = result
        assert isinstance(code, int)
        assert isinstance(out, str)
        assert isinstance(err, str)

    def test_exception_returns_minus_one(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event as mock_event:
            # Command that doesn't exist
            code, out, err = run_script(
                "exc_job", "mystage", ["/nonexistent/binary/xyz"]
            )

        assert code == -1
        assert err != ""
        assert any(c.args[1] == "script_error" for c in mock_event.call_args_list)

    def test_append_event_called_on_success(self):
        p_load, p_save, p_event = self._patches()
        with p_load, p_save, p_event as mock_event:
            run_script("evt_job", "mystage", ["echo", "hi"])

        assert any(c.args[1] == "script_output" for c in mock_event.call_args_list)
