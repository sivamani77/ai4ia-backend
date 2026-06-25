"""Unit tests for app.pipeline.base — StageResult, PipelineContext, CONTRACTS,
run_script, and kill_running_stage."""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.base import (
    CONTRACTS,
    PipelineContext,
    StageResult,
    kill_running_stage,
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
    def test_successful_script(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "test_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                code, out, err = run_script("test_job", "mystage", ["echo", "hello"])

        assert code == 0
        assert "hello" in out

    def test_failing_script_returns_nonzero(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "fail_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                code, out, err = run_script("fail_job", "mystage", ["false"])

        assert code != 0

    def test_stdout_log_written(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "log_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                run_script("log_job", "mystage", ["echo", "hello world"])

        log_file = logs_dir / "mystage.stdout.log"
        assert log_file.exists()
        assert "hello world" in log_file.read_text()

    def test_stderr_log_written(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "err_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                run_script("err_job", "mystage",
                           ["sh", "-c", "echo errline >&2"])

        log_file = logs_dir / "mystage.stderr.log"
        assert log_file.exists()
        assert "errline" in log_file.read_text()

    def test_pid_file_cleaned_up_after_success(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "pid_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                run_script("pid_job", "mystage", ["true"])

        pid_file = logs_dir / "mystage.pid"
        assert not pid_file.exists()

    def test_returns_tuple_of_three(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "t3_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                result = run_script("t3_job", "mystage", ["true"])

        assert len(result) == 3
        code, out, err = result
        assert isinstance(code, int)
        assert isinstance(out, str)
        assert isinstance(err, str)

    def test_exception_returns_minus_one(self, tmp_path):
        logs_dir = tmp_path / "jobs" / "exc_job" / "logs"
        logs_dir.mkdir(parents=True)

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("app.pipeline.base.PROJECT_ROOT", tmp_path):
                # Command that doesn't exist
                code, out, err = run_script("exc_job", "mystage",
                                            ["/nonexistent/binary/xyz"])

        assert code == -1
        assert err != ""


# ---------------------------------------------------------------------------
# kill_running_stage
# ---------------------------------------------------------------------------

class TestKillRunningStage:
    def test_no_pid_file_returns_false(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            result = kill_running_stage("job1", "mystage")
        assert result is False

    def test_valid_pid_sends_sigterm(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        pid_file = logs_dir / "mystage.pid"
        pid_file.write_text("99999")  # use an unlikely-to-exist PID

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("os.kill") as mock_kill:
                result = kill_running_stage("job1", "mystage")

        mock_kill.assert_called_once_with(99999, signal.SIGTERM)
        assert result is True

    def test_process_not_found_returns_false(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        pid_file = logs_dir / "mystage.pid"
        pid_file.write_text("99999")

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("os.kill", side_effect=ProcessLookupError):
                result = kill_running_stage("job1", "mystage")

        assert result is False

    def test_corrupt_pid_file_returns_false(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        pid_file = logs_dir / "mystage.pid"
        pid_file.write_text("not-a-number")

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            result = kill_running_stage("job1", "mystage")

        assert result is False

    def test_pid_file_deleted_after_kill(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        pid_file = logs_dir / "mystage.pid"
        pid_file.write_text("99999")

        with patch("app.pipeline.base._logs_dir", return_value=logs_dir):
            with patch("os.kill"):
                kill_running_stage("job1", "mystage")

        assert not pid_file.exists()
