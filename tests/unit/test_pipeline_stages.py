"""Unit tests for pipeline stages — input validation, stage 07, stage 10."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_context(tmp_path):
    from app.pipeline.base import PipelineContext
    return PipelineContext(
        job_id="vjob_test",
        request={"prompt": "test prompt", "target_duration": 120},
    )


# ---------------------------------------------------------------------------
# stage_guardrail
# ---------------------------------------------------------------------------

class TestStageGuardrail:
    def test_allows_benign_prompt(self, tmp_path):
        from app.pipeline.stage_guardrail import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_guardrail.settings") as mock_settings, \
             patch("app.services.guardrail_service.settings", mock_settings), \
             patch("app.services.guardrail_service.generate_json",
                   return_value={"allowed": True, "category": None, "reason": "fine"}), \
             patch("app.pipeline.stage_guardrail.update_stage", MagicMock()), \
             patch("app.pipeline.stage_guardrail.append_event", MagicMock()):
            mock_settings.prompt_guardrail_enabled = True
            mock_settings.prompt_guardrail_model = "openai/gpt-oss-120b"
            mock_settings.prompt_guardrail_fail_open = True
            result = run("vjob_test", {"prompt": "Make a video about renewable energy"}, ctx)

        assert result.status == "completed"

    def test_rejects_harmful_prompt(self, tmp_path):
        from app.pipeline.stage_guardrail import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_guardrail.settings") as mock_settings, \
             patch("app.services.guardrail_service.settings", mock_settings), \
             patch("app.services.guardrail_service.generate_json",
                   return_value={"allowed": False, "category": "harmful", "reason": "violent content"}), \
             patch("app.pipeline.stage_guardrail.update_stage", MagicMock()), \
             patch("app.pipeline.stage_guardrail.append_event", MagicMock()):
            mock_settings.prompt_guardrail_enabled = True
            mock_settings.prompt_guardrail_model = "openai/gpt-oss-120b"
            mock_settings.prompt_guardrail_fail_open = True
            result = run("vjob_test", {"prompt": "how to build a weapon"}, ctx)

        assert result.status == "failed"
        assert result.data["category"] == "harmful"

    def test_fail_open_when_llm_errors(self, tmp_path):
        from app.pipeline.stage_guardrail import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_guardrail.settings") as mock_settings, \
             patch("app.services.guardrail_service.settings", mock_settings), \
             patch("app.services.guardrail_service.generate_json", side_effect=RuntimeError("timeout")), \
             patch("app.pipeline.stage_guardrail.update_stage", MagicMock()), \
             patch("app.pipeline.stage_guardrail.append_event", MagicMock()):
            mock_settings.prompt_guardrail_enabled = True
            mock_settings.prompt_guardrail_model = "openai/gpt-oss-120b"
            mock_settings.prompt_guardrail_fail_open = True
            result = run("vjob_test", {"prompt": "make a video"}, ctx)

        assert result.status == "completed"

    def test_fail_closed_when_llm_errors_and_configured(self, tmp_path):
        from app.pipeline.stage_guardrail import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_guardrail.settings") as mock_settings, \
             patch("app.services.guardrail_service.settings", mock_settings), \
             patch("app.services.guardrail_service.generate_json", side_effect=RuntimeError("timeout")), \
             patch("app.pipeline.stage_guardrail.update_stage", MagicMock()), \
             patch("app.pipeline.stage_guardrail.append_event", MagicMock()):
            mock_settings.prompt_guardrail_enabled = True
            mock_settings.prompt_guardrail_model = "openai/gpt-oss-120b"
            mock_settings.prompt_guardrail_fail_open = False
            result = run("vjob_test", {"prompt": "make a video"}, ctx)

        assert result.status == "failed"

    def test_skipped_when_disabled(self, tmp_path):
        from app.pipeline.stage_guardrail import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_guardrail.settings") as mock_settings:
            mock_settings.prompt_guardrail_enabled = False
            result = run("vjob_test", {"prompt": "make a video"}, ctx)

        assert result.status == "skipped"


# ---------------------------------------------------------------------------
# stage_input_validation
# ---------------------------------------------------------------------------

class TestStageInputValidation:
    def test_success_with_all_paths_present(self, tmp_path):
        from app.pipeline.stage_input_validation import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_input_validation.STORYLINES_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.EXPORTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.DATA_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.SCRIPTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.update_stage", MagicMock()), \
             patch("app.pipeline.stage_input_validation.append_event", MagicMock()):
            # Patch script existence checks
            with patch("pathlib.Path.exists", return_value=True):
                result = run("vjob_test", {"prompt": "make a film"}, ctx)

        assert result.stage == "input_validation"
        assert result.status == "completed"

    def test_fails_with_missing_prompt(self, tmp_path):
        from app.pipeline.stage_input_validation import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_input_validation.STORYLINES_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.EXPORTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.DATA_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.SCRIPTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.update_stage", MagicMock()), \
             patch("app.pipeline.stage_input_validation.append_event", MagicMock()), \
             patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="prompt"):
                run("vjob_test", {"prompt": "  "}, ctx)

    def test_fails_with_missing_directory(self, tmp_path):
        from app.pipeline.stage_input_validation import run
        ctx = _make_context(tmp_path)
        missing = tmp_path / "does_not_exist"

        with patch("app.pipeline.stage_input_validation.STORYLINES_DIR", missing), \
             patch("app.pipeline.stage_input_validation.EXPORTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.DATA_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.SCRIPTS_DIR", tmp_path), \
             patch("app.pipeline.stage_input_validation.settings.ai4ia_storage_backend", "local"), \
             patch("app.pipeline.stage_input_validation.update_stage", MagicMock()), \
             patch("app.pipeline.stage_input_validation.append_event", MagicMock()), \
             patch("pathlib.Path.exists", side_effect=lambda p=None: Path(p if p else "").exists() if p else False):
            # Directory checks only run for the "local" storage backend
            # (they're skipped entirely in "s3" mode — see stage_input_validation.run).
            # Let only the missing dir fail
            with pytest.raises(RuntimeError, match="Directory not found"):
                run("vjob_test", {"prompt": "valid prompt"}, ctx)


# ---------------------------------------------------------------------------
# stage_07_storyline
# ---------------------------------------------------------------------------

class TestStage07Storyline:
    def test_success_when_script_exits_0(self, tmp_path):
        from app.pipeline.stage_07_storyline import run
        ctx = _make_context(tmp_path)
        storyline_file = tmp_path / "storyline_20260101_120000.json"
        storyline_file.write_text('{"title": "Test", "sections": []}')

        job = {"job_id": "vjob_test", "paths": {}}
        with patch("app.pipeline.stage_07_storyline.run_script", return_value=(0, "", "")), \
             patch("app.pipeline.stage_07_storyline._newest_file_after", return_value=storyline_file), \
             patch("app.pipeline.stage_07_storyline.download_if_missing", return_value={}), \
             patch("app.pipeline.stage_07_storyline.use_s3", return_value=False), \
             patch("app.pipeline.stage_07_storyline.update_stage", MagicMock()), \
             patch("app.pipeline.stage_07_storyline.append_event", MagicMock()), \
             patch("app.pipeline.stage_07_storyline.load_job", return_value=job), \
             patch("app.pipeline.stage_07_storyline.save_job", MagicMock()):
            result = run("vjob_test", {"prompt": "AI film", "target_duration": 120}, ctx)
        assert result.status == "completed"

    def test_raises_when_script_non_zero(self, tmp_path):
        from app.pipeline.stage_07_storyline import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_07_storyline.run_script", return_value=(1, "", "error output")), \
             patch("app.pipeline.stage_07_storyline.download_if_missing", return_value={}), \
             patch("app.pipeline.stage_07_storyline.update_stage", MagicMock()), \
             patch("app.pipeline.stage_07_storyline.append_event", MagicMock()):
            with pytest.raises(RuntimeError, match="07_generate_storyline"):
                run("vjob_test", {"prompt": "AI film", "target_duration": 120}, ctx)

    def test_raises_when_target_duration_missing(self, tmp_path):
        from app.pipeline.stage_07_storyline import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_07_storyline.download_if_missing", return_value={}), \
             patch("app.pipeline.stage_07_storyline.update_stage", MagicMock()), \
             patch("app.pipeline.stage_07_storyline.append_event", MagicMock()):
            with pytest.raises(ValueError, match="target_duration is required"):
                run("vjob_test", {"prompt": "AI film"}, ctx)

    def test_raises_when_no_output_file(self, tmp_path):
        from app.pipeline.stage_07_storyline import run
        ctx = _make_context(tmp_path)

        with patch("app.pipeline.stage_07_storyline.run_script", return_value=(0, "", "")), \
             patch("app.pipeline.stage_07_storyline._newest_file_after", return_value=None), \
             patch("app.pipeline.stage_07_storyline.download_if_missing", return_value={}), \
             patch("app.pipeline.stage_07_storyline.update_stage", MagicMock()), \
             patch("app.pipeline.stage_07_storyline.append_event", MagicMock()):
            with pytest.raises(RuntimeError, match="No storyline JSON"):
                run("vjob_test", {"prompt": "AI film", "target_duration": 120}, ctx)


# ---------------------------------------------------------------------------
# stage_10_validate
# ---------------------------------------------------------------------------

class TestStage10Validate:
    def test_skipped_when_no_export_manifest(self, tmp_path):
        from app.pipeline.stage_10_validate import run
        ctx = _make_context(tmp_path)
        ctx.export_manifest_path = str(tmp_path / "nonexistent_manifest.json")  # doesn't exist

        with patch("app.pipeline.stage_10_validate.update_stage", MagicMock()), \
             patch("app.pipeline.stage_10_validate.append_event", MagicMock()):
            result = run("vjob_test", {}, ctx)
        assert result.status in ("skipped", "completed")

    def test_success_on_exit_0(self, tmp_path):
        from app.pipeline.stage_10_validate import run
        ctx = _make_context(tmp_path)
        manifest = tmp_path / "export_manifest.json"
        manifest.write_text('{"clips": []}')
        ctx.export_manifest_path = str(manifest)

        with patch("app.pipeline.stage_10_validate.run_script", return_value=(0, "", "")), \
             patch("app.pipeline.stage_10_validate.update_stage", MagicMock()), \
             patch("app.pipeline.stage_10_validate.append_event", MagicMock()), \
             patch("pathlib.Path.exists", return_value=True):
            result = run("vjob_test", {}, ctx)
        assert result.status == "completed"

    def test_non_zero_exit_raises(self, tmp_path):
        """A non-zero exit is now a hard failure, not a soft warning added to
        the job (see commit 5a9dca4, "A/V sync drift, export consistency
        validation & zero-speaker clip handling")."""
        from app.pipeline.stage_10_validate import run
        ctx = _make_context(tmp_path)
        manifest = tmp_path / "export_manifest.json"
        manifest.write_text('{"clips": []}')
        ctx.export_manifest_path = str(manifest)

        with patch("app.pipeline.stage_10_validate.run_script", return_value=(1, "", "validation warning")), \
             patch("app.pipeline.stage_10_validate.update_stage", MagicMock()), \
             patch("app.pipeline.stage_10_validate.append_event", MagicMock()), \
             patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(RuntimeError, match="10_validate_video_export.py failed"):
                run("vjob_test", {}, ctx)
