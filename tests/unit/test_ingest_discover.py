"""_stage_discover — was a bare subprocess.run with zero test coverage and no
captured output. Now routed through _run_script (via its cmd= override, since
yt-dlp is a binary, not a PYTHON_BIN-run script) so a discovery failure is
visible the same way every other stage's is."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def no_archive(tmp_path, monkeypatch):
    """No local download archive / raw video dir, so every channel video
    counts as new."""
    import app.services.ingest_job_runner as runner
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "RAW_VIDEO_DIR", tmp_path / "raw" / "youtube" / "ai4ia_channel")
    monkeypatch.setattr(runner, "_ingest_use_s3", lambda: False)


class TestStageDiscoverRoutesThroughRunScript:
    def test_uses_run_script_not_a_bare_subprocess(self, no_archive):
        """The regression this whole fix addresses: discovery's output must
        be captured and surfaced like every other stage's, not silently lost
        to a bare subprocess.run."""
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "_run_script",
                          return_value=(0, "1\tabc\tTitle A\n", "")) as mock_run, \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            runner._stage_discover("job1", "https://youtube.com/@x", None)

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs.get("cmd") is not None
        assert mock_run.call_args.args[3] == "discover"  # stage name

    def test_parses_channel_listing_from_run_script_stdout(self, no_archive):
        import app.services.ingest_job_runner as runner

        stdout = "1\tabc\tTitle A\n2\tdef\tTitle B\n"
        with patch.object(runner, "_run_script", return_value=(0, stdout, "")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            new_videos = runner._stage_discover("job1", "https://youtube.com/@x", None)

        ids = {v["video_id"] for v in new_videos}
        assert ids == {"abc", "def"}

    def test_101_is_the_expected_max_downloads_exit_code(self, no_archive):
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "_run_script", return_value=(101, "1\tabc\tT\n", "")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            new_videos = runner._stage_discover("job1", "https://youtube.com/@x", 1)

        assert len(new_videos) == 1

    def test_other_nonzero_exit_raises(self, no_archive):
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "_run_script", return_value=(1, "", "some yt-dlp error")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            with pytest.raises(RuntimeError, match="some yt-dlp error"):
                runner._stage_discover("job1", "https://youtube.com/@x", None)

    def test_timeout_surfaces_as_a_runtime_error(self, no_archive):
        """_run_script itself never raises on timeout (it returns rc=-1) —
        the caller is responsible for turning that into a failure."""
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "_run_script",
                          return_value=(-1, "", "Script timed out after 120s: yt-dlp")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            with pytest.raises(RuntimeError, match="timed out"):
                runner._stage_discover("job1", "https://youtube.com/@x", None)

    def test_cookies_are_passed_through_when_configured(self, no_archive):
        import app.services.ingest_job_runner as runner

        with patch.object(runner.settings, "youtube_cookies_file", "/tmp/cookies.txt"), \
             patch.object(runner, "_run_script", return_value=(0, "", "")) as mock_run, \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "upsert_video_stage"):
            runner._stage_discover("job1", "https://youtube.com/@x", None)

        cmd = mock_run.call_args.kwargs["cmd"]
        assert "--cookies" in cmd
        assert "/tmp/cookies.txt" in cmd
