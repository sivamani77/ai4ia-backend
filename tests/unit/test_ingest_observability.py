"""Ingest failure reporting and subprocess-output visibility.

Both behaviours here were learned from a real run: job
ijob_20260920_143200_cf022f1b downloaded 0 of 43 videos over 15 minutes and
reported status "completed", progress 100, error null. Nothing in the API
could show why, because the stage log files are written by whichever
container ran the stage while the log-tail endpoint reads its own filesystem.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@contextlib.contextmanager
def _runner_harness(saved: dict | None = None):
    """Patch the ingest store so run_ingest_job can be driven without a DB."""
    import app.services.ingest_job_runner as runner

    job: dict = {"job_id": "job1", "status": "queued", "stage_status": {}}

    def _save(job_id, data):
        job.update(data)
        if saved is not None:
            saved.update(data)

    with patch.object(runner, "load_ingest_job", side_effect=lambda jid: dict(job)), \
         patch.object(runner, "save_ingest_job", side_effect=_save), \
         patch.object(runner, "append_ingest_event"), \
         patch.object(runner, "_emit_ingest_decision_record"), \
         patch.object(runner, "_is_cancelling", return_value=False):
        yield job


class TestTotalDownloadFailureIsAFailure:
    def test_job_fails_when_every_download_fails(self):
        """The regression that hid a 43/43 cookie-wall failure behind a green
        'completed' badge."""
        import app.services.ingest_job_runner as runner

        new_videos = [{"video_id": f"v{i}", "title": "t"} for i in range(43)]
        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover", return_value=new_videos), \
             patch.object(runner, "_stage_download", return_value=[]), \
             patch.object(runner, "_stage_extract_audio") as later_stage:
            runner.run_ingest_job("job1")

        assert job["status"] == "failed"
        assert job["error"]["stage"] == "download"
        assert "43" in job["error"]["message"]
        later_stage.assert_not_called()

    def test_failure_still_records_the_counts(self):
        """_fail_job does not set summary, so the counts must be written
        first — losing them would make the failure unexplainable."""
        import app.services.ingest_job_runner as runner

        new_videos = [{"video_id": "v1", "title": "t"}]
        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover", return_value=new_videos), \
             patch.object(runner, "_stage_download", return_value=[]), \
             patch.object(runner, "_stage_extract_audio"):
            runner.run_ingest_job("job1")

        assert job["summary"]["downloaded"] == 0
        assert job["summary"]["new_videos_found"] == 1

    def test_error_points_at_the_usual_cause(self):
        import app.services.ingest_job_runner as runner

        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover",
                          return_value=[{"video_id": "v1", "title": "t"}]), \
             patch.object(runner, "_stage_download", return_value=[]), \
             patch.object(runner, "_stage_extract_audio"):
            runner.run_ingest_job("job1")

        assert "YOUTUBE_COOKIES_FILE" in job["error"]["details"]

    def test_empty_channel_still_completes_successfully(self):
        """"Nothing new to sync" is a success and must not be swept into the
        new failure path."""
        import app.services.ingest_job_runner as runner

        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover", return_value=[]), \
             patch.object(runner, "_stage_download") as download:
            runner.run_ingest_job("job1")

        assert job["status"] == "completed"
        download.assert_not_called()

    def test_partial_download_success_still_proceeds(self):
        """Only a *total* failure aborts — one good video carries on."""
        import app.services.ingest_job_runner as runner

        new_videos = [{"video_id": "v1"}, {"video_id": "v2"}]
        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover", return_value=new_videos), \
             patch.object(runner, "_stage_download", return_value=["v1"]), \
             patch.object(runner, "_stage_extract_audio", return_value=["v1"]) as extract, \
             patch.object(runner, "_stage_transcribe", return_value=["v1"]), \
             patch.object(runner, "_stage_create_fragments", return_value=["v1"]), \
             patch.object(runner, "_stage_classify", return_value=["v1"]), \
             patch.object(runner, "_stage_embed", return_value=["v1"]):
            runner.run_ingest_job("job1")

        extract.assert_called_once()
        assert job["status"] == "completed"


class TestRequireSomeSuccess:
    """Every stage past download had the identical shape download did: verify
    per video, then report 'completed' regardless of whether anything
    survived. Fixing only download would have just moved the same silent
    failure one stage further down the pipeline."""

    def test_raises_when_every_video_in_a_batch_failed(self):
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_update_job"):
            with pytest.raises(RuntimeError, match="All 2 videos failed transcribe"):
                runner._require_some_success("job1", "transcribe", ["v1", "v2"], [], ["v1", "v2"])

    def test_does_not_raise_on_partial_success(self):
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_update_job") as mock_update:
            runner._require_some_success("job1", "transcribe", ["v1", "v2"], ["v1"], ["v2"])
        mock_update.assert_not_called()

    def test_does_not_raise_when_the_batch_was_empty(self):
        """video_ids=[] means the stage had nothing to do, not that it failed
        -- that case is handled by each stage's own early return."""
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_update_job") as mock_update:
            runner._require_some_success("job1", "transcribe", [], [], [])
        mock_update.assert_not_called()

    def test_marks_the_stage_failed_before_raising(self):
        """So the job's stage_status carries the failure and counts even
        though _fail_job (called by run_ingest_job's except) doesn't set them."""
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_update_job") as mock_update:
            with pytest.raises(RuntimeError):
                runner._require_some_success("job1", "classify", ["v1"], [], ["v1"])
        mock_update.assert_called_once()
        args = mock_update.call_args.args
        assert args[1] == "classify"
        assert args[2] == "failed"

    @pytest.mark.parametrize("stage_fn_name", [
        "_stage_extract_audio", "_stage_transcribe",
        "_stage_create_fragments", "_stage_classify",
    ])
    def test_real_stage_raises_when_verification_finds_nothing(self, stage_fn_name):
        """Drives the REAL stage function (not a mock standing in for it) to
        a genuine zero-survivor outcome: the script "succeeds" but every
        per-video output file is missing, exactly like a cookie-wall failure
        that only breaks part way through a stage's script."""
        import app.services.ingest_job_runner as runner
        stage_fn = getattr(runner, stage_fn_name)

        with patch.object(runner, "_ingest_use_s3", return_value=False), \
             patch.object(runner, "_run_script", return_value=(0, "", "")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "upsert_video_stage"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "_emit_ingest_trace"), \
             patch.object(runner, "_resolve_stem", side_effect=lambda vid: f"0001_{vid}_T"), \
             patch("pathlib.Path.exists", return_value=False):  # no output ever lands
            with pytest.raises(RuntimeError, match="All 1 videos failed"):
                stage_fn("job1", ["v1"])

    def test_total_failure_partway_through_the_pipeline_fails_the_whole_job(self):
        """End-to-end version: a real stage raising via _require_some_success
        propagates through run_ingest_job's except into a failed job, not an
        all-zeros green completion."""
        import app.services.ingest_job_runner as runner

        with _runner_harness() as job, \
             patch.object(runner, "_stage_discover",
                          return_value=[{"video_id": "v1"}]), \
             patch.object(runner, "_stage_download", return_value=["v1"]), \
             patch.object(runner, "_stage_extract_audio",
                          side_effect=RuntimeError("All 1 videos failed transcribe")):
            runner.run_ingest_job("job1")

        assert job["status"] == "failed"


class TestScriptOutputReachesTheEventStream:
    """Log files live on whichever container ran the stage; Postgres is the
    only channel the API and worker both see."""

    def _drain(self, lines: list[str], tmp_path: Path, events: list):
        import app.services.ingest_job_runner as runner

        def _capture(job_id, event_type, data=None):
            events.append((event_type, data))

        with patch.object(runner, "append_ingest_event", side_effect=_capture), \
             patch.object(runner, "_log", return_value=logging.getLogger("t")), \
             patch.object(runner, "DATA_DIR", tmp_path), \
             patch("subprocess.Popen") as popen:
            proc = MagicMock()
            proc.stdout = iter(lines)
            proc.stderr = iter([])
            proc.wait.return_value = 0
            proc.returncode = 0
            popen.return_value = proc
            runner._run_script("script.py", [], "job1", "download", timeout=5)

    def test_stdout_lines_become_script_log_events(self, tmp_path):
        events: list = []
        self._drain(["hello\n", "world\n"], tmp_path, events)

        logs = [d for t, d in events if t == "script_log"]
        assert logs, "no script_log events emitted"
        assert any("hello" in line for d in logs for line in d["lines"])

    def test_events_carry_the_stage_and_stream(self, tmp_path):
        events: list = []
        self._drain(["boom\n"], tmp_path, events)

        logs = [d for t, d in events if t == "script_log"]
        assert logs[0]["stage"] == "download"
        assert logs[0]["stream"] == "stdout"

    def test_lines_are_batched_not_one_event_each(self, tmp_path):
        """yt-dlp is chatty; one row per line would hammer Postgres."""
        events: list = []
        self._drain([f"line {i}\n" for i in range(40)], tmp_path, events)

        logs = [d for t, d in events if t == "script_log"]
        assert len(logs) < 40
        assert sum(len(d["lines"]) for d in logs) == 40

    def test_absurdly_long_lines_are_capped(self, tmp_path):
        """A yt-dlp progress bar uses \\r, so it can arrive as one huge line."""
        import app.services.ingest_job_runner as runner
        events: list = []
        self._drain(["x" * 50_000 + "\n"], tmp_path, events)

        logs = [d for t, d in events if t == "script_log"]
        assert all(len(line) <= runner._EVENT_LINE_CHARS
                   for d in logs for line in d["lines"])

    def test_event_failure_never_breaks_the_pipeline(self, tmp_path):
        """Observability must not be able to fail a job."""
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "append_ingest_event", side_effect=Exception("db down")), \
             patch.object(runner, "_log", return_value=logging.getLogger("t")), \
             patch.object(runner, "DATA_DIR", tmp_path), \
             patch("subprocess.Popen") as popen:
            proc = MagicMock()
            proc.stdout = iter(["a\n"])
            proc.stderr = iter([])
            proc.wait.return_value = 0
            proc.returncode = 0
            popen.return_value = proc
            code, out, err = runner._run_script("s.py", [], "job1", "download", timeout=5)

        assert code == 0
        assert "a" in out

    def test_log_files_are_still_written(self, tmp_path):
        """Events are additive — the existing file-based tail must keep working
        for anyone on the same filesystem."""
        events: list = []
        self._drain(["kept\n"], tmp_path, events)

        stdout_log = tmp_path / "jobs" / "job1" / "logs" / "download.stdout.log"
        assert stdout_log.exists()
        assert "kept" in stdout_log.read_text()
