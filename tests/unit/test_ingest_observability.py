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
