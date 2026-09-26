"""Resuming a failed ingest job from where it died, instead of restarting at
discover.

Discover only finds videos not yet in yt-dlp's download archive, so a video
that WAS downloaded but failed a later stage was permanently invisible to
every future sync — the audit finding this closes. It also replaces
trigger_single_video_sync's already-downloaded branch, which used to reset 5
ingest_status rows to "pending" and then start a plain job that begins at
discover and never reads those rows.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestResolveResumeCandidates:
    def test_prefers_the_stage_own_failed_list(self):
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {
            "current_stage": "transcribe",
            "stage_status": {
                "extract_audio": {"succeeded": ["a", "b"]},
                "transcribe": {"succeeded": [], "failed": ["a", "b"]},
            },
        }
        stage, ids = resolve_resume_candidates(job)
        assert stage == "transcribe"
        assert ids == ["a", "b"]

    def test_falls_back_to_the_predecessor_when_the_stage_recorded_nothing(self):
        """Covers a stage that crashed before reaching _require_some_success,
        so it never wrote its own failed list."""
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {
            "current_stage": "classify",
            "stage_status": {
                "create_fragments": {"succeeded": ["a", "b", "c"]},
                "classify": {"status": "running"},  # crashed mid-stage
            },
        }
        stage, ids = resolve_resume_candidates(job)
        assert stage == "classify"
        assert ids == ["a", "b", "c"]

    def test_download_uses_the_downloaded_key_not_succeeded(self):
        """download's _update_job call uses downloaded=, every tail stage
        uses succeeded= — extract_audio's predecessor lookup must know that."""
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {
            "current_stage": "extract_audio",
            "stage_status": {
                "download": {"downloaded": ["a", "b"], "succeeded": ["should not be used"]},
                "extract_audio": {"status": "running"},
            },
        }
        stage, ids = resolve_resume_candidates(job)
        assert ids == ["a", "b"]

    def test_explicit_from_stage_overrides_current_stage(self):
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {
            "current_stage": "embed",
            "stage_status": {
                "create_fragments": {"succeeded": ["a"]},
                "classify": {"failed": ["a"]},
            },
        }
        stage, ids = resolve_resume_candidates(job, from_stage="classify")
        assert stage == "classify"
        assert ids == ["a"]

    @pytest.mark.parametrize("stage", ["discover", "download", "queued", "unknown", None])
    def test_refuses_non_resumable_stages(self, stage):
        """discover/download failures mean the video was never fully
        downloaded, so a plain POST /sync already retries them correctly."""
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {"current_stage": stage, "stage_status": {}}
        with pytest.raises(ValueError, match="Cannot resume"):
            resolve_resume_candidates(job)

    def test_refuses_when_nothing_is_recorded_anywhere(self):
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {"current_stage": "transcribe", "stage_status": {}}
        with pytest.raises(ValueError, match="Nothing to resume"):
            resolve_resume_candidates(job)

    def test_missing_stage_status_key_does_not_crash(self):
        from app.services.ingest_job_runner import resolve_resume_candidates
        job = {"current_stage": "transcribe"}  # no stage_status at all
        with pytest.raises(ValueError):
            resolve_resume_candidates(job)


class TestCreateResumeJob:
    def test_seeds_the_new_job_with_resume_fields(self):
        import app.services.ingest_job_runner as runner

        saved = {}

        def _save(job_id, data):
            saved.update(data)

        with patch.object(runner, "create_ingest_job", return_value="ijob_new"), \
             patch.object(runner, "load_ingest_job", return_value={}), \
             patch.object(runner, "save_ingest_job", side_effect=_save):
            job_id = runner.create_resume_job(
                resume_from_stage="classify", resume_video_ids=["a", "b"],
                channel_url="https://x", resumed_from_job_id="ijob_old",
            )

        assert job_id == "ijob_new"
        assert saved["resume_from_stage"] == "classify"
        assert saved["resume_video_ids"] == ["a", "b"]
        assert saved["resumed_from_job_id"] == "ijob_old"

    def test_resumed_from_job_id_is_optional(self):
        """The single-video retrigger path resumes without an originating
        failed job — it force-reprocesses one video regardless of history."""
        import app.services.ingest_job_runner as runner

        saved = {}
        with patch.object(runner, "create_ingest_job", return_value="ijob_new"), \
             patch.object(runner, "load_ingest_job", return_value={}), \
             patch.object(runner, "save_ingest_job", side_effect=lambda jid, data: saved.update(data)):
            runner.create_resume_job(resume_from_stage="extract_audio", resume_video_ids=["v1"])

        assert "resumed_from_job_id" not in saved


class TestRunResumedStages:
    def test_runs_only_the_stages_from_from_stage_onward(self):
        import app.services.ingest_job_runner as runner

        calls = []
        fns = {
            "extract_audio": lambda jid, ids: (calls.append("extract_audio"), ids)[1],
            "transcribe": lambda jid, ids: (calls.append("transcribe"), ids)[1],
            "create_fragments": lambda jid, ids: (calls.append("create_fragments"), ids)[1],
            "classify": lambda jid, ids: (calls.append("classify"), ids)[1],
            "embed": lambda jid, ids: (calls.append("embed"), ids)[1],
        }
        with patch.object(runner, "_RESUMABLE_STAGE_FNS", fns), \
             patch.object(runner, "_is_cancelling", return_value=False), \
             patch.object(runner, "_complete_job"), \
             patch.object(runner, "_emit_ingest_decision_record"):
            runner._run_resumed_stages("job1", "create_fragments", ["v1"])

        assert calls == ["create_fragments", "classify", "embed"]

    def test_threads_ids_from_one_stage_to_the_next(self):
        import app.services.ingest_job_runner as runner

        fns = {
            "classify": lambda jid, ids: ["v1"],  # v2 dropped by classify
            "embed": lambda jid, ids: ids,
        }
        received_by_embed = []
        fns["embed"] = lambda jid, ids: (received_by_embed.append(list(ids)), ids)[1]

        with patch.object(runner, "_RESUMABLE_STAGE_FNS", fns), \
             patch.object(runner, "_is_cancelling", return_value=False), \
             patch.object(runner, "_complete_job"), \
             patch.object(runner, "_emit_ingest_decision_record"):
            runner._run_resumed_stages("job1", "classify", ["v1", "v2"])

        assert received_by_embed == [["v1"]]

    def test_completes_the_job_with_a_resume_summary(self):
        """_RESUMABLE_STAGE_FNS captures each _stage_* function reference at
        module-import time, so patching runner._stage_embed directly (rather
        than the dict entry) would not reach the lookup _run_resumed_stages
        actually uses."""
        import app.services.ingest_job_runner as runner

        summaries = []
        fns = dict(runner._RESUMABLE_STAGE_FNS, embed=lambda jid, ids: ["v1"])
        with patch.object(runner, "_RESUMABLE_STAGE_FNS", fns), \
             patch.object(runner, "_is_cancelling", return_value=False), \
             patch.object(runner, "_complete_job", side_effect=lambda jid, s: summaries.append(s)), \
             patch.object(runner, "_emit_ingest_decision_record"):
            runner._run_resumed_stages("job1", "embed", ["v1"])

        assert summaries[0]["resumed_from_stage"] == "embed"
        assert summaries[0]["resumed_video_count"] == 1

    def test_cancellation_stops_before_the_next_stage(self):
        import app.services.ingest_job_runner as runner

        with patch.object(runner, "_is_cancelling", return_value=True), \
             patch.object(runner, "_cancel_job") as mock_cancel, \
             patch.object(runner, "_stage_transcribe") as mock_transcribe:
            runner._run_resumed_stages("job1", "transcribe", ["v1"])

        mock_transcribe.assert_not_called()
        mock_cancel.assert_called_once()

    def test_raises_on_an_unresumable_stage_rather_than_failing_the_job_itself(self):
        """run_ingest_job's own except handler owns turning this into a job
        failure — a second failure path here would risk a racy double write."""
        import app.services.ingest_job_runner as runner
        with pytest.raises(RuntimeError, match="Cannot resume"):
            runner._run_resumed_stages("job1", "download", ["v1"])

    def test_raises_when_given_no_video_ids(self):
        import app.services.ingest_job_runner as runner
        with pytest.raises(RuntimeError, match="no video_ids"):
            runner._run_resumed_stages("job1", "classify", [])


class TestRunIngestJobDispatchesToResume:
    def test_resume_from_stage_skips_discover(self):
        import app.services.ingest_job_runner as runner

        job = {"status": "queued", "resume_from_stage": "classify", "resume_video_ids": ["v1"]}
        with patch.object(runner, "load_ingest_job", return_value=dict(job)), \
             patch.object(runner, "save_ingest_job"), \
             patch.object(runner, "_run_resumed_stages") as mock_resume, \
             patch.object(runner, "_stage_discover") as mock_discover:
            runner.run_ingest_job("job1")

        mock_resume.assert_called_once_with("job1", "classify", ["v1"])
        mock_discover.assert_not_called()

    def test_no_resume_field_runs_the_normal_pipeline(self):
        import app.services.ingest_job_runner as runner

        job = {"status": "queued"}
        with patch.object(runner, "load_ingest_job", return_value=dict(job)), \
             patch.object(runner, "save_ingest_job"), \
             patch.object(runner, "_run_resumed_stages") as mock_resume, \
             patch.object(runner, "_stage_discover", return_value=[]), \
             patch.object(runner, "_complete_job"), \
             patch.object(runner, "_is_cancelling", return_value=False):
            runner.run_ingest_job("job1")

        mock_resume.assert_not_called()

    def test_a_second_failure_during_resume_still_fails_the_job(self):
        """The resumed stage raising propagates to the same top-level except
        every other stage failure already goes through."""
        import app.services.ingest_job_runner as runner

        job = {"status": "queued", "resume_from_stage": "embed", "resume_video_ids": ["v1"]}
        saved = {}
        with patch.object(runner, "load_ingest_job", side_effect=lambda jid: dict(job, **saved)), \
             patch.object(runner, "save_ingest_job", side_effect=saved.update), \
             patch.object(runner, "_run_resumed_stages",
                          side_effect=RuntimeError("All 1 videos failed embed")), \
             patch.object(runner, "_fail_job") as mock_fail:
            runner.run_ingest_job("job1")

        mock_fail.assert_called_once()
        assert mock_fail.call_args.args[0] == "job1"
