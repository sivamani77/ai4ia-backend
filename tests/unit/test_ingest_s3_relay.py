"""S3 input rehydration for the ingest stages.

Each ingest stage uploads its outputs and deletes the local copies to free
disk, but the stage scripts only ever glob local directories. Without a
download step the next stage starts against an empty directory and silently
produces nothing, which is what made s3 mode a no-op end to end.

The local-mode tests matter as much as the s3 ones: they pin that nothing new
happens when AI4IA_STORAGE_BACKEND is not s3.
"""
from __future__ import annotations

import contextlib
import re
from unittest.mock import MagicMock, patch

import pytest


def _fake_s3_client(keys: list[str]):
    """A boto3 stub whose paginator lists `keys`."""
    s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    s3.get_paginator.return_value = paginator
    return s3


@contextlib.contextmanager
def _s3_listing(keys: list[str]):
    s3 = _fake_s3_client(keys)
    with patch("app.services.ingest_job_runner._s3_client", return_value=(s3, "test-bucket")):
        yield s3


class TestVideoIdFromName:
    def test_parses_mp4(self):
        from app.services.ingest_job_runner import _video_id_from_name
        assert _video_id_from_name("0001_abc123_Some Title.mp4") == "abc123"

    def test_parses_info_json_double_suffix(self):
        from app.services.ingest_job_runner import _video_id_from_name
        assert _video_id_from_name("0001_abc123_Some Title.info.json") == "abc123"

    def test_returns_none_for_unparseable(self):
        from app.services.ingest_job_runner import _video_id_from_name
        assert _video_id_from_name("notavideo.mp4") is None


class TestS3DownloadDir:
    def test_downloads_everything_when_no_video_filter(self, tmp_path):
        from app.services.ingest_job_runner import _s3_download_dir
        keys = ["raw/fragments/0001_aaa_T.fragments.json",
                "raw/fragments/0002_bbb_T.fragments.json"]
        with _s3_listing(keys) as s3:
            count = _s3_download_dir("raw/fragments", tmp_path)
        assert count == 2
        assert s3.download_file.call_count == 2

    def test_filters_to_requested_videos(self, tmp_path):
        from app.services.ingest_job_runner import _s3_download_dir
        keys = ["raw/audio/0001_aaa_T.wav",
                "raw/audio/0002_bbb_T.wav",
                "raw/audio/0003_ccc_T.wav"]
        with _s3_listing(keys) as s3:
            count = _s3_download_dir("raw/audio", tmp_path, video_ids=["bbb"])
        assert count == 1
        downloaded = s3.download_file.call_args.args[1]
        assert "0002_bbb_T.wav" in downloaded

    def test_filters_by_suffix(self, tmp_path):
        from app.services.ingest_job_runner import _s3_download_dir
        keys = ["raw/youtube/ai4ia_channel/0001_aaa_T.mp4",
                "raw/youtube/ai4ia_channel/0001_aaa_T.info.json",
                "raw/youtube/ai4ia_channel/0001_aaa_T.jpg"]
        with _s3_listing(keys) as s3:
            count = _s3_download_dir("raw/youtube/ai4ia_channel", tmp_path,
                                     video_ids=["aaa"], suffixes=(".mp4",))
        assert count == 1
        assert s3.download_file.call_args.args[1].endswith(".mp4")

    def test_skips_files_already_on_disk(self, tmp_path):
        from app.services.ingest_job_runner import _s3_download_dir
        (tmp_path / "0001_aaa_T.wav").write_text("already here")
        keys = ["raw/audio/0001_aaa_T.wav", "raw/audio/0002_bbb_T.wav"]
        with _s3_listing(keys) as s3:
            count = _s3_download_dir("raw/audio", tmp_path)
        assert count == 1
        assert s3.download_file.call_count == 1

    def test_one_failure_does_not_abort_the_rest(self, tmp_path):
        from app.services.ingest_job_runner import _s3_download_dir
        keys = ["raw/audio/0001_aaa_T.wav", "raw/audio/0002_bbb_T.wav"]
        with _s3_listing(keys) as s3:
            s3.download_file.side_effect = [Exception("boom"), None]
            count = _s3_download_dir("raw/audio", tmp_path)
        assert count == 1


@contextlib.contextmanager
def _stage_harness(use_s3: bool, chunks: list[list[str]] | None = None):
    """Neutralize everything a stage does except the S3 calls under test.

    Yields (download_mock, cleanup_mock, run_script_mock).
    """
    import app.services.ingest_job_runner as runner

    plan = patch.object(runner, "_plan_chunks", side_effect=lambda vids, *a, **k: chunks) \
        if chunks is not None else contextlib.nullcontext()

    with patch.object(runner, "_ingest_use_s3", return_value=use_s3), \
         patch.object(runner, "_s3_download_dir", return_value=0) as mock_download, \
         patch.object(runner, "_s3_cleanup_dir", return_value=0) as mock_cleanup, \
         patch.object(runner, "_s3_upload_dir", return_value=0), \
         patch.object(runner, "_s3_sizes", return_value={}), \
         patch.object(runner, "_assert_free_space"), \
         patch.object(runner, "_run_script", return_value=(0, "", "")) as mock_run, \
         patch.object(runner, "_update_job"), \
         patch.object(runner, "upsert_video_stage"), \
         patch.object(runner, "append_ingest_event"), \
         patch.object(runner, "_emit_ingest_trace"), \
         patch.object(runner, "_resolve_stem", side_effect=lambda vid: f"0001_{vid}_T"), \
         plan:
        yield mock_download, mock_cleanup, mock_run


def _prefixes(mock_download) -> list[str]:
    return [call.args[0] for call in mock_download.call_args_list]


def _video_ids_arg(call):
    """The video_ids the call scoped to, whether passed positionally or by
    keyword. None means "fetch the whole prefix"."""
    if len(call.args) >= 3:
        return call.args[2]
    return call.kwargs.get("video_ids")


class TestStagesRehydrateInS3Mode:
    def test_extract_audio_pulls_the_mp4s(self):
        from app.services.ingest_job_runner import _stage_extract_audio
        with _stage_harness(use_s3=True) as (mock_download, _, _run):
            _stage_extract_audio("job1", ["aaa"])
        assert "raw/youtube/ai4ia_channel" in _prefixes(mock_download)

    def test_transcribe_pulls_the_audio(self):
        from app.services.ingest_job_runner import _stage_transcribe
        with _stage_harness(use_s3=True) as (mock_download, _, _run):
            _stage_transcribe("job1", ["aaa"])
        assert "raw/audio" in _prefixes(mock_download)

    def test_create_fragments_pulls_transcripts_and_video_metadata(self):
        from app.services.ingest_job_runner import _stage_create_fragments
        with _stage_harness(use_s3=True) as (mock_download, _, _run):
            _stage_create_fragments("job1", ["aaa"])
        prefixes = _prefixes(mock_download)
        assert "raw/transcripts" in prefixes
        # 04 reads {stem}.info.json for video_year / video_upload_date.
        assert "raw/youtube/ai4ia_channel" in prefixes

    def test_classify_pulls_the_whole_fragments_prefix(self):
        from app.services.ingest_job_runner import _stage_classify
        with _stage_harness(use_s3=True) as (mock_download, _, _run):
            _stage_classify("job1", ["aaa"])
        calls = {c.args[0]: c for c in mock_download.call_args_list}
        assert "raw/fragments" in calls
        # Not scoped to this job's videos — 05 rewrites the whole catalog.
        assert _video_ids_arg(calls["raw/fragments"]) is None

    def test_embed_pulls_the_whole_classified_prefix(self):
        from app.services.ingest_job_runner import _stage_embed
        with _stage_harness(use_s3=True) as (mock_download, _, _run):
            _stage_embed("job1", ["aaa"])
        calls = {c.args[0]: c for c in mock_download.call_args_list}
        assert "raw/fragments_classified" in calls
        assert _video_ids_arg(calls["raw/fragments_classified"]) is None


class TestChunking:
    def test_groups_videos_under_the_budget(self):
        from app.services.ingest_job_runner import _chunk_by_bytes
        sizes = {"a": 400, "b": 400, "c": 400}
        chunks = _chunk_by_bytes(["a", "b", "c"], sizes, budget_bytes=1000)
        assert chunks == [["a", "b"], ["c"]]

    def test_oversized_video_gets_its_own_chunk_rather_than_being_skipped(self):
        """The corpus has a 1302MB video against a 46MB median — an item over
        the whole budget must still be processed."""
        from app.services.ingest_job_runner import _chunk_by_bytes
        sizes = {"small": 100, "huge": 5000}
        chunks = _chunk_by_bytes(["small", "huge"], sizes, budget_bytes=1000)
        assert chunks == [["small"], ["huge"]]
        assert [v for c in chunks for v in c] == ["small", "huge"]

    def test_unknown_sizes_still_flow_through(self):
        from app.services.ingest_job_runner import _chunk_by_bytes
        chunks = _chunk_by_bytes(["a", "b"], {}, budget_bytes=1000)
        assert chunks == [["a", "b"]]

    def test_every_video_lands_in_exactly_one_chunk(self):
        from app.services.ingest_job_runner import _chunk_by_bytes
        ids = [f"v{i}" for i in range(50)]
        sizes = {v: (i % 7) * 100 for i, v in enumerate(ids)}
        chunks = _chunk_by_bytes(ids, sizes, budget_bytes=500)
        flat = [v for c in chunks for v in c]
        assert flat == ids
        assert len(set(flat)) == len(ids)

    def test_local_mode_is_always_one_chunk(self):
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_ingest_use_s3", return_value=False):
            assert runner._plan_chunks(["a", "b", "c"], "raw/audio") == [["a", "b", "c"]]

    def test_falls_back_to_one_chunk_when_sizing_fails(self):
        import app.services.ingest_job_runner as runner
        with patch.object(runner, "_ingest_use_s3", return_value=True), \
             patch.object(runner, "_s3_sizes", side_effect=Exception("s3 down")):
            assert runner._plan_chunks(["a", "b"], "raw/audio") == [["a", "b"]]

    def test_stage_runs_the_script_once_per_chunk_and_keeps_earlier_logs(self):
        from app.services.ingest_job_runner import _stage_extract_audio
        with _stage_harness(use_s3=True, chunks=[["a"], ["b"]]) as (_dl, _cl, mock_run):
            _stage_extract_audio("job1", ["a", "b"])
        assert mock_run.call_count == 2
        truncate_flags = [c.kwargs.get("truncate") for c in mock_run.call_args_list]
        # Only the first chunk may clear the stage log.
        assert truncate_flags == [True, False]


class TestEmbedUploadsEverythingStorylineNeeds:
    """stage_07_storyline's _prepare_embeddings requires the jsonl, the .npy
    matrix and the .json metadata, and raises if any is missing. Ingest used
    to upload only the jsonl, so a clean ingest could still leave video
    generation unable to start."""

    def test_uploads_matrix_and_metadata_alongside_the_jsonl(self):
        import app.services.ingest_job_runner as runner
        from app.services.ingest_job_runner import _stage_embed

        with patch.object(runner, "_ingest_use_s3", return_value=True), \
             patch.object(runner, "_s3_download_dir", return_value=0), \
             patch.object(runner, "_s3_cleanup_dir", return_value=0), \
             patch.object(runner, "_s3_upload_dir", return_value=0) as mock_upload, \
             patch.object(runner, "_run_script", return_value=(0, "", "")), \
             patch.object(runner, "_update_job"), \
             patch.object(runner, "upsert_video_stage"), \
             patch.object(runner, "append_ingest_event"), \
             patch.object(runner, "_emit_ingest_trace"), \
             patch.object(runner, "_resolve_stem", side_effect=lambda vid: f"0001_{vid}_T"), \
             patch("pathlib.Path.exists", return_value=False):  # skip the chroma import step
            _stage_embed("job1", ["aaa"])

        patterns = [c.args[2] for c in mock_upload.call_args_list if len(c.args) >= 3]
        assert "*.jsonl" in patterns
        assert "*.npy" in patterns
        assert "*.json" in patterns


class TestCleanupAfterEachStage:
    def test_extract_audio_releases_the_mp4s(self):
        from app.services.ingest_job_runner import _stage_extract_audio
        with _stage_harness(use_s3=True) as (_dl, mock_cleanup, _run):
            _stage_extract_audio("job1", ["aaa"])
        assert mock_cleanup.called

    def test_transcribe_releases_the_wavs(self):
        from app.services.ingest_job_runner import _stage_transcribe
        with _stage_harness(use_s3=True) as (_dl, mock_cleanup, _run):
            _stage_transcribe("job1", ["aaa"])
        assert mock_cleanup.called

    def test_cleanup_happens_per_chunk_not_once_at_the_end(self):
        """Peak disk is only bounded if each chunk's inputs go away before the
        next chunk's arrive."""
        from app.services.ingest_job_runner import _stage_extract_audio
        with _stage_harness(use_s3=True, chunks=[["a"], ["b"]]) as (_dl, mock_cleanup, _run):
            _stage_extract_audio("job1", ["a", "b"])
        assert mock_cleanup.call_count == 2

    def test_cleanup_deletes_only_matching_files(self, tmp_path):
        from app.services.ingest_job_runner import _s3_cleanup_dir
        (tmp_path / "0001_aaa_T.mp4").write_bytes(b"x" * 10)
        (tmp_path / "0002_bbb_T.mp4").write_bytes(b"x" * 10)
        (tmp_path / "0001_aaa_T.info.json").write_text("{}")

        freed = _s3_cleanup_dir(tmp_path, video_ids=["aaa"], suffixes=(".mp4",))

        assert freed == 10
        assert not (tmp_path / "0001_aaa_T.mp4").exists()
        assert (tmp_path / "0002_bbb_T.mp4").exists()      # other video untouched
        assert (tmp_path / "0001_aaa_T.info.json").exists()  # other suffix untouched

    def test_cleanup_is_a_noop_on_a_missing_directory(self, tmp_path):
        from app.services.ingest_job_runner import _s3_cleanup_dir
        assert _s3_cleanup_dir(tmp_path / "nope") == 0


class TestFreeSpaceGuard:
    def test_raises_when_the_volume_is_too_full(self, tmp_path):
        import app.services.ingest_job_runner as runner
        usage = MagicMock(free=100)
        with patch("shutil.disk_usage", return_value=usage):
            with pytest.raises(RuntimeError, match="Insufficient disk space"):
                runner._assert_free_space(1_000_000, tmp_path)

    def test_passes_when_there_is_room(self, tmp_path):
        import app.services.ingest_job_runner as runner
        usage = MagicMock(free=10_000_000)
        with patch("shutil.disk_usage", return_value=usage):
            runner._assert_free_space(1_000_000, tmp_path)

    def test_unreadable_stat_never_blocks_a_job(self, tmp_path):
        import app.services.ingest_job_runner as runner
        with patch("shutil.disk_usage", side_effect=OSError("no stat")):
            runner._assert_free_space(1_000_000, tmp_path)


class TestLocalModeUnchanged:
    """The whole point of guarding on _ingest_use_s3: local runs must behave
    exactly as they did before rehydration existed."""

    @pytest.mark.parametrize("stage_name", [
        "_stage_extract_audio",
        "_stage_transcribe",
        "_stage_create_fragments",
        "_stage_classify",
        "_stage_embed",
    ])
    def test_no_s3_download_in_local_mode(self, stage_name):
        import app.services.ingest_job_runner as runner
        stage = getattr(runner, stage_name)
        with _stage_harness(use_s3=False) as (mock_download, _, _run):
            stage("job1", ["aaa"])
        mock_download.assert_not_called()

    @pytest.mark.parametrize("stage_name", [
        "_stage_extract_audio",
        "_stage_transcribe",
        "_stage_create_fragments",
        "_stage_classify",
        "_stage_embed",
    ])
    def test_no_deletion_in_local_mode(self, stage_name):
        """The critical one. In local mode these files are the source of truth,
        not a cache — deleting them destroys data that exists nowhere else."""
        import app.services.ingest_job_runner as runner
        stage = getattr(runner, stage_name)
        with _stage_harness(use_s3=False) as (_dl, mock_cleanup, _run):
            stage("job1", ["aaa"])
        mock_cleanup.assert_not_called()

    def test_stage_still_runs_the_script_exactly_once_in_local_mode(self):
        """Local mode is a single chunk, so nothing about the subprocess call
        or its logging changes."""
        from app.services.ingest_job_runner import _stage_extract_audio
        with _stage_harness(use_s3=False) as (_dl, _cl, mock_run):
            _stage_extract_audio("job1", ["a", "b", "c"])
        assert mock_run.call_count == 1
        assert mock_run.call_args.kwargs.get("truncate") is True


class TestDownloadArchiveIsNeverTouched:
    """data/metadata/youtube/download_archive.txt is yt-dlp's dedup record.
    Losing it re-downloads the whole channel — ~31GB onto a 30GiB volume."""

    def test_cleanup_targets_media_dirs_only(self):
        import inspect
        import app.services.ingest_job_runner as runner
        src = inspect.getsource(runner)
        calls = re.findall(r"_s3_cleanup_dir\((.*?)[,)]", src, re.S)
        # Guard against the regex silently matching nothing and passing vacuously.
        assert len(calls) >= 5, f"expected several cleanup sites, found {len(calls)}"
        for call in calls:
            assert "metadata" not in call, f"cleanup must never target metadata: {call}"

    def test_sweep_leaves_the_archive_in_place(self, tmp_path):
        from app.services.ingest_job_runner import _s3_cleanup_dir
        archive = tmp_path / "download_archive.txt"
        archive.write_text("youtube abc123\n")
        (tmp_path / "0001_aaa_T.mp4").write_bytes(b"x")

        _s3_cleanup_dir(tmp_path, suffixes=(".mp4",))

        assert archive.exists()
