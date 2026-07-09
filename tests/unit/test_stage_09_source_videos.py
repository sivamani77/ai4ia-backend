"""Tests for stage_09_export's source-video download/verification.

Root cause under investigation: a deployed job hit
`ExportIntegrityError: ... source video not found` for a video confirmed
present in S3 (verified directly via head-object) and correctly matched by
the stem-substring logic when tested in isolation. The download call itself
had no verification and no failure surfaced if it silently didn't produce a
usable file, so any transient issue (partial download, disk space, a slow
network hiccup swallowed by boto3's default timeouts) would only be
discovered much later by 09's pre-render integrity check, with a less
actionable error. This hardens _download_source_videos to verify its own
output and fail immediately naming the specific source_video.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

import app.pipeline.stage_09_export as stage_09  # noqa: E402


class TestGetNeededSourceVideos:
    def test_collects_from_export_timeline(self):
        mdata = {"export_timeline": [{"source_video": "video_a"}, {"source_video": "video_b"}]}
        assert stage_09._get_needed_source_videos(mdata) == {"video_a", "video_b"}

    def test_collects_from_storyline_export_timeline(self):
        mdata = {"storyline": {"export_timeline": [{"source_video": "video_c"}]}}
        assert stage_09._get_needed_source_videos(mdata) == {"video_c"}

    def test_ignores_items_without_source_video(self):
        mdata = {"export_timeline": [{"type": "title_card"}, {"source_video": "video_a"}]}
        assert stage_09._get_needed_source_videos(mdata) == {"video_a"}

    def test_empty_manifest_returns_empty_set(self):
        assert stage_09._get_needed_source_videos({}) == set()


class TestDownloadSourceVideos:
    def test_no_needed_videos_is_a_noop(self, tmp_path):
        with patch.object(stage_09, "list_keys") as mock_list:
            stage_09._download_source_videos({}, tmp_path)
            mock_list.assert_not_called()

    def test_successful_download_verified(self, tmp_path, monkeypatch):
        mdata = {"export_timeline": [{"source_video": "0100_video"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/0100_video.mp4"])

        def fake_download(key, local):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"fake video content")
            return local

        monkeypatch.setattr(stage_09, "download", fake_download)
        stage_09._download_source_videos(mdata, tmp_path)
        assert (tmp_path / "0100_video.mp4").exists()

    def test_already_present_file_is_not_redownloaded(self, tmp_path, monkeypatch):
        local = tmp_path / "0100_video.mp4"
        local.write_bytes(b"already here")
        mdata = {"export_timeline": [{"source_video": "0100_video"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/0100_video.mp4"])

        calls = []
        monkeypatch.setattr(stage_09, "download", lambda key, local: calls.append(key))
        stage_09._download_source_videos(mdata, tmp_path)
        assert calls == []

    def test_no_matching_s3_key_raises_with_source_video_name(self, tmp_path, monkeypatch):
        mdata = {"export_timeline": [{"source_video": "0100_missing_video"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/other_video.mp4"])
        with pytest.raises(RuntimeError, match="0100_missing_video"):
            stage_09._download_source_videos(mdata, tmp_path)

    def test_download_that_produces_no_file_raises(self, tmp_path, monkeypatch):
        """Regression: previously a download() that silently failed to produce
        a usable local file left no trace until 09's pre-render check, much
        later, with a less actionable error."""
        mdata = {"export_timeline": [{"source_video": "0100_video"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/0100_video.mp4"])
        monkeypatch.setattr(stage_09, "download", lambda key, local: local)  # no-op: never writes the file
        with pytest.raises(RuntimeError, match="0100_video"):
            stage_09._download_source_videos(mdata, tmp_path)

    def test_download_that_produces_empty_file_raises(self, tmp_path, monkeypatch):
        mdata = {"export_timeline": [{"source_video": "0100_video"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/0100_video.mp4"])

        def fake_download(key, local):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch()  # zero-byte file
            return local

        monkeypatch.setattr(stage_09, "download", fake_download)
        with pytest.raises(RuntimeError, match="did not produce a usable file"):
            stage_09._download_source_videos(mdata, tmp_path)

    def test_multiple_failures_all_named_in_one_error(self, tmp_path, monkeypatch):
        mdata = {"export_timeline": [{"source_video": "video_a"}, {"source_video": "video_b"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: [])
        with pytest.raises(RuntimeError) as exc_info:
            stage_09._download_source_videos(mdata, tmp_path)
        assert "video_a" in str(exc_info.value)
        assert "video_b" in str(exc_info.value)

    def test_fullwidth_colon_title_matches_correctly(self, tmp_path, monkeypatch):
        """Regression check: source_video strings containing a fullwidth colon
        (：, U+FF1A, common in these video titles) must still substring-match
        the S3 key correctly."""
        title = "0100_ikdcwVcRSPo_Jamaica Satellite Event - Generative AI： Impact on Misinformation & Disinformation"
        mdata = {"export_timeline": [{"source_video": title}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: [f"raw/youtube/ai4ia_channel/{title}.mp4"])

        def fake_download(key, local):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"content")
            return local

        monkeypatch.setattr(stage_09, "download", fake_download)
        stage_09._download_source_videos(mdata, tmp_path)
        assert (tmp_path / f"{title}.mp4").exists()

    def test_id_prefix_fallback_when_title_punctuation_was_normalized(self, tmp_path, monkeypatch):
        """Regression: a real deployed failure. The manifest's source_video had
        ASCII punctuation ('(6N)_:' ... 'XYNOGENY?') while the actual S3 filename
        kept the original fullwidth Unicode punctuation ('(6N)：' ... 'XYNOGENY？')
        — some LLM round-trip (storyline generation or narrative assembly)
        silently normalized the title text. The exact-substring match fails, but
        the stable '0254_RXXzSGoo0KQ' id prefix is untouched and must resolve it.
        """
        requested = "0254_RXXzSGoo0KQ_(6N)_: C Nelson - XYNOGENY? How We Share Decision-Making"
        real_filename = "0254_RXXzSGoo0KQ_(6N)： C Nelson - XYNOGENY？ How We Share Decision-Making.mp4"
        mdata = {"export_timeline": [{"source_video": requested}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: [f"raw/youtube/ai4ia_channel/{real_filename}"])

        def fake_download(key, local):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"content")
            return local

        monkeypatch.setattr(stage_09, "download", fake_download)
        stage_09._download_source_videos(mdata, tmp_path)  # must not raise
        assert (tmp_path / real_filename).exists()

    def test_video_id_prefix_extraction(self, stage_09_mod=stage_09):
        assert stage_09._video_id_prefix("0254_RXXzSGoo0KQ_(6N)： C Nelson") == "0254_RXXzSGoo0KQ_"
        assert stage_09._video_id_prefix("not_a_valid_prefix") is None

    def test_no_matching_prefix_still_raises(self, tmp_path, monkeypatch):
        mdata = {"export_timeline": [{"source_video": "0999_totallyMissing_Some Title"}]}
        monkeypatch.setattr(stage_09, "list_keys", lambda prefix: ["raw/youtube/ai4ia_channel/0254_other_video.mp4"])
        with pytest.raises(RuntimeError, match="0999_totallyMissing"):
            stage_09._download_source_videos(mdata, tmp_path)


class TestExportTimeout:
    """Stage 09 render+transcribe timeout must scale with requested duration so a
    legitimate long CPU render isn't killed mid-encode (regression: a 600s job
    hit the old flat 900s ceiling and failed)."""

    def test_short_requests_get_floor(self):
        assert stage_09._export_timeout_for(0) == 1800
        assert stage_09._export_timeout_for(120) == 1800

    def test_scales_with_duration(self):
        assert stage_09._export_timeout_for(600) == 4200  # 600 + 600*6

    def test_ten_minute_request_exceeds_old_900s_ceiling(self):
        # The exact regression: 600s target must allow far more than 900s.
        assert stage_09._export_timeout_for(600) > 900


class TestS3ClientTimeouts:
    def test_s3_client_configured_with_generous_timeouts(self, monkeypatch):
        """Raw videos run to ~1GB; boto3's default 60s read timeout and
        minimal retries are too tight for that over a shared network link."""
        import app.services.storage_service as storage_service

        captured = {}

        class FakeBoto3:
            @staticmethod
            def client(service, **kwargs):
                captured.update(kwargs)
                return object()

        monkeypatch.setattr(storage_service, "_OBJECT_STORAGE_ACCESS_KEY", "key")
        monkeypatch.setattr(storage_service, "_OBJECT_STORAGE_SECRET_KEY", "secret")
        monkeypatch.setattr(storage_service, "_OBJECT_STORAGE_REGION", "us-east-1")
        monkeypatch.setattr(storage_service, "_OBJECT_STORAGE_ENDPOINT", "")
        monkeypatch.setattr(storage_service, "_OBJECT_STORAGE_BUCKET", "bucket")

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "boto3", FakeBoto3)

        storage_service.S3StorageService()
        assert "config" in captured
        cfg = captured["config"]
        assert cfg.read_timeout >= 300
        assert cfg.connect_timeout >= 30
