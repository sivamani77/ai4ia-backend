"""Unit tests for app.services.storage_service."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestStorageObject:
    def _make_obj(self, **overrides):
        from app.services.storage_service import StorageObject
        defaults = dict(
            object_id="obj_001",
            job_id="vjob_test",
            kind="video",
            storage_backend="local",
            content_type="video/mp4",
            filename="output.mp4",
        )
        defaults.update(overrides)
        return StorageObject(**defaults)

    def test_to_dict_returns_dict(self):
        d = self._make_obj().to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_object_id(self):
        d = self._make_obj(object_id="special_id").to_dict()
        assert d["object_id"] == "special_id"

    def test_to_dict_has_content_type(self):
        d = self._make_obj(content_type="application/json").to_dict()
        assert d["content_type"] == "application/json"

    def test_to_dict_has_filename(self):
        d = self._make_obj(filename="clip.mp4").to_dict()
        assert d["filename"] == "clip.mp4"


class TestSafePath:
    def test_valid_allowed_path_returns_path(self, tmp_path):
        from app.services.storage_service import _safe_path, ALLOWED_MEDIA_ROOTS
        # Add tmp_path to ALLOWED_MEDIA_ROOTS so _safe_path will accept it
        p = tmp_path / "some_file.mp4"
        with patch("app.services.storage_service.ALLOWED_MEDIA_ROOTS", [tmp_path]):
            result = _safe_path(str(p))
        assert result is not None

    def test_path_outside_allowed_roots_returns_none(self, tmp_path):
        from app.services.storage_service import _safe_path
        # /etc/passwd should not be in any allowed root
        with patch("app.services.storage_service.ALLOWED_MEDIA_ROOTS", [tmp_path]):
            result = _safe_path("/etc/passwd")
        assert result is None

    def test_invalid_path_returns_none(self):
        from app.services.storage_service import _safe_path
        with patch("app.services.storage_service.ALLOWED_MEDIA_ROOTS", []):
            result = _safe_path("")
        assert result is None


class TestGuessContentType:
    def test_json_extension(self):
        from app.services.storage_service import _guess_content_type
        assert _guess_content_type(Path("file.json")) == "application/json"

    def test_mp4_extension(self):
        from app.services.storage_service import _guess_content_type
        assert _guess_content_type(Path("video.mp4")) == "video/mp4"

    def test_unknown_extension_returns_octet_stream(self):
        from app.services.storage_service import _guess_content_type
        ct = _guess_content_type(Path("file.xyzunknown123"))
        assert ct == "application/octet-stream"

    def test_wav_extension(self):
        from app.services.storage_service import _guess_content_type
        ct = _guess_content_type(Path("audio.wav"))
        assert "wav" in ct  # audio/wav or audio/x-wav

    def test_srt_extension(self):
        from app.services.storage_service import _guess_content_type
        ct = _guess_content_type(Path("captions.srt"))
        assert ct in ("application/x-subrip", "text/plain", "application/octet-stream")


class TestSaveLoadRegistry:
    def test_round_trip(self, tmp_path):
        from app.services.storage_service import save_registry, load_registry
        reg = {"objects": {"f1": {"kind": "video", "local_path": "/data/f1.mp4"}}}
        with patch("app.services.storage_service._LOCAL_JOBS_DIR", tmp_path):
            save_registry("vjob_test", reg)
            loaded = load_registry("vjob_test")
        assert loaded == reg

    def test_missing_raises_file_not_found(self, tmp_path):
        from app.services.storage_service import load_registry
        with patch("app.services.storage_service._LOCAL_JOBS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                load_registry("vjob_nonexistent")

    def test_saves_as_json(self, tmp_path):
        from app.services.storage_service import save_registry
        reg = {"objects": {}}
        with patch("app.services.storage_service._LOCAL_JOBS_DIR", tmp_path):
            save_registry("vjob_j1", reg)
        saved_path = tmp_path / "vjob_j1" / "media_registry.json"
        raw = json.loads(saved_path.read_text())
        assert raw == {"objects": {}}


class TestLocalStorageService:
    def _make_obj(self, **overrides):
        from app.services.storage_service import StorageObject
        defaults = dict(
            object_id="obj_001",
            job_id="vjob_test",
            kind="video",
            storage_backend="local",
            content_type="video/mp4",
            filename="output.mp4",
        )
        defaults.update(overrides)
        return StorageObject(**defaults)

    def test_exists_true(self, tmp_path):
        from app.services.storage_service import LocalStorageService
        svc = LocalStorageService()
        (tmp_path / "file.mp4").write_bytes(b"data")
        obj = self._make_obj(local_path=str(tmp_path / "file.mp4"))
        with patch("app.services.storage_service.ALLOWED_MEDIA_ROOTS", [tmp_path]):
            result = svc.exists(obj)
        assert result is True

    def test_exists_false_no_local_path(self, tmp_path):
        from app.services.storage_service import LocalStorageService
        svc = LocalStorageService()
        obj = self._make_obj(local_path=None)
        assert svc.exists(obj) is False

    def test_get_local_path_returns_none_without_path(self, tmp_path):
        from app.services.storage_service import LocalStorageService
        svc = LocalStorageService()
        obj = self._make_obj(local_path=None)
        result = svc.get_local_path(obj)
        assert result is None

    def test_open_stream_raises_when_no_local_path(self):
        from app.services.storage_service import LocalStorageService
        svc = LocalStorageService()
        obj = self._make_obj(local_path=None)
        with pytest.raises(FileNotFoundError):
            list(svc.open_stream(obj))


class TestGetStorageService:
    def test_returns_local_service(self):
        import app.services.storage_service as ss
        from app.services.storage_service import LocalStorageService
        # Reset singleton
        ss._service_instance = None
        with patch.object(ss, "STORAGE_BACKEND", "local"):
            svc = ss.get_storage_service()
        assert isinstance(svc, LocalStorageService)
        ss._service_instance = None

    def test_caches_instance(self):
        import app.services.storage_service as ss
        ss._service_instance = None
        with patch.object(ss, "STORAGE_BACKEND", "local"):
            svc1 = ss.get_storage_service()
            svc2 = ss.get_storage_service()
        assert svc1 is svc2
        ss._service_instance = None
