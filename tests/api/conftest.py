from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_app(mock_settings, mock_job_store, monkeypatch):
    """Boot the FastAPI app with safe in-memory job storage."""
    try:
        import pythonjsonlogger.json  # noqa: F401
    except ModuleNotFoundError:
        pkg = types.ModuleType("pythonjsonlogger")
        sub = types.ModuleType("pythonjsonlogger.json")

        class _JsonFormatter:  # pragma: no cover - tiny compatibility shim
            def __init__(self, *args, **kwargs):
                pass

            def format(self, record):  # noqa: D401
                return record.getMessage()

        sub.JsonFormatter = _JsonFormatter
        pkg.json = sub
        sys.modules.setdefault("pythonjsonlogger", pkg)
        sys.modules.setdefault("pythonjsonlogger.json", sub)

    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        chroma = types.ModuleType("chromadb")

        class _DummyClient:
            def get_collection(self, *args, **kwargs):
                raise RuntimeError("chromadb stub")

        class _DummySettings:
            def __init__(self, *args, **kwargs):
                pass

        chroma.PersistentClient = lambda *args, **kwargs: _DummyClient()
        chroma.HttpClient = lambda *args, **kwargs: _DummyClient()
        config = types.ModuleType("chromadb.config")
        config.Settings = _DummySettings
        chroma.config = config
        sys.modules.setdefault("chromadb", chroma)
        sys.modules.setdefault("chromadb.config", config)

    import app.main as main
    import app.api.video_jobs as video_jobs

    monkeypatch.setattr(main, "_reap_orphaned_jobs", lambda: None)
    monkeypatch.setattr(main, "_reap_orphaned_ingest_jobs", lambda: None)
    monkeypatch.setattr(video_jobs, "_create_limiter", video_jobs.TokenBucket(rate=5, per=60.0))

    for path in (
        mock_settings.ai4ia_local_exports_dir,
        mock_settings.ai4ia_local_assets_dir,
        mock_settings.ai4ia_local_jobs_dir,
        mock_settings.project_root / "data" / "raw" / "youtube" / "ai4ia_channel",
        mock_settings.project_root / "logs",
        mock_settings.project_root / "temp",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    main.app.state.mock_settings = mock_settings
    return main.app


@pytest.fixture
def client(api_app):
    with TestClient(api_app) as test_client:
        yield test_client
