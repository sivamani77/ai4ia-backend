from __future__ import annotations

import copy
import shutil
from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]


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

    # app.core.taxonomy_loader reads config/taxonomy.json under settings.project_root
    # at import time (module-level, fails fast if missing). mock_settings points
    # project_root at a throwaway tmp_path, so make sure the real taxonomy config
    # is there before the first `import app.main` triggers that load. Only matters
    # when this is the first test in the session to import app.main — once
    # taxonomy_loader has run, its module-level globals are cached for the process.
    taxonomy_dst = mock_settings.project_root / "config" / "taxonomy.json"
    taxonomy_dst.parent.mkdir(parents=True, exist_ok=True)
    if not taxonomy_dst.exists():
        shutil.copyfile(_REPO_ROOT / "config" / "taxonomy.json", taxonomy_dst)

    import app.main as main
    import app.api.video_jobs as video_jobs
    import app.api.video_media as video_media
    import app.services.storage_service as storage_service
    import app.services.media_registry as media_registry_mod
    import app.core.paths as paths_mod
    import app.api.archive as archive_mod
    import app.api.deps as deps_mod

    monkeypatch.setattr(main, "_reap_orphaned_jobs", lambda: None)
    monkeypatch.setattr(main, "_reap_orphaned_ingest_jobs", lambda: None)
    monkeypatch.setattr(video_jobs, "_create_limiter", video_jobs.TokenBucket(rate=5, per=60.0))

    # app.api.video_jobs did `from app.services.video_job_runner import load_job,
    # list_jobs, load_events` and `from app.services.job_store import
    # load_events_after` — plain names bound once, at whichever test first
    # triggers `import app.main` in this process. mock_job_store only patches
    # the *source* modules (app.services.job_store / video_job_runner), which
    # doesn't reach those already-bound copies for every test after the first.
    # Rebind them here, every test, straight onto this test's own
    # mock_job_store dicts so GET/list/events routes see what this test seeded.
    def _vj_load_job(job_id):
        if job_id not in mock_job_store["jobs"]:
            raise FileNotFoundError(f"Job not found: {job_id}")
        return copy.deepcopy(mock_job_store["jobs"][job_id])

    def _vj_list_jobs():
        return [copy.deepcopy(j) for j in mock_job_store["jobs"].values()]

    def _vj_load_events(job_id):
        return list(mock_job_store["events"].get(job_id, []))

    def _vj_load_events_after(job_id, after_id=0):
        return [e for e in mock_job_store["events"].get(job_id, []) if e["id"] > after_id]

    monkeypatch.setattr(video_jobs, "load_job", _vj_load_job)
    monkeypatch.setattr(video_jobs, "list_jobs", _vj_list_jobs)
    monkeypatch.setattr(video_jobs, "load_events", _vj_load_events)
    monkeypatch.setattr(video_jobs, "load_events_after", _vj_load_events_after)
    # app.api.video_media has the same `from app.services.video_job_runner
    # import load_job` staleness issue — it serves /video-jobs/{id}/video/*,
    # /assets/*, /manifest, /media, /thumbnail, /subtitles/* etc.
    monkeypatch.setattr(video_media, "load_job", _vj_load_job)

    # app.services.storage_service computes several path constants from
    # `settings.*` at *module import time* (STORAGE_BACKEND, _LOCAL_EXPORTS_DIR,
    # _LOCAL_ASSETS_DIR, _LOCAL_JOBS_DIR, ALLOWED_MEDIA_ROOTS) and freezes them
    # into module globals — captured once, at whichever test first imports the
    # module in this process. Every later test's mock_settings.project_root
    # (a fresh tmp_path) is then ignored by any registry/asset lookup, so
    # media registries written under the *current* test's tmp_path are
    # invisible to code reading via these stale constants. Rebind them to the
    # current test's mock_settings every test.
    monkeypatch.setattr(storage_service, "STORAGE_BACKEND", mock_settings.ai4ia_storage_backend)
    monkeypatch.setattr(storage_service, "_LOCAL_EXPORTS_DIR", mock_settings.ai4ia_local_exports_dir)
    monkeypatch.setattr(storage_service, "_LOCAL_ASSETS_DIR", mock_settings.ai4ia_local_assets_dir)
    monkeypatch.setattr(storage_service, "_LOCAL_JOBS_DIR", mock_settings.ai4ia_local_jobs_dir)
    monkeypatch.setattr(storage_service, "ALLOWED_MEDIA_ROOTS", [
        mock_settings.ai4ia_local_exports_dir,
        mock_settings.ai4ia_local_assets_dir,
        mock_settings.ai4ia_local_jobs_dir,
    ])
    # media_registry.py separately did `from app.services.storage_service
    # import _LOCAL_JOBS_DIR` — its own frozen copy, needs its own rebind.
    monkeypatch.setattr(media_registry_mod, "_LOCAL_JOBS_DIR", mock_settings.ai4ia_local_jobs_dir)

    # app.core.paths has the identical frozen-at-import-time problem for its
    # whole set of directory constants (PROJECT_ROOT, DATA_DIR, RAW_VIDEO_DIR,
    # etc. — all derived from settings.project_root once, at first import).
    # Many api/* and services/* modules do `from app.core.paths import X`
    # directly, so rebind the constants here, recomputed the same way
    # paths.py itself does, from this test's mock_settings.project_root.
    _proj_root = mock_settings.project_root
    _data_dir = _proj_root / "data"
    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", _proj_root)
    monkeypatch.setattr(paths_mod, "BACKEND_DIR", _proj_root / "backend")
    monkeypatch.setattr(paths_mod, "SCRIPTS_DIR", _proj_root / "scripts")
    monkeypatch.setattr(paths_mod, "CONFIG_DIR", _proj_root / "config")
    monkeypatch.setattr(paths_mod, "DATA_DIR", _data_dir)
    monkeypatch.setattr(paths_mod, "LOGS_DIR", _proj_root / "logs")
    monkeypatch.setattr(paths_mod, "TEMP_DIR", _proj_root / "temp")
    monkeypatch.setattr(paths_mod, "RUNTIME_DIR", _proj_root / "backend" / "runtime")
    monkeypatch.setattr(paths_mod, "RAW_VIDEO_DIR", _data_dir / "raw" / "youtube" / "ai4ia_channel")
    monkeypatch.setattr(paths_mod, "AUDIO_DIR", _data_dir / "audio")
    monkeypatch.setattr(paths_mod, "TRANSCRIPTS_DIR", _data_dir / "transcripts")
    monkeypatch.setattr(paths_mod, "FRAGMENTS_DIR", _data_dir / "fragments")
    monkeypatch.setattr(paths_mod, "CLASSIFIED_FRAGMENTS_DIR", _data_dir / "fragments_classified")
    monkeypatch.setattr(paths_mod, "EMBEDDINGS_DIR", _data_dir / "embeddings")
    monkeypatch.setattr(paths_mod, "CHROMA_DIR", _data_dir / "chroma")
    monkeypatch.setattr(paths_mod, "STORYLINES_DIR", _data_dir / "storylines")
    monkeypatch.setattr(paths_mod, "INSIGHTS_DIR", _data_dir / "insights")
    monkeypatch.setattr(paths_mod, "GENERATED_ASSETS_DIR", mock_settings.ai4ia_local_assets_dir)
    monkeypatch.setattr(paths_mod, "EXPORTS_DIR", mock_settings.ai4ia_local_exports_dir)
    monkeypatch.setattr(paths_mod, "METADATA_DIR", _data_dir / "metadata" / "youtube")
    monkeypatch.setattr(paths_mod, "SPEAKERS_DIR", _data_dir / "metadata" / "speakers")
    monkeypatch.setattr(paths_mod, "LATEST_ASSET_MANIFEST", mock_settings.ai4ia_local_assets_dir / "latest_asset_manifest.json")
    # app.api.archive did `from app.core.paths import RAW_VIDEO_DIR` — its own
    # frozen copy, same staleness pattern as everything else above.
    monkeypatch.setattr(archive_mod, "RAW_VIDEO_DIR", _data_dir / "raw" / "youtube" / "ai4ia_channel")
    # archive.py also did `from app.core.settings import settings` — a name
    # bound once, at whichever test first imports it in this process, to
    # THAT test's mock_settings object. Since later tests get their own,
    # separate mock_settings instance, mutations like `mock_settings.
    # ai4ia_storage_backend = "s3"` in one test can otherwise "leak" as a
    # frozen value into every later test that never touches storage_backend
    # itself. Rebind straight onto this test's own mock_settings, every time.
    monkeypatch.setattr(archive_mod, "settings", mock_settings)
    # app.api.deps did `from app.core.settings import settings` too — and
    # unlike everything else above, some unit tests (e.g.
    # test_require_models_dep.py) do `from app.api.deps import
    # require_models` at MODULE level, so this binding is captured during
    # pytest collection itself, before any fixture (mock_settings included)
    # ever runs — permanently frozen to the real, unmocked settings
    # singleton (model_gate_enabled defaults True there) unless rebound
    # here every test.
    monkeypatch.setattr(deps_mod, "settings", mock_settings)

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
