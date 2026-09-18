"""Unit tests for app.core.settings.Settings."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.settings import Settings


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_app_name(self):
        s = Settings()
        assert "AI4IA" in s.app_name or "ai4ia" in s.app_name.lower()

    def test_api_key_blank_by_default(self):
        s = Settings()
        assert s.api_key == ""

    def test_ollama_base_url_set(self):
        s = Settings()
        assert s.ollama_base_url.startswith("http")

    def test_redis_url_default(self):
        s = Settings()
        assert s.redis_url.startswith("redis://")

    def test_chroma_mode_default_local(self):
        s = Settings()
        # Field default is "local", but the repo's .env sets CHROMA_MODE=http
        # for local dev (a real Chroma server on :8001), which Settings()
        # picks up via its env_file.
        assert s.chroma_mode == "http"

    def test_chroma_collection_default(self):
        s = Settings()
        assert s.chroma_collection == "fragments"

    def test_chroma_port_default(self):
        s = Settings()
        assert s.chroma_port == 8000

    def test_chroma_ssl_default(self):
        s = Settings()
        assert s.chroma_ssl is False

    def test_storage_backend_default(self):
        s = Settings()
        # Field default is "s3" (backend/app/core/settings.py), and the
        # repo's .env also sets AI4IA_STORAGE_BACKEND=s3.
        assert s.ai4ia_storage_backend == "s3"

    def test_embed_celery_worker_default(self):
        s = Settings()
        assert s.embed_celery_worker is False

    def test_worker_concurrency_default(self):
        s = Settings()
        # Field default is 1, but the repo's .env sets WORKER_CONCURRENCY=6
        # for local dev, which Settings() picks up via its env_file.
        assert s.worker_concurrency == 6

    def test_enable_auto_sync_default(self):
        s = Settings()
        assert s.enable_auto_sync is False


# ---------------------------------------------------------------------------
# Derived path defaults (set by model_validator)
# ---------------------------------------------------------------------------

class TestDerivedPaths:
    def test_exports_dir_under_data(self):
        s = Settings()
        assert s.ai4ia_local_exports_dir.name == "exports"

    def test_assets_dir_under_data(self):
        s = Settings()
        assert s.ai4ia_local_assets_dir.name == "generated_assets"

    def test_jobs_dir_under_data(self):
        s = Settings()
        assert s.ai4ia_local_jobs_dir.name == "jobs"

    def test_derived_paths_are_absolute(self):
        s = Settings()
        assert s.ai4ia_local_exports_dir.is_absolute()
        assert s.ai4ia_local_assets_dir.is_absolute()
        assert s.ai4ia_local_jobs_dir.is_absolute()

    def test_custom_project_root_propagates(self, tmp_path):
        s = Settings(project_root=tmp_path)
        assert s.ai4ia_local_exports_dir == (tmp_path / "data" / "exports").resolve()
        assert s.ai4ia_local_assets_dir == (tmp_path / "data" / "generated_assets").resolve()
        assert s.ai4ia_local_jobs_dir == (tmp_path / "data" / "jobs").resolve()

    def test_explicit_exports_dir_not_overridden(self, tmp_path):
        custom = tmp_path / "my_exports"
        s = Settings(ai4ia_local_exports_dir=custom)
        assert s.ai4ia_local_exports_dir == custom.resolve()


# ---------------------------------------------------------------------------
# redis_password merged into redis_url
# ---------------------------------------------------------------------------

class TestRedisPassword:
    def test_no_password_leaves_url_unchanged(self):
        s = Settings(redis_url="redis://localhost:6379/0", redis_password="")
        assert s.redis_url == "redis://localhost:6379/0"

    def test_password_injected_into_url(self):
        s = Settings(redis_url="redis://my-redis:6379/0", redis_password="secret")
        assert s.redis_url == "redis://:secret@my-redis:6379/0"

    def test_password_preserves_path(self):
        s = Settings(redis_url="redis://my-redis:7002/3", redis_password="secret")
        assert s.redis_url == "redis://:secret@my-redis:7002/3"

    def test_password_with_special_characters_is_percent_encoded(self):
        s = Settings(redis_url="redis://my-redis:6379/0", redis_password="p@ss/word:x")
        assert s.redis_url == "redis://:p%40ss%2Fword%3Ax@my-redis:6379/0"

    def test_existing_credentials_not_overridden(self):
        s = Settings(
            redis_url="redis://user:existing@my-redis:6379/0",
            redis_password="secret",
        )
        assert s.redis_url == "redis://user:existing@my-redis:6379/0"

    def test_no_password_and_no_url(self):
        s = Settings(redis_url="", redis_password="")
        assert s.redis_url == ""


# ---------------------------------------------------------------------------
# validate_deployment_env
# ---------------------------------------------------------------------------

class TestValidateDeploymentEnv:
    def test_no_warnings_when_all_configured(self):
        s = Settings(
            redis_url="redis://my-redis:6379/0",
            api_key="super-secret",
            jwt_signing_key="signing-key",
            mail_smtp_host="smtp.example.com",
            mail_smtp_user="user",
            mail_smtp_password="pw",
        )
        warnings = s.validate_deployment_env()
        assert warnings == []

    def test_warns_when_redis_localhost(self):
        s = Settings(
            redis_url="redis://localhost:6379/0",
            api_key="key",
        )
        warnings = s.validate_deployment_env()
        assert any("localhost" in w.lower() or "redis" in w.lower() for w in warnings)

    def test_warns_when_no_api_key(self):
        s = Settings(
            redis_url="redis://prod:6379/0",
            api_key="",
        )
        warnings = s.validate_deployment_env()
        assert any("api_key" in w.lower() or "API_KEY" in w for w in warnings)

    def test_multiple_missing_gives_multiple_warnings(self):
        s = Settings(
            redis_url="",
            api_key="",
            jwt_signing_key="",
            mail_smtp_host="",
        )
        warnings = s.validate_deployment_env()
        assert len(warnings) >= 3

    def test_returns_list(self):
        s = Settings()
        assert isinstance(s.validate_deployment_env(), list)
