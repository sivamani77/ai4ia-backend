from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = "AI4IA Video Studio Backend"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    api_key: str = ""           # Set to enforce X-API-Key auth; blank = dev mode (no auth)
    allowed_origins: list[str] = ["*"]  # CORS allowed origins

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://ollama-mobius-sales.mobiusdtaas.ai"

    # ------------------------------------------------------------------
    # Redis / Celery
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Supabase
    # ------------------------------------------------------------------
    supabase_url: str = ""
    supabase_service_key: str = ""

    # ------------------------------------------------------------------
    # LLM models
    # ------------------------------------------------------------------
    speaker_extraction_model: str = "llama3.1:70b"
    insight_model: str = "llama3.1:70b"

    # ------------------------------------------------------------------
    # YouTube ingestion
    # ------------------------------------------------------------------
    youtube_channel_url: str = "https://www.youtube.com/@ai4iaconference861/videos"
    # Set to true only in deployed environments to enable periodic auto-sync
    enable_auto_sync: bool = False
    # Cron schedule for auto-sync (Celery beat format, UTC)
    auto_sync_cron: str = "0 2 * * *"  # 02:00 UTC daily

    # ------------------------------------------------------------------
    # Celery worker configuration (used by worker startup, not the API)
    # ------------------------------------------------------------------
    embed_celery_worker: bool = False
    worker_concurrency: int = 1
    worker_queue: str = "video_jobs"
    worker_loglevel: str = "info"

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    # chroma_mode: "local"  → PersistentClient (data/chroma/ on disk)
    #              "http"   → HttpClient (remote Chroma server, e.g. Azure)
    chroma_mode: str = "local"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_auth_token: str = ""        # set for bearer-token-protected deployments
    chroma_collection: str = "fragments"

    # ------------------------------------------------------------------
    # Storage backend
    # ------------------------------------------------------------------
    ai4ia_storage_backend: str = "local"

    # Object storage (placeholders — not yet implemented)
    ai4ia_object_storage_bucket: str = ""
    ai4ia_object_storage_endpoint: str = ""
    ai4ia_object_storage_region: str = ""
    ai4ia_object_storage_access_key: str = ""
    ai4ia_object_storage_secret_key: str = ""

    # ------------------------------------------------------------------
    # Project root — anchor for all derived paths.
    # Defaults to the repo root (four parents up from this file).
    # ------------------------------------------------------------------
    project_root: Path = Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------------
    # Derived paths — all computed from project_root in the validator
    # below.  Declared here so they are proper typed fields on the model.
    # ------------------------------------------------------------------
    ai4ia_local_exports_dir: Path = Path()
    ai4ia_local_assets_dir: Path = Path()
    ai4ia_local_jobs_dir: Path = Path()

    @model_validator(mode="after")
    def _set_path_defaults(self) -> "Settings":
        data_dir = self.project_root / "data"

        if self.ai4ia_local_exports_dir == Path():
            self.ai4ia_local_exports_dir = (data_dir / "exports").resolve()
        else:
            self.ai4ia_local_exports_dir = self.ai4ia_local_exports_dir.resolve()

        if self.ai4ia_local_assets_dir == Path():
            self.ai4ia_local_assets_dir = (data_dir / "generated_assets").resolve()
        else:
            self.ai4ia_local_assets_dir = self.ai4ia_local_assets_dir.resolve()

        if self.ai4ia_local_jobs_dir == Path():
            self.ai4ia_local_jobs_dir = (data_dir / "jobs").resolve()
        else:
            self.ai4ia_local_jobs_dir = self.ai4ia_local_jobs_dir.resolve()

        return self

    def validate_deployment_env(self) -> list[str]:
        warnings: list[str] = []
        if not self.redis_url:
            warnings.append(
                "REDIS_URL is not set — thread fallback active, jobs lost on server restart"
            )
        elif "localhost" in self.redis_url:
            warnings.append(
                "Redis is localhost — jobs will not survive restarts in a cloud deployment"
            )
        if not self.supabase_url:
            warnings.append("SUPABASE_URL is not set")
        if not self.supabase_service_key:
            warnings.append("SUPABASE_SERVICE_KEY is not set")
        if not self.api_key:
            warnings.append("API_KEY is not set — running in open dev mode, no auth enforced")
        return warnings

    model_config = {
        # Search for .env in cwd first, then one level up (repo root when running from backend/)
        "env_file": [".env", "../.env"],
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
