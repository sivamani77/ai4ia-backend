from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
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
    # User authentication (JWT access + rotating refresh tokens)
    # ------------------------------------------------------------------
    jwt_signing_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "ai4ia-backend"
    access_token_ttl_seconds: int = 15 * 60          # 15 minutes
    refresh_token_ttl_seconds: int = 30 * 24 * 60 * 60  # 30 days
    refresh_cookie_name: str = "refresh_token"
    refresh_cookie_domain: str = ""   # blank = host-only cookie
    refresh_cookie_secure: bool = True

    # Argon2id password hashing cost parameters
    argon2_time_cost: int = 3
    argon2_memory_cost_kib: int = 64 * 1024   # 64 MiB
    argon2_parallelism: int = 2

    # OTP (email verification codes for signup + password reset)
    otp_ttl_minutes: int = 10
    otp_max_attempts: int = 5
    otp_resend_cooldown_seconds: int = 60

    # Login brute-force lockout
    login_max_failed_attempts: int = 5
    login_lockout_seconds: int = 15 * 60  # 15 minutes

    # ------------------------------------------------------------------
    # Mail (SMTP) — OTP delivery
    # ------------------------------------------------------------------
    mail_smtp_host: str = ""
    mail_smtp_port: int = 587
    mail_smtp_user: str = ""
    mail_smtp_password: str = ""
    mail_from_address: str = ""
    mail_from_name: str = "AI4IA (No Reply)"

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://ollama-mobius-sales.mobiusdtaas.ai"

    # ------------------------------------------------------------------
    # Redis / Celery
    # ------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Postgres — direct connection for the video-job pipeline tables
    # (jobs, job_events, agent_decision_records, llm_call_traces). Replaces
    # the Supabase REST client for these four tables: the REST/PostgREST
    # layer was the source of the "RemoteProtocolError: Server disconnected"
    # errors seen during long-running jobs, not the underlying database — a
    # pooled, persistent direct connection doesn't have that failure mode.
    # docker-compose.infra.yml's postgres service matches this default for
    # local testing.
    # ------------------------------------------------------------------
    database_url: str = "postgresql://ai4ia:ai4ia_local_dev@localhost:5432/ai4ia"

    # ------------------------------------------------------------------
    # Supabase — auth project only (app_users, otp_codes, refresh_tokens,
    # auth_audit). The pipeline/jobs project above has been replaced by
    # direct Postgres for the four tables listed above; supabase_url/
    # supabase_service_key are kept only in case any other pipeline table
    # (storylines, insights, fragments) still reads from that project.
    # ------------------------------------------------------------------
    supabase_url: str = ""
    supabase_service_key: str = ""

    # ------------------------------------------------------------------
    # Supabase — auth project (app_users, otp_codes, refresh_tokens,
    # auth_audit). This is a SEPARATE Supabase project from the one above —
    # app_users already existed there before the backend owned auth, and the
    # frontend's anon/publishable key for it must never be used server-side.
    # The backend needs that project's service-role key to hash/verify
    # passwords and manage tokens.
    # ------------------------------------------------------------------
    auth_supabase_url: str = ""
    auth_supabase_service_key: str = ""

    # ------------------------------------------------------------------
    # vLLM (unused by the video pipeline as of the gpt-oss:120b/Ollama
    # switch below; kept for rollback and because other features'
    # `*_model` settings still reference the same "openai/gpt-oss-20b"
    # identifier as a default value, independent of vLLM being reachable)
    # ------------------------------------------------------------------
    vllm_base_url: str = "http://vllm-gpt-oss.mobiusdtaas.ai"
    vllm_model: str = "openai/gpt-oss-20b"
    vllm_chat_endpoint: str = "/v1/chat/completions"

    # ------------------------------------------------------------------
    # Video-pipeline strong model (storyline generation + LLM editorial
    # validation in scripts/07_generate_storyline.py and
    # scripts/07b_validate_storyline_with_llm.py) — served by the same
    # Ollama host as embeddings, via /api/generate. Scoped to the video
    # pipeline only; other features keep their own model settings below.
    #
    # storyline_llm_backend switches these two scripts between Ollama
    # (default, unchanged) and a dedicated vLLM gpt-oss-120b deployment
    # (OpenAI-compatible /v1/chat/completions) — separate host/model from
    # the vllm_* settings above, which serve the smaller 20b model for
    # other features and are untouched by this flag.
    # ------------------------------------------------------------------
    storyline_strong_model: str = "gpt-oss:120b"
    storyline_llm_backend: str = "ollama"  # "ollama" | "vllm"
    storyline_vllm_base_url: str = "http://vllm-gpt-120.mobiusdtaas.ai"
    storyline_vllm_model: str = "openai/gpt-oss-120b"

    # ------------------------------------------------------------------
    # LLM models
    # ------------------------------------------------------------------
    speaker_extraction_model: str = "openai/gpt-oss-120b"
    insight_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "nomic-embed-text:v1.5"
    fast_llm_model: str = "llama3.1:8b"

    # ------------------------------------------------------------------
    # Response translation
    # ------------------------------------------------------------------
    translation_enabled: bool = False
    # Provider: "vllm" now; "deepl" / "libretranslate" adapters can be added
    # behind the same interface later.
    translation_provider: str = "vllm"
    # Languages clients may request (ISO 639-1). English is the source
    # language, so requesting it is a no-op.
    translation_supported_languages: list[str] = [
        "fr", "de", "es", "pt", "it", "nl", "hi", "ja", "zh",
    ]
    translation_cache_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days
    # Responses larger than this are returned untranslated (fail-open).
    translation_max_body_bytes: int = 200_000

    # ------------------------------------------------------------------
    # Storyline / export tuning
    # ------------------------------------------------------------------
    # Clip diversity caps — max times a single source video / speaker may
    # appear in one generated storyline. Enforced independently in the
    # storyline-generation stage and the video-export stage; must stay in
    # sync, hence a single shared source here.
    max_uses_per_source_video: int = Field(default=2, gt=0)
    max_uses_per_speaker: int = Field(default=2, gt=0)

    # Fragmenting rules — how transcript segments are grouped into fragments.
    fragment_min_duration_seconds: float = Field(default=35, gt=0)
    fragment_target_duration_seconds: float = Field(default=75, gt=0)
    fragment_max_duration_seconds: float = Field(default=130, gt=0)
    fragment_min_words: int = Field(default=35, gt=0)
    fragment_target_words: int = Field(default=120, gt=0)
    fragment_max_words: int = Field(default=230, gt=0)

    # ------------------------------------------------------------------
    # Storyline cache (app/services/storyline_cache.py) — skips stages
    # 07_storyline + 07b_validate_storyline entirely on a hit, since 07b's
    # output is what 08/09 actually consume.
    # ------------------------------------------------------------------
    # L1: exact match on (normalized prompt, target_duration, style,
    # validation_mode, visual_mode, max_validation_iterations, corpus_version).
    # Safe to leave on — zero false-positive risk.
    storyline_cache_enabled: bool = True
    storyline_cache_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days
    # Bump this after re-ingesting new source videos — it's part of the L1/L2
    # cache key, so every cached entry is implicitly invalidated by a version
    # change without needing a manual purge.
    storyline_cache_corpus_version: str = "v1"
    # L2: semantic/paraphrase match via embedding similarity + an LLM
    # confirmation gate on candidate hits. Off by default — a false-positive
    # hit here serves the wrong topic's video, not just a slow one, so this
    # should only be enabled once there's real traffic data to tune the
    # threshold against.
    storyline_cache_l2_enabled: bool = False
    # Calibrated against nomic-embed-text:v1.5 with the "classification: "
    # prefix (see storyline_cache._embed_text): true paraphrases scored
    # 0.87-0.91, clearly unrelated prompts scored 0.62-0.65, but a
    # same-theme-different-topic prompt ALSO scored ~0.87 — cosine
    # similarity alone cannot safely distinguish those two cases. This
    # threshold is set low enough to admit both as LLM-confirmation
    # candidates; _llm_confirms_same_topic is what actually decides,
    # not this number. Do not treat this as a precision knob on its own.
    storyline_cache_l2_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # Cap on how many L2 candidate embeddings to keep/scan (linear scan is
    # fine at this scale; this just bounds memory/latency as the cache grows).
    storyline_cache_l2_max_candidates: int = Field(default=2000, gt=0)

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

    # Progressive HLS streaming during export (stage 09) — off by default,
    # enable once the frontend player is ready to consume a growing playlist.
    ai4ia_enable_hls_streaming: bool = False

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

        if not (
            self.fragment_min_duration_seconds
            <= self.fragment_target_duration_seconds
            <= self.fragment_max_duration_seconds
        ):
            raise ValueError(
                "fragment duration thresholds must satisfy "
                "min <= target <= max "
                f"(got {self.fragment_min_duration_seconds}, "
                f"{self.fragment_target_duration_seconds}, {self.fragment_max_duration_seconds})"
            )
        if not (self.fragment_min_words <= self.fragment_target_words <= self.fragment_max_words):
            raise ValueError(
                "fragment word thresholds must satisfy min <= target <= max "
                f"(got {self.fragment_min_words}, {self.fragment_target_words}, {self.fragment_max_words})"
            )

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
        if not self.auth_supabase_url:
            warnings.append("AUTH_SUPABASE_URL is not set — user auth (app_users) unavailable")
        if not self.auth_supabase_service_key:
            warnings.append("AUTH_SUPABASE_SERVICE_KEY is not set — user auth (app_users) unavailable")
        if not self.api_key:
            warnings.append("API_KEY is not set — running in open dev mode, no auth enforced")
        if not self.jwt_signing_key:
            warnings.append("JWT_SIGNING_KEY is not set — user auth tokens cannot be issued")
        if not self.mail_smtp_host or not self.mail_smtp_user or not self.mail_smtp_password:
            warnings.append("MAIL_SMTP_* is not fully configured — OTP emails cannot be sent")
        return warnings

    model_config = {
        # Search for .env in cwd first, then one level up (repo root when running from backend/)
        "env_file": [".env", "../.env"],
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
