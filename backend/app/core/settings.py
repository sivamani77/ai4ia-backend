from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

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
    # Set when Redis requires auth and the password isn't already embedded
    # in redis_url. Injected into redis_url by _apply_redis_password below;
    # leave unset and redis_url is used as-is (unauthenticated, as before).
    redis_password: str = ""

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
    # Supabase — pipeline project, LEGACY. Every pipeline table (fragments,
    # storylines, insights, ingest_jobs, ingest_status, ingest_job_events)
    # now lives in Postgres (app.db.schema.ensure_pipeline_schema) plus
    # Chroma for fragment vectors. These two settings are kept only so the
    # repositories can fall back to reading rows that predate the Postgres
    # cutover — nothing writes through this project anymore. Safe to leave
    # unset once a cluster's Postgres holds all pipeline data.
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
    vllm_base_url: str = "http://vllm-gpt-120.mobiusdtaas.ai"
    vllm_model: str = "openai/gpt-oss-120b"
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
    storyline_llm_backend: str = "vllm"  # "ollama" | "vllm"
    storyline_vllm_base_url: str = "http://vllm-gpt-120.mobiusdtaas.ai"
    storyline_vllm_model: str = "openai/gpt-oss-120b"

    # ------------------------------------------------------------------
    # LLM models
    # ------------------------------------------------------------------
    speaker_extraction_model: str = "openai/gpt-oss-120b"
    insight_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "nomic-embed-text:v1.5"
    fast_llm_model: str = "llama3.1:8b"

    # How long a recorded model-host outcome stays meaningful. Past this the
    # host reports "unknown" rather than a stale up/down — see
    # app/core/dependency_state.py. Deliberately generous: the model hosts
    # are scale-to-zero, so a quiet period is normal, not a fault, and the
    # cure for "unknown" is never to start polling them.
    llm_health_stale_after_seconds: int = 900

    # ------------------------------------------------------------------
    # Model gate (app/services/model_gate.py) — targets Ollama (embeddings)
    # and the 120b vLLM host only; the 20b deployment is unused. Unlike
    # dependency_state above, the gate DOES probe the hosts directly (a
    # feature request needs a true answer, not "nobody has called it
    # recently") — but only ever on a real login or a real feature request,
    # never on a timer, so the scale-to-zero idle window is still reachable.
    # ------------------------------------------------------------------
    model_gate_enabled: bool = True
    model_warmup_on_login: bool = True
    # Caches a real probe result for this long — a stampede guard so N
    # concurrent requests don't each send their own probe, not a substitute
    # for the probe. Must stay well under the GPU's scale-to-zero idle
    # timeout, or the gate can wave a request through to a host that just
    # disappeared.
    model_probe_cache_seconds: int = 15
    # How long an interactive request (insights/relay/compare) blocks for a
    # cold model before returning 503 + Retry-After. Video jobs use a much
    # longer wait_seconds directly against model_gate, not this setting.
    model_gate_inline_wait_seconds: int = 10
    # Overall deadline for the exponential-backoff warmer before it gives up
    # and records state="error". A 120b cold start is minutes, not seconds.
    model_warm_deadline_seconds: int = 600
    # Distributed lock (SET NX EX) so two API pods don't both hammer a cold
    # GPU with duplicate warm attempts for the same target.
    model_warm_lock_seconds: int = 300
    # Ollama keep_alive on the warm-up embed call — keeps the model resident
    # for this long after the call so the next real embed doesn't cold-load.
    ollama_keep_alive: str = "5m"
    ollama_warm_models: list[str] = ["nomic-embed-text:v1.5"]
    # Timeout for the warmer's confirmation embed call (the one that forces
    # the model off disk into memory after /api/pull). Deliberately much
    # longer than is_ready()'s fast 5s check — a cold host waking from
    # scale-to-zero routinely needs more than 5s to actually load the model,
    # and a too-short timeout here just makes the warmer retry (and pile up
    # duplicate in-flight load requests) instead of waiting for the one
    # already in progress to finish.
    model_embedding_warm_timeout_seconds: int = 45

    # ------------------------------------------------------------------
    # Prompt guardrail — runs as the very first pipeline stage, before any
    # other work starts, and rejects harmful/abusive/illegal/off-topic
    # prompts via an LLM classification call (app/pipeline/stage_guardrail.py).
    # ------------------------------------------------------------------
    prompt_guardrail_enabled: bool = True
    # The 120b vLLM model already serving insights/storyline. It is far more
    # reliable than the fast 8b model on the off_topic and prompt-extraction
    # cases, which are the ones that actually fire here, and the host is
    # already warm for the rest of the pipeline. Must stay a name present in
    # llm_service._VLLM_BASE_URL_BY_MODEL, or generate_json will route the
    # call to Ollama with a model Ollama does not have.
    prompt_guardrail_model: str = "openai/gpt-oss-120b"
    # If the LLM call itself fails (timeout, unparsable output), allow the
    # prompt through rather than blocking legitimate users on infra failure.
    prompt_guardrail_fail_open: bool = True

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
    ai4ia_storage_backend: str = "s3"

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
    def _apply_redis_password(self) -> "Settings":
        """Inject redis_password into redis_url, unless a password is already embedded.

        Lets deployments supply REDIS_URL and REDIS_PASSWORD as separate env
        vars/secrets instead of hand-assembling one auth URL. No-op when
        redis_password is unset, so unauthenticated Redis behaves exactly
        as before.
        """
        if not self.redis_password or not self.redis_url:
            return self

        parts = urlsplit(self.redis_url)
        if "@" in parts.netloc:
            return self  # credentials already embedded — leave as-is

        host = f"[{parts.hostname}]" if parts.hostname and ":" in parts.hostname else (parts.hostname or "")
        netloc = f":{quote(self.redis_password, safe='')}@{host}"
        if parts.port:
            netloc += f":{parts.port}"
        self.redis_url = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        return self

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
        # SUPABASE_URL/SUPABASE_SERVICE_KEY are now optional: the pipeline
        # runs on Postgres + Chroma (see app.db.schema), and these only
        # enable a legacy read fallback for rows not yet backfilled — no
        # warning needed when they're unset.
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
