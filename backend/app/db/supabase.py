from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.core.settings import settings

# Pipeline project — LEGACY. fragments/storylines/insights/ingest_* now live
# in Postgres (see app.db.postgres, app.db.schema.ensure_pipeline_schema).
# get_supabase_client() below is kept only so repositories can fall back to
# reading rows that predate the Postgres cutover and haven't been backfilled
# yet — nothing should ever write through this client again. Safe to leave
# SUPABASE_URL/SUPABASE_SERVICE_KEY unset once the cluster's Postgres has all
# pipeline data (the fallback simply no-ops when unset).
SUPABASE_URL = settings.supabase_url
SUPABASE_SERVICE_KEY = settings.supabase_service_key

# Auth project — separate Supabase project holding app_users/otp_codes/
# refresh_tokens/auth_audit. Deliberately NOT the same client as the
# legacy pipeline project above, and NOT part of this Postgres migration.
AUTH_SUPABASE_URL = settings.auth_supabase_url
AUTH_SUPABASE_SERVICE_KEY = settings.auth_supabase_service_key

logger = logging.getLogger("ai4ia.db.supabase")

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


class SupabaseConfigurationError(RuntimeError):
    """Raised when required Supabase settings are missing."""


class SupabaseDependencyError(RuntimeError):
    """Raised when the Supabase SDK is unavailable."""


class SupabaseClientError(RuntimeError):
    """Raised when the Supabase client cannot be created."""


def _validate_supabase_settings() -> None:
    missing: list[str] = []

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_KEY")

    if missing:
        raise SupabaseConfigurationError(
            f"Missing Supabase configuration: {', '.join(missing)}"
        )


def _create_supabase_client() -> Client:
    if create_client is None:
        raise SupabaseDependencyError(
            "The 'supabase' package is not installed. Install backend dependencies first."
        )

    _validate_supabase_settings()

    try:
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as exc:  # pragma: no cover - depends on external service
        logger.exception("Failed to initialize Supabase client")
        raise SupabaseClientError("Failed to initialize Supabase client") from exc

    logger.info("Initialized Supabase client")
    return client


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """LEGACY read-only fallback client for the pipeline project. Repositories
    use this to look up rows not yet present in Postgres — do not write
    through this client, and do not add new callers outside that fallback
    path."""
    return _create_supabase_client()


def reset_supabase_client() -> None:
    """Drop the cached client so the next get_supabase_client() builds a fresh
    one with a new connection pool. Used to recover from a server-dropped
    HTTP/2 keepalive connection (RemoteProtocolError: Server disconnected)
    after a long-running job leaves the pooled connection idle."""
    get_supabase_client.cache_clear()


def _validate_auth_supabase_settings() -> None:
    missing: list[str] = []

    if not AUTH_SUPABASE_URL:
        missing.append("AUTH_SUPABASE_URL")
    if not AUTH_SUPABASE_SERVICE_KEY:
        missing.append("AUTH_SUPABASE_SERVICE_KEY")

    if missing:
        raise SupabaseConfigurationError(
            f"Missing auth Supabase configuration: {', '.join(missing)}"
        )


def _create_auth_supabase_client() -> Client:
    if create_client is None:
        raise SupabaseDependencyError(
            "The 'supabase' package is not installed. Install backend dependencies first."
        )

    _validate_auth_supabase_settings()

    try:
        client = create_client(AUTH_SUPABASE_URL, AUTH_SUPABASE_SERVICE_KEY)
    except Exception as exc:  # pragma: no cover - depends on external service
        logger.exception("Failed to initialize auth Supabase client")
        raise SupabaseClientError("Failed to initialize auth Supabase client") from exc

    logger.info("Initialized auth Supabase client (app_users project)")
    return client


@lru_cache(maxsize=1)
def get_auth_supabase_client() -> Client:
    """Client for the SEPARATE Supabase project holding app_users, otp_codes,
    refresh_tokens, and auth_audit. Do not use this for jobs/pipeline tables,
    and do not use get_supabase_client() for auth tables — the two projects
    are not interchangeable."""
    return _create_auth_supabase_client()


def reset_auth_supabase_client() -> None:
    get_auth_supabase_client.cache_clear()
