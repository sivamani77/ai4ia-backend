from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.core.settings import settings

SUPABASE_URL = settings.supabase_url
SUPABASE_SERVICE_KEY = settings.supabase_service_key

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
    return _create_supabase_client()
