"""Shared DB utility helpers used across all repositories."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def normalize_value(value: Any) -> Any:
    """Recursively coerce a value to a JSON-safe type for Supabase upserts."""
    if isinstance(value, Mapping):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(i) for i in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "model_dump"):
        return normalize_value(value.model_dump())
    return value
