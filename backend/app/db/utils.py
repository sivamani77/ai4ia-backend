"""Shared DB utility helpers used across all repositories."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


def normalize_value(value: Any) -> Any:
    """Recursively coerce a value to a JSON-safe type.

    Originally written for Supabase upserts; now also the normalization layer
    every psycopg2-backed repository applies on read. psycopg2's RealDictCursor
    hands back live Python objects for typed columns — `datetime` for
    timestamptz and `UUID` for uuid — where PostgREST returned ISO strings and
    plain strings. Callers (Pydantic response models, the JWT `sub` claim,
    dict comparisons) all expect the string shape, so the UUID branch below is
    load-bearing for the auth tables, whose primary keys are uuid.
    """
    if isinstance(value, Mapping):
        return {str(k): normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(i) for i in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "model_dump"):
        return normalize_value(value.model_dump())
    return value
