"""Prometheus metric definitions, shared across the app as singletons.

Import the specific metric object you need (e.g. `from app.core.metrics
import jobs_created_total`) and call `.inc()`/`.observe()`/`.set()` at the
point the event happens. Values are served by the instrumentator-mounted
`/metrics` endpoint (see main.py) for scraping — nothing here talks to a
Prometheus server directly.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Job pipeline
# ---------------------------------------------------------------------------

jobs_created_total = Counter("jobs_created_total", "Video jobs created")

job_completions_total = Counter(
    "job_completions_total",
    "Video jobs reaching a terminal status",
    ["status"],  # completed | failed | cancelled
)

job_duration_seconds = Histogram(
    "job_duration_seconds",
    "End-to-end video job duration, from creation to completion",
)

job_stage_transitions_total = Counter(
    "job_stage_transitions_total",
    "Pipeline stage status transitions",
    ["stage", "status"],
)

job_stage_duration_seconds = Histogram(
    "job_stage_duration_seconds",
    "Per-stage duration, from running to completed/failed",
    ["stage"],
)

jobs_orphan_reaped_total = Counter(
    "jobs_orphan_reaped_total",
    "Jobs marked failed on startup because their owning process was gone",
)

job_queue_position = Histogram(
    "job_queue_position",
    "Queue position of a job at creation time",
    buckets=(0, 1, 2, 3, 5, 8, 13, 20, float("inf")),
)

# ---------------------------------------------------------------------------
# Storyline cache (app/services/storyline_cache.py)
# ---------------------------------------------------------------------------

storyline_cache_hits_total = Counter(
    "storyline_cache_hits_total",
    "Storyline cache hits, skipping 07_storyline + 07b_validate_storyline",
    ["layer"],  # l1 | l2
)

storyline_cache_misses_total = Counter(
    "storyline_cache_misses_total",
    "Storyline cache misses — full LLM generation ran",
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

redis_reachable = Gauge("redis_reachable", "Redis reachability (1=up, 0=down)")

dependency_up = Gauge(
    "dependency_up",
    "Dependency health (1=up, 0=down)",
    ["dependency"],
)

dependency_check_duration_seconds = Histogram(
    "dependency_check_duration_seconds",
    "Latency of a dependency health check",
    ["dependency"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

auth_events_total = Counter(
    "auth_events_total",
    "Auth flow outcomes",
    ["event", "outcome"],  # outcome: success | failure | rate_limited
)

auth_refresh_reuse_detected_total = Counter(
    "auth_refresh_reuse_detected_total",
    "Refresh token reuse detected — a token already consumed was presented "
    "again, which indicates possible token theft",
)

rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total",
    "Requests rejected by an in-process rate limiter",
    ["limiter"],
)
