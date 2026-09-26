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

jobs_abandoned_total = Counter(
    "jobs_abandoned_total",
    "Jobs refused as poison pills after repeated broker redelivery (worker loss)",
)

job_claims_total = Counter(
    "job_claims_total",
    "Video job deliveries by claim outcome",
    ["outcome"],  # run, retry, terminal, superseded, leased, worker_lost, worker_lost_repeatedly
)

jobs_retried_total = Counter(
    "jobs_retried_total",
    "Video jobs rerun after their worker was lost",
    ["path"],  # claim, sweeper
)

jobs_failed_by_sweeper_total = Counter(
    "jobs_failed_by_sweeper_total",
    "Video jobs failed by the recovery sweeper",
    ["reason"],  # worker_lost, worker_lost_repeatedly, deadline_exceeded, queue_expired, expired
)

jobs_redispatched_total = Counter(
    "jobs_redispatched_total",
    "Queued video jobs re-sent because their queue message was lost",
)

job_sweeper_runs_total = Counter(
    "job_sweeper_runs_total",
    "Recovery sweeper passes",
    ["outcome"],  # swept, not_leader, error
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

# Model-host health, derived from real traffic rather than from probes. The
# vLLM/Ollama hosts are scale-to-zero GPUs, so polling them for a gauge is
# what keeps them resident — alert on the failure rate of calls we were
# making anyway instead:
#     rate(llm_calls_total{outcome="failure"}[10m])
# Note these are only emitted by long-lived processes (API, Celery worker).
# The scripts/07* subprocesses record to dependency_state but are never
# scraped, since they exit with the job.
llm_calls_total = Counter(
    "llm_calls_total",
    "LLM / embedding HTTP calls by model host and outcome",
    ["host", "outcome"],  # outcome: success | failure
)

llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "Latency of an LLM / embedding call, including retries",
    ["host"],
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
