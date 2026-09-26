"""Recovery sweeper for video jobs (docs/design/job-recovery-plan.md, section 6).

Runs in the API. Each pass takes a per-environment advisory lock, so one API
process acts at a time, and handles only jobs dispatched to this
environment's queue (settings.worker_queue):

  deadline    running past the runtime cap plus the stale window (the worker
              fails its own job at the cap, so this only catches a silent one)
  worker lost running with no heartbeat for job_stale_after_seconds: requeued
              while it has retries left (job_lease.loss_verdict), else failed
  expired     queued longer than job_queued_max_age_seconds, counted from
              the requeue for a retried job
  legacy      rows from before leases, stuck for 6 hours (either environment)
  lost msg    queued, never claimed, and the queue is empty: re-send it

Every change is one conditional UPDATE, so a worker writing at the same
moment always wins or loses cleanly.
"""
from __future__ import annotations

import logging
import random
import time
import uuid

from app.core import metrics as _metrics
from app.core.settings import settings
from app.db.postgres import pg_session
from app.services import job_lease, job_store

logger = logging.getLogger("ai4ia.services.job_sweeper")

_LOST_MESSAGE_GRACE_SECONDS = 120
_LEGACY_MAX_AGE_SECONDS = 6 * 3600

# When sweeps started succeeding without a break, in this process. Stale
# heartbeats are only acted on after a full stale window of healthy sweeps,
# so a database outage (no heartbeats could be written) or an API restart
# never fails a live job.
_healthy_since: float | None = None

_ERRORS = {
    "deadline_exceeded": {
        "type": "deadline_exceeded",
        "message": "The job ran past its time limit and its worker stopped responding.",
        "user_message": "Your video took too long to generate and was stopped. "
                        "Please try again, or choose a shorter duration.",
    },
    "worker_lost": job_lease.WORKER_LOST_ERROR,
    "worker_lost_repeatedly": job_lease.WORKER_LOST_REPEATEDLY_ERROR,
    "queue_expired": {
        "type": "queue_expired",
        "message": "The job waited in the queue past its limit.",
        "user_message": "The system was too busy to start your video. Please try again.",
    },
    "expired": {
        "type": "expired",
        "message": "The job was left unfinished by an earlier version of the server.",
        "user_message": "Your video could not be finished. Please try again.",
    },
}


def sweep_once() -> str:
    """Run one pass; returns swept, not_leader or error."""
    global _healthy_since
    queue = settings.worker_queue
    try:
        with pg_session() as cur:
            cur.execute("select pg_try_advisory_lock(hashtext(%s)) as locked", (f"job_sweeper:{queue}",))
            if not cur.fetchone()["locked"]:
                _mark_healthy()
                return "not_leader"
            try:
                _run_rules(queue)
            finally:
                cur.execute("select pg_advisory_unlock(hashtext(%s))", (f"job_sweeper:{queue}",))
    except Exception:  # noqa: BLE001
        _healthy_since = None
        logger.exception("Job sweeper pass failed")
        return "error"
    _mark_healthy()
    return "swept"


def _mark_healthy() -> None:
    global _healthy_since
    if _healthy_since is None:
        _healthy_since = time.monotonic()


def _heartbeats_trustworthy() -> bool:
    return _healthy_since is not None and time.monotonic() - _healthy_since >= settings.job_stale_after_seconds


def _run_rules(queue: str) -> None:
    stale = settings.job_stale_after_seconds
    params = {"queue": queue, "stale": stale}

    _fail(
        "deadline_exceeded",
        "worker_queue = %(queue)s and status = 'running'"
        " and started_at < now() - make_interval(secs => %(limit)s)",
        {**params, "limit": settings.job_max_runtime_seconds + stale},
    )
    if _heartbeats_trustworthy():
        _recover_lost_workers(queue)
    _fail(
        "queue_expired",
        "worker_queue = %(queue)s and status = 'queued'"
        " and case when attempts = 0 then created_at"
        " else coalesce((metadata->>'requeued_at')::timestamptz, dispatched_at) end"
        " < now() - make_interval(secs => %(limit)s)",
        {**params, "limit": settings.job_queued_max_age_seconds},
    )
    _fail(
        "expired",
        "worker_queue is null and status in ('queued', 'running')"
        " and created_at < now() - make_interval(secs => %(limit)s)",
        {"limit": _LEGACY_MAX_AGE_SECONDS},
    )
    _resend_lost_messages(queue)


def _fail(reason: str, where_sql: str, params: dict) -> None:
    error = _ERRORS[reason]
    rows = job_store.get_job_repo().fail_where(where_sql, params, error=error)
    for row in rows:
        logger.warning("Sweeper failed job %s (%s) at stage %s", row["id"], reason, row["stage"])
        job_lease.record_failure(row["id"], reason, error, by_sweeper=True)


def _recover_lost_workers(queue: str) -> None:
    """Requeue each running job whose heartbeat went stale while it has
    retries left; fail the rest. Every write is conditional on the heartbeat
    still being stale, so a worker that beats at the same moment wins."""
    stale = settings.job_stale_after_seconds
    repo = job_store.get_job_repo()
    for row in repo.lost_worker_candidates(queue, stale):
        attempts = int(row.get("attempts") or 0)
        error = job_lease.loss_verdict(row["id"], attempts, row["stage"])
        if error is not None:
            _fail(
                error["type"],
                "worker_queue = %(queue)s and id = %(id)s and status = 'running'"
                " and lease_owner is not distinct from %(owner)s"
                " and heartbeat_at < now() - make_interval(secs => %(stale)s)",
                {"queue": queue, "id": row["id"], "owner": row["lease_owner"], "stale": stale},
            )
            continue
        new_id = uuid.uuid4().hex
        if not repo.requeue_lost(row["id"], worker_queue=queue, expected_owner=row["lease_owner"],
                                 stale_seconds=stale, new_dispatch_id=new_id):
            continue
        job_lease.record_interrupted(row["id"], attempts, row["stage"], path="sweeper")
        _send(row["id"], row.get("request") or {}, new_id)


def _send(job_id: str, request: dict, dispatch_id: str) -> bool:
    """Send a job's message after its row has committed, so older messages
    are already void. A failed send leaves the job queued with no message;
    the lost-message rule re-sends it."""
    from app.services.video_job_runner import start_video_job_background

    try:
        start_video_job_background(job_id, request, dispatch_id=dispatch_id)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Sweeper could not send job %s", job_id, exc_info=True)
        return False


def _resend_lost_messages(queue: str) -> None:
    max_retries = settings.job_max_worker_loss_retries
    candidates = job_store.get_job_repo().lost_queue_candidates(queue, _LOST_MESSAGE_GRACE_SECONDS, max_retries)
    if not candidates:
        return
    size = _queue_size(queue)
    if size != 0:
        # Unknown (Redis down) or messages still waiting: the job's message may
        # be among them, and re-sending would put it at the back of the queue.
        return
    for row in candidates:
        new_id = uuid.uuid4().hex
        if not job_store.get_job_repo().redispatch(row["id"], old_dispatch_id=row["dispatch_id"],
                                                   new_dispatch_id=new_id, max_retries=max_retries):
            continue
        if _send(row["id"], row.get("request") or {}, new_id):
            _metrics.jobs_redispatched_total.inc()
            logger.warning("Sweeper re-sent job %s: its queue message was lost", row["id"])


def _queue_size(queue: str) -> int | None:
    """Messages waiting in the queue, across its priority sub-queues; None if
    Redis can't be reached."""
    from kombu.exceptions import ChannelError

    from app.services.video_job_runner import _redis_reachable
    from app.worker.celery_app import celery_app

    # kombu retries a lost connection forever by default; never let the
    # sweeper wait on Redis while it holds the leader lock.
    if not _redis_reachable():
        return None
    try:
        with celery_app.connection_for_write(
            transport_options={"socket_timeout": 5, "socket_connect_timeout": 5},
        ) as conn:
            conn.ensure_connection(max_retries=1, timeout=5)
            try:
                return conn.default_channel.queue_declare(queue=queue, passive=True).message_count
            except ChannelError:
                return 0  # Redis removes empty lists, so a missing queue is empty
    except Exception:  # noqa: BLE001
        logger.warning("Sweeper could not read queue %s", queue, exc_info=True)
        return None


async def run_forever() -> None:
    """The API's background loop: one pass per interval, with jitter so API
    processes don't all try the lock at once."""
    import asyncio

    logger.info("Job sweeper running for queue %s", settings.worker_queue)
    while True:
        await asyncio.sleep(settings.job_sweep_interval_seconds + random.uniform(0, 10))
        outcome = await asyncio.to_thread(sweep_once)
        _metrics.job_sweeper_runs_total.labels(outcome=outcome).inc()
