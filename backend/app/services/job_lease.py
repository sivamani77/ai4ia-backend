"""Leases on running video jobs: one runner per job, a heartbeat while it
runs, and a stop signal when the job should end early.

See docs/job-recovery-plan.md for the design. The worker claims a job before
running it, a thread stamps a heartbeat, and the run checks stop_reason() at
stage boundaries and while a script runs.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass, field

from app.core import metrics as _metrics
from app.core.settings import settings
from app.services import job_store

logger = logging.getLogger("ai4ia.services.job_lease")

HOST = socket.gethostname()

STOP_CANCELLED = "cancelled"
STOP_SUPERSEDED = "superseded"
STOP_HEARTBEAT_LOST = "heartbeat_lost"
STOP_DEADLINE = "deadline"

_TERMINAL = {"cancelled", "completed", "failed"}

WORKER_LOST_ERROR = {
    "type": "worker_lost",
    "message": "The worker running this job stopped before it finished.",
    "user_message": "The server processing your video stopped unexpectedly. Please try again.",
}


@dataclass
class _Lease:
    job_id: str
    owner: str
    started: float
    last_ok: float
    stop_reason: str | None = None
    stopped: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


_leases: dict[str, _Lease] = {}
_lock = threading.Lock()


def owner_id(task_id: str | None) -> str:
    return f"{HOST}:{os.getpid()}:{task_id or '-'}"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def claim(job_id: str, dispatch_id: str | None, owner: str) -> str:
    """Try to take the job. Returns "run" when this delivery should run it;
    otherwise why not: terminal, superseded, leased (a live worker has it),
    worker_lost (its worker is gone, so the job was failed) or missing."""
    if job_store.claim_job(job_id, dispatch_id=dispatch_id, owner=owner, worker_queue=settings.worker_queue):
        _metrics.job_claims_total.labels(outcome="run").inc()
        return "run"

    state = job_store.job_lease_state(job_id)
    outcome = _explain_lost_claim(job_id, dispatch_id, state)
    _metrics.job_claims_total.labels(outcome=outcome).inc()
    logger.info("[%s] Delivery not run: %s", job_id, outcome)
    return outcome


def _explain_lost_claim(job_id: str, dispatch_id: str | None, state: dict | None) -> str:
    if state is None:
        return "missing"
    if state["status"] in _TERMINAL:
        return "terminal"
    if state["dispatch_id"] != dispatch_id:
        return "superseded"
    if state["status"] != "running":
        # Queued but already claimed once: a stale copy of a finished attempt.
        return "superseded"
    holder = state["lease_owner"]
    if not _holder_is_dead(holder, state["heartbeat_age"]):
        return "leased"
    # No retries (decision 1): the job's only run was lost, so it fails.
    _kill_leftover_scripts(job_id, state)
    if job_store.fail_job_if_lease(job_id, expected_owner=holder, error=WORKER_LOST_ERROR):
        record_failure(job_id, "worker_lost", WORKER_LOST_ERROR)
    return "worker_lost"


def _holder_is_dead(holder: str | None, heartbeat_age: float | None) -> bool:
    if holder is None or heartbeat_age is None:
        return True
    if float(heartbeat_age) > settings.job_stale_after_seconds:
        return True
    host, _, rest = holder.partition(":")
    pid_text = rest.partition(":")[0]
    if host != HOST or not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if pid == os.getpid():
        # This process is handling the current delivery, so it isn't running
        # the old one: that attempt's process died and the pid was reused.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _kill_leftover_scripts(job_id: str, state: dict) -> None:
    """Kill a dead attempt's script group left running on this host. The group
    is killed whole once its leader (the script) is confirmed to name the job;
    if the leader is gone, only members whose command line names the job are
    killed. Either way a reused pid can never hit another job."""
    if state.get("active_host") != HOST or not str(state.get("active_pid") or "").isdigit():
        return
    kill_job_script_group(job_id, int(state["active_pid"]))


def kill_job_script_group(job_id: str, pgid: int) -> None:
    try:
        with open(f"/proc/{pgid}/cmdline", "rb") as fh:
            leader_is_ours = job_id.encode() in fh.read()
    except OSError:
        leader_is_ours = False
    if leader_is_ours:
        try:
            os.killpg(pgid, signal.SIGKILL)
            logger.warning("[%s] Killed leftover script group %s", job_id, pgid)
            return
        except OSError:
            pass
    try:
        entries = os.listdir("/proc")
    except OSError:
        return
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            if int(fields[2]) != pgid:  # after ")": state, ppid, pgrp
                continue
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                if job_id.encode() not in fh.read():
                    continue
            os.kill(int(entry), signal.SIGKILL)
            logger.warning("[%s] Killed leftover script process %s", job_id, entry)
        except (OSError, IndexError, ValueError):
            continue


def record_failure(job_id: str, reason: str, error: dict, *, by_sweeper: bool = False) -> None:
    """Events and records for a job failed outside the runner."""
    from app.services import processing_records

    try:
        job_store.append_event(job_id, "failed", "recovery", {
            "error_type": error["type"], "message": error["message"],
        })
    except Exception:  # noqa: BLE001
        logger.warning("[%s] Could not record failure event", job_id, exc_info=True)
    try:
        processing_records.finish(job_id, "failed")
    except Exception:  # noqa: BLE001
        logger.warning("[%s] Could not close processing record", job_id, exc_info=True)
    _metrics.job_completions_total.labels(status="failed").inc()
    if by_sweeper:
        _metrics.jobs_failed_by_sweeper_total.labels(reason=reason).inc()


# ---------------------------------------------------------------------------
# Heartbeat and stop signal
# ---------------------------------------------------------------------------

def start(job_id: str, owner: str) -> None:
    now = time.monotonic()
    lease = _Lease(job_id=job_id, owner=owner, started=now, last_ok=now)
    lease.thread = threading.Thread(target=_beat, args=(lease,), name=f"heartbeat-{job_id}", daemon=True)
    with _lock:
        _leases[job_id] = lease
    lease.thread.start()


def release(job_id: str) -> str | None:
    """Stop the heartbeat and forget the lease; returns its owner (None when
    this process holds no lease on the job, e.g. in the API)."""
    with _lock:
        lease = _leases.pop(job_id, None)
    if lease is None:
        return None
    lease.stopped.set()
    if lease.thread is not None and lease.thread is not threading.current_thread():
        lease.thread.join(timeout=5)
    return lease.owner


def stop_reason(job_id: str) -> str | None:
    with _lock:
        lease = _leases.get(job_id)
    if lease is None:
        return None
    if lease.stop_reason:
        return lease.stop_reason
    if time.monotonic() - lease.started > settings.job_max_runtime_seconds:
        return STOP_DEADLINE
    return None


def _beat(lease: _Lease) -> None:
    interval = settings.job_heartbeat_interval_seconds
    give_up_after = 0.8 * settings.job_stale_after_seconds
    while not lease.stopped.wait(interval):
        try:
            ok = job_store.heartbeat_job(lease.job_id, lease.owner)
        except Exception:  # noqa: BLE001
            if time.monotonic() - lease.last_ok > give_up_after:
                # The sweeper will soon treat this job as lost; stop before a
                # second run could start rather than race it.
                logger.error("[%s] Heartbeat failing for %.0fs; stopping the run",
                             lease.job_id, time.monotonic() - lease.last_ok)
                lease.stop_reason = STOP_HEARTBEAT_LOST
                return
            logger.warning("[%s] Heartbeat write failed", lease.job_id, exc_info=True)
            continue
        if ok:
            lease.last_ok = time.monotonic()
            continue
        try:
            state = job_store.job_lease_state(lease.job_id)
        except Exception:  # noqa: BLE001
            state = None
        lease.stop_reason = STOP_CANCELLED if state and state["status"] == "cancelled" else STOP_SUPERSEDED
        logger.warning("[%s] Lease lost (%s); stopping the run", lease.job_id, lease.stop_reason)
        return


def raise_if_stopped(job_id: str, stage_name: str) -> None:
    """Raise the exception matching the job's stop reason, if it has one."""
    from app.pipeline.base import JobCancelledException, JobDeadlineExceeded, JobSupersededException

    reason = stop_reason(job_id)
    if reason is None:
        return
    if reason == STOP_CANCELLED:
        raise JobCancelledException(job_id, stage_name)
    if reason == STOP_DEADLINE:
        raise JobDeadlineExceeded(job_id, stage_name)
    raise JobSupersededException(job_id, stage_name, reason)
