"""Job leases and the recovery sweeper against a real Postgres, in a
throwaway schema (docs/design/job-recovery-plan.md).

Skips unless the local infra Postgres (docker-compose.infra.yml, :15432) is up.
"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

BASE_URL = "postgresql://ai4ia:ai4ia_local_dev@localhost:15432/ai4ia"
_JOBS_DDL = Path(__file__).resolve().parents[2] / "db" / "init" / "001_schema.sql"
QUEUE = "video_jobs_test"


def _reachable() -> bool:
    try:
        psycopg2.connect(BASE_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres on :15432 not running")


@pytest.fixture
def db(monkeypatch):
    """(JobRepository, cursor) on a fresh schema with jobs plus the lease columns."""
    from app.db import postgres, schema as schema_module
    from app.db.repositories.job_repository import JobRepository

    name = f"test_jl_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(BASE_URL)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(f"create schema {name}")
    ddl = _JOBS_DDL.read_text()
    cur.execute(f"set search_path to {name}; "
                + ddl[ddl.index("create table if not exists jobs"):ddl.index("create index if not exists idx_jobs_user_id")])
    lease_ddl = next(s for s in schema_module._STATEMENTS if "idx_jobs_running_heartbeat" in s)
    cur.execute(lease_ddl)
    monkeypatch.setattr(postgres.settings, "database_url", f"{BASE_URL}?options=-csearch_path%3D{name}")
    postgres.reset_pool()
    try:
        yield JobRepository(), cur
    finally:
        postgres.reset_pool()
        cur.execute(f"drop schema {name} cascade")
        admin.close()


def _new_job(repo, *, dispatch_id="d1", user="u1") -> str:
    job_id = f"vjob_{uuid.uuid4().hex[:10]}"
    assert repo.create_job_within_quota(
        {"id": job_id, "job_id": job_id, "status": "queued", "user_id": user, "metadata": {}},
        user_id=user, max_running=5, max_queued=5, max_active=50,
        dispatch_id=dispatch_id, worker_queue=QUEUE,
    ) is None
    return job_id


def _status(cur, job_id) -> str:
    cur.execute("select status from jobs where id = %s", (job_id,))
    return cur.fetchone()[0]


def test_parallel_deliveries_only_one_claims(db):
    repo, _ = db
    job_id = _new_job(repo)
    wins: list[bool] = []
    barrier = threading.Barrier(8)

    def deliver(i):
        barrier.wait()
        wins.append(repo.claim(job_id, dispatch_id="d1", owner=f"w{i}", worker_queue=QUEUE))

    threads = [threading.Thread(target=deliver, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1


def test_claim_needs_the_current_dispatch_id_and_a_first_run(db):
    repo, cur = db
    job_id = _new_job(repo)
    assert not repo.claim(job_id, dispatch_id="old", owner="w", worker_queue=QUEUE)
    assert repo.claim(job_id, dispatch_id="d1", owner="w", worker_queue=QUEUE)
    cur.execute("update jobs set status = 'queued' where id = %s", (job_id,))
    assert not repo.claim(job_id, dispatch_id="d1", owner="w2", worker_queue=QUEUE)  # attempts is 1


def test_rows_from_before_leases_are_claimed_by_messages_without_an_id(db):
    repo, cur = db
    cur.execute("insert into jobs (id, status, metadata) values ('legacy_1', 'queued', '{}')")
    assert repo.claim("legacy_1", dispatch_id=None, owner="w", worker_queue=QUEUE)
    cur.execute("select worker_queue from jobs where id = 'legacy_1'")
    assert cur.fetchone()[0] == QUEUE  # adopted by the environment that ran it


def test_heartbeat_and_finish_are_fenced_on_the_owner(db):
    repo, cur = db
    job_id = _new_job(repo)
    repo.claim(job_id, dispatch_id="d1", owner="w1", worker_queue=QUEUE)

    assert repo.heartbeat(job_id, "w1")
    assert not repo.heartbeat(job_id, "intruder")
    assert not repo.finish(job_id, "completed", "intruder")
    assert _status(cur, job_id) == "running"
    assert repo.finish(job_id, "completed", "w1")
    assert _status(cur, job_id) == "completed"
    assert not repo.heartbeat(job_id, "w1")


def test_finish_without_owner_needs_an_unclaimed_job(db):
    repo, cur = db
    job_id = _new_job(repo)
    assert repo.finish(job_id, "failed", None)  # API failing a job that never got queued
    assert _status(cur, job_id) == "failed"


def test_snapshot_saves_cannot_change_a_leased_jobs_status_except_to_cancel(db):
    repo, cur = db
    job_id = _new_job(repo)
    repo.claim(job_id, dispatch_id="d1", owner="w1", worker_queue=QUEUE)

    for attempted in ("queued", "completed", "failed"):
        repo.upsert_job({"id": job_id, "status": attempted, "metadata": {"status": attempted}})
        assert _status(cur, job_id) == "running"

    repo.upsert_job({"id": job_id, "status": "cancelled", "metadata": {}})
    assert _status(cur, job_id) == "cancelled"


def test_legacy_rows_keep_snapshot_status_writes(db):
    repo, cur = db
    cur.execute("insert into jobs (id, status, metadata) values ('legacy_2', 'queued', '{}')")
    repo.upsert_job({"id": "legacy_2", "status": "running", "metadata": {}})
    assert _status(cur, "legacy_2") == "running"


def test_redispatch_is_compare_and_set(db):
    repo, _ = db
    job_id = _new_job(repo)
    assert repo.redispatch(job_id, old_dispatch_id="d1", new_dispatch_id="d2")
    assert not repo.redispatch(job_id, old_dispatch_id="d1", new_dispatch_id="d3")
    assert repo.claim(job_id, dispatch_id="d2", owner="w", worker_queue=QUEUE)


def test_sweeper_fails_only_dead_jobs_of_its_own_queue(db, monkeypatch):
    from app.services import job_sweeper

    repo, cur = db
    alive, dead, other_env = _new_job(repo), _new_job(repo), _new_job(repo)
    for job_id in (alive, dead, other_env):
        repo.claim(job_id, dispatch_id="d1", owner="w", worker_queue=QUEUE)
    cur.execute("update jobs set heartbeat_at = now() - interval '10 minutes' where id in (%s, %s)", (dead, other_env))
    cur.execute("update jobs set worker_queue = 'video_jobs' where id = %s", (other_env,))

    monkeypatch.setattr(job_sweeper.settings, "worker_queue", QUEUE, raising=False)
    monkeypatch.setattr(job_sweeper.settings, "job_stale_after_seconds", 300, raising=False)
    monkeypatch.setattr(job_sweeper, "_healthy_since", -1e9)
    monkeypatch.setattr(job_sweeper, "_queue_size", lambda queue: 5)
    monkeypatch.setattr(job_sweeper.job_lease, "record_failure", lambda *a, **k: None)

    assert job_sweeper.sweep_once() == "swept"
    assert _status(cur, dead) == "failed"
    assert _status(cur, alive) == "running"
    assert _status(cur, other_env) == "running"
    cur.execute("select error->>'type', metadata->'error'->>'stage' from jobs where id = %s", (dead,))
    assert cur.fetchone() == ("worker_lost", "unknown")


def test_only_one_sweeper_leads_at_a_time(db, monkeypatch):
    from app.services import job_sweeper

    monkeypatch.setattr(job_sweeper.settings, "worker_queue", QUEUE, raising=False)
    holder = psycopg2.connect(BASE_URL)
    try:
        holder.cursor().execute("select pg_advisory_lock(hashtext(%s))", (f"job_sweeper:{QUEUE}",))
        assert job_sweeper.sweep_once() == "not_leader"
    finally:
        holder.close()
    monkeypatch.setattr(job_sweeper, "_queue_size", lambda queue: 5)
    assert job_sweeper.sweep_once() == "swept"


def test_late_snapshot_cannot_erase_how_a_leased_job_ended(db):
    """The sweeper failed the job; the worker, still alive, then saves a snapshot."""
    repo, cur = db
    job_id = _new_job(repo)
    repo.claim(job_id, dispatch_id="d1", owner="w1", worker_queue=QUEUE)
    repo.fail_if_lease(job_id, expected_owner="w1", error={"type": "worker_lost", "message": "gone"})

    repo.upsert_job({"id": job_id, "status": "running", "error": None, "completed_at": None,
                     "metadata": {"status": "running", "current_stage": "09_export", "progress_percent": 40}})

    cur.execute("select status, error->>'type', completed_at is not null,"
                " metadata->'error'->>'type', metadata->>'current_stage' from jobs where id = %s", (job_id,))
    assert cur.fetchone() == ("failed", "worker_lost", True, "worker_lost", "09_export")
