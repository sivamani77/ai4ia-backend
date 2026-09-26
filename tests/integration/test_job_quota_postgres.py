"""Video job quota (Q-01) against a real Postgres, in a throwaway schema.

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


def _reachable() -> bool:
    try:
        psycopg2.connect(BASE_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres on :15432 not running")


@pytest.fixture
def repo(monkeypatch):
    """A JobRepository on a fresh schema holding just the jobs table."""
    from app.db import postgres
    from app.db.repositories.job_repository import JobRepository

    schema = f"test_jq_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(BASE_URL)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(f"create schema {schema}")
    ddl = _JOBS_DDL.read_text()
    jobs_ddl = ddl[ddl.index("create table if not exists jobs"):ddl.index("create index if not exists idx_jobs_created_at")]
    cur.execute(f"set search_path to {schema}; {jobs_ddl}")
    monkeypatch.setattr(postgres.settings, "database_url", f"{BASE_URL}?options=-csearch_path%3D{schema}")
    postgres.reset_pool()
    try:
        yield JobRepository()
    finally:
        postgres.reset_pool()
        cur.execute(f"drop schema {schema} cascade")
        admin.close()


def _job(user_id: str, status: str = "queued") -> dict:
    job_id = f"vjob_{uuid.uuid4().hex[:12]}"
    return {"id": job_id, "job_id": job_id, "status": status, "user_id": user_id, "metadata": {}}


def test_parallel_submissions_cannot_exceed_the_per_user_quota(repo):
    results: list[str | None] = []
    barrier = threading.Barrier(8)

    def submit():
        barrier.wait()
        results.append(repo.create_job_within_quota(
            _job("alice"), user_id="alice", max_running=1, max_queued=3, max_active=50,
        ))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(None) == 3
    assert results.count("user_queued") == 5
    rows, total = repo.list_by_user_page("alice", 50, 0)
    assert total == 3


def test_limits_report_which_one_was_hit(repo):
    assert repo.create_job_within_quota(_job("bob", "running"), user_id="bob",
                                        max_running=1, max_queued=3, max_active=50) is None
    assert repo.create_job_within_quota(_job("bob"), user_id="bob",
                                        max_running=1, max_queued=3, max_active=50) == "user_running"
    assert repo.create_job_within_quota(_job("carol"), user_id="carol",
                                        max_running=1, max_queued=3, max_active=1) == "active"


def test_paging_and_queue_position(repo):
    ids = []
    for _ in range(3):
        job = _job("dave")
        repo.create_job_within_quota(job, user_id="dave", max_running=1, max_queued=5, max_active=50)
        ids.append(job["id"])

    page, total = repo.list_by_user_page("dave", 2, 0)
    assert total == 3 and len(page) == 2
    assert repo.list_by_user_page("nobody", 2, 0) == ([], 0)
    assert repo.queue_position(ids[-1]) == 3


def test_submission_gives_up_when_the_lock_is_held_too_long(repo):
    """Waiting holds a pool connection, so after lock_timeout the answer is "busy"."""
    from app.db import postgres
    from app.db.repositories.job_repository import _JOB_QUOTA_LOCK_KEY

    holder = psycopg2.connect(postgres.settings.database_url)
    try:
        holder.cursor().execute("select pg_advisory_lock(%s)", (_JOB_QUOTA_LOCK_KEY,))
        assert repo.create_job_within_quota(_job("erin"), user_id="erin",
                                            max_running=1, max_queued=3, max_active=50) == "busy"
    finally:
        holder.close()
    assert repo.list_by_user_page("erin", 5, 0) == ([], 0)
