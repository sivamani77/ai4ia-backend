"""Feedback repository against a real Postgres, in a throwaway schema
(docs/design/feedback-auto-reply-design.md): the auto-reply columns and the
per-user rate-limit count query.

Skips unless the local infra Postgres (docker-compose.infra.yml, :15432) is up.
"""
from __future__ import annotations

import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

BASE_URL = "postgresql://ai4ia:ai4ia_local_dev@localhost:15432/ai4ia"


def _reachable() -> bool:
    try:
        psycopg2.connect(BASE_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres on :15432 not running")


@pytest.fixture
def db(monkeypatch):
    """FeedbackRepository on a fresh schema with the feedbacks table."""
    from app.db import postgres, schema as schema_module
    from app.db.repositories.feedback_repository import FeedbackRepository

    name = f"test_fb_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(BASE_URL)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(f"create schema {name}")
    cur.execute(f"set search_path to {name}")
    for statement in schema_module._STATEMENTS:
        if "feedbacks" in statement:
            cur.execute(statement)
    monkeypatch.setattr(postgres.settings, "database_url", f"{BASE_URL}?options=-csearch_path%3D{name}")
    postgres.reset_pool()
    try:
        yield FeedbackRepository()
    finally:
        postgres.reset_pool()
        cur.execute(f"drop schema {name} cascade")
        admin.close()


def _new_feedback(repo, *, user_id="u1") -> str:
    fid = f"fb_{uuid.uuid4().hex[:10]}"
    repo.create({"id": fid, "user_id": user_id, "username": "alice", "email": "a@example.com",
                 "feedback": "the export button is broken", "status": "open"})
    return fid


def test_set_auto_reply_leaves_status_open(db):
    repo = db
    fid = _new_feedback(repo)
    updated = repo.set_auto_reply(fid, category="bug", reply="It's on our list.")
    assert updated["status"] == "open"
    assert updated["auto_reply"] == "It's on our list."
    assert updated["auto_reply_category"] == "bug"
    assert updated["auto_reply_sent_at"] is not None


def test_set_auto_reply_never_touches_resolution_message(db):
    repo = db
    fid = _new_feedback(repo)
    repo.set_auto_reply(fid, category="question", reply="Here's how that works.")
    resolved = repo.resolve(fid, resolution_message="Fixed manually", resolved_by="admin-1")
    assert resolved["status"] == "resolved"
    assert resolved["resolution_message"] == "Fixed manually"
    assert resolved["auto_reply"] == "Here's how that works."  # untouched by the human resolve


def test_count_recent_by_user(db):
    repo = db
    assert repo.count_recent_by_user("u1") == 0
    _new_feedback(repo, user_id="u1")
    _new_feedback(repo, user_id="u1")
    _new_feedback(repo, user_id="u2")
    assert repo.count_recent_by_user("u1") == 2
    assert repo.count_recent_by_user("u2") == 1
    assert repo.count_recent_by_user("u1", hours=0) == 0
