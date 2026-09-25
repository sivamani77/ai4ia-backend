"""Insight owner backfill (C-01) against a real Postgres, in a throwaway schema.

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


def _insight_statements() -> list[str]:
    from app.db import schema

    return [s for s in schema._STATEMENTS if "insights" in s and "ingest" not in s]


@pytest.fixture
def cur():
    schema = f"test_io_{uuid.uuid4().hex[:8]}"
    conn = psycopg2.connect(BASE_URL)
    conn.autocommit = True
    c = conn.cursor()
    c.execute(f"create schema {schema}")
    c.execute(f"set search_path to {schema}")
    try:
        yield c
    finally:
        c.execute(f"drop schema {schema} cascade")
        conn.close()


def _run(cur, statements):
    for statement in statements:
        cur.execute(statement)


def _owners(cur) -> dict:
    cur.execute("select id, user_id from insights order by id")
    return dict(cur.fetchall())


def test_backfill_assigns_only_unambiguous_owners(cur):
    statements = _insight_statements()
    create_table = statements[0]
    cur.execute(create_table)
    cur.execute("create table agent_decision_records (id text, insight_id text, user_id text)")
    cur.execute("insert into insights (id) values ('i_single'), ('i_collision'), ('i_none'), ('i_owned')")
    cur.execute("""
        insert into agent_decision_records (id, insight_id, user_id) values
            ('r1', 'i_single', 'alice'),
            ('r2', 'i_single', 'alice'),
            ('r3', 'i_collision', 'alice'),
            ('r4', 'i_collision', 'bob'),
            ('r5', 'i_owned', 'alice'),
            ('r6', 'i_none', null)
    """)
    cur.execute("alter table insights add column user_id text")
    cur.execute("update insights set user_id = 'carol' where id = 'i_owned'")

    _run(cur, statements)
    _run(cur, statements)  # idempotent on a second startup

    assert _owners(cur) == {
        "i_collision": None,   # two users share the old second-resolution id
        "i_none": None,
        "i_owned": "carol",    # an existing owner is never overwritten
        "i_single": "alice",
    }


def test_backfill_is_skipped_without_decision_records(cur):
    _run(cur, _insight_statements())

    cur.execute("select count(*) from insights")
    assert cur.fetchone() == (0,)
