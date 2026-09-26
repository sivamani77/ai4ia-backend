"""Supabase auth catch-up against a real Postgres, in a throwaway schema.
Supabase is faked; only its paged reads are replaced.

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
    """(migration module, cursor, supabase dict) on a fresh schema with the auth tables."""
    from app.db import postgres
    from app.db.schema import ensure_auth_schema
    from app.services import auth_data_migration as m

    name = f"test_ac_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(BASE_URL)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(f"create schema {name}")
    monkeypatch.setattr(postgres.settings, "database_url", f"{BASE_URL}?options=-csearch_path%3D{name}")
    postgres.reset_pool()
    ensure_auth_schema()
    cur.execute(f"set search_path to {name}")

    supabase: dict[str, list[dict]] = {table: [] for table, _ in m.TABLES}
    monkeypatch.setattr(m, "fetch_table", lambda table, columns: iter([supabase[table]] if supabase[table] else []))
    try:
        yield m, cur, supabase
    finally:
        postgres.reset_pool()
        cur.execute(f"drop schema {name} cascade")
        admin.close()


def _user(**kw) -> dict:
    row = {
        "id": str(uuid.uuid4()), "username": f"u{uuid.uuid4().hex[:6]}", "password_hash": "h1",
        "password": "", "email_id": f"{uuid.uuid4().hex[:6]}@example.com", "persona": "general",
        "status": "active", "email_verified": True, "failed_login_count": 0, "locked_until": None,
        "created_at": "2026-09-01T00:00:00+00:00", "updated_at": "2026-09-01T00:00:00+00:00",
    }
    row.update(kw)
    return row


def _insert_pg_user(cur, row: dict) -> None:
    cols = ", ".join(row)
    cur.execute(f"insert into app_users ({cols}) values ({', '.join(['%s'] * len(row))})", list(row.values()))


def _token(user_id: str, **kw) -> dict:
    row = {
        "id": str(uuid.uuid4()), "user_id": user_id, "token_hash": uuid.uuid4().hex,
        "family_id": "f1", "expires_at": "2026-12-01T00:00:00+00:00",
        "created_at": "2026-09-01T00:00:00+00:00", "used_at": None, "revoked_at": None,
        "user_agent": None, "ip": None,
    }
    row.update(kw)
    return row


def test_inserts_users_missing_in_postgres(db):
    m, cur, supabase = db
    new_user = _user(password=None)
    supabase["app_users"] = [new_user]

    report = m.catch_up()

    assert report["app_users"]["inserted"] == 1
    cur.execute("select password_hash, password from app_users where id = %s", (new_user["id"],))
    assert cur.fetchone() == ("h1", "")


def test_newer_supabase_edit_wins(db):
    m, cur, supabase = db
    user = _user()
    _insert_pg_user(cur, user)
    supabase["app_users"] = [{**user, "password_hash": "h2", "updated_at": "2026-09-20T00:00:00+00:00"}]

    report = m.catch_up()

    assert report["app_users"]["updated"] == 1
    cur.execute("select password_hash from app_users where id = %s", (user["id"],))
    assert cur.fetchone()[0] == "h2"


def test_newer_postgres_edit_is_kept(db):
    m, cur, supabase = db
    user = _user(password_hash="pg-new", updated_at="2026-09-25T00:00:00+00:00")
    _insert_pg_user(cur, user)
    supabase["app_users"] = [{**user, "password_hash": "old", "updated_at": "2026-09-20T00:00:00+00:00"}]

    report = m.catch_up()

    assert report["app_users"]["unchanged"] == 1
    cur.execute("select password_hash from app_users where id = %s", (user["id"],))
    assert cur.fetchone()[0] == "pg-new"


def test_email_clash_is_skipped_and_the_rest_continue(db):
    m, cur, supabase = db
    _insert_pg_user(cur, _user(email_id="same@example.com"))
    clash = _user(email_id="SAME@example.com")
    fine = _user()
    supabase["app_users"] = [clash, fine]
    supabase["refresh_tokens"] = [_token(clash["id"]), _token(fine["id"])]

    report = m.catch_up()

    assert report["app_users"] == {"inserted": 1, "updated": 0, "unchanged": 0, "skipped": 1}
    # The clashing user's token has no parent row, so it is skipped too.
    assert report["refresh_tokens"]["inserted"] == 1
    assert report["refresh_tokens"]["skipped"] == 1


def test_refresh_token_is_never_reopened(db):
    m, cur, supabase = db
    user = _user()
    _insert_pg_user(cur, user)
    used = _token(user["id"])
    revoked_in_pg = _token(user["id"], revoked_at="2026-09-25T00:00:00+00:00")
    for t in (used, revoked_in_pg):
        cols = ", ".join(t)
        cur.execute(f"insert into refresh_tokens ({cols}) values ({', '.join(['%s'] * len(t))})", list(t.values()))
    supabase["app_users"] = [user]
    supabase["refresh_tokens"] = [
        {**used, "used_at": "2026-09-20T00:00:00+00:00"},  # rotated in Supabase after the copy
        {**revoked_in_pg, "revoked_at": None},             # Supabase copy is older
    ]

    report = m.catch_up()

    assert report["refresh_tokens"] == {"inserted": 0, "updated": 1, "unchanged": 1, "skipped": 0}
    cur.execute("select used_at is not null from refresh_tokens where id = %s", (used["id"],))
    assert cur.fetchone()[0] is True
    cur.execute("select revoked_at is not null from refresh_tokens where id = %s", (revoked_in_pg["id"],))
    assert cur.fetchone()[0] is True


def test_second_pass_changes_nothing(db):
    m, cur, supabase = db
    user = _user()
    supabase["app_users"] = [user]
    supabase["refresh_tokens"] = [_token(user["id"])]

    m.catch_up()
    report = m.catch_up()

    assert report["app_users"] == {"inserted": 0, "updated": 0, "unchanged": 1, "skipped": 0}
    assert report["refresh_tokens"] == {"inserted": 0, "updated": 0, "unchanged": 1, "skipped": 0}


def test_catch_up_once_reports_done(db):
    m, _, supabase = db
    supabase["app_users"] = [_user()]
    assert m.catch_up_once("test") == "done"
