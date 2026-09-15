"""Unit tests for the auth repositories after the Supabase -> Postgres migration.

These assert the things the migration could plausibly have broken without any
test noticing: the SQL each method emits, that dict-driven INSERTs only ever
name known columns, and that psycopg2's UUID/datetime objects are normalized
back to the strings callers expect.

The round-trip behaviour against a real database is covered separately by
running against the local Docker Postgres (docker-compose.infra.yml).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.db.repositories.auth_audit_repository import AuthAuditRepository
from app.db.repositories.otp_repository import OtpRepository
from app.db.repositories.refresh_token_repository import RefreshTokenRepository


def _norm(sql: str) -> str:
    return " ".join(sql.split()).lower()


# ---------------------------------------------------------------------------
# OtpRepository
# ---------------------------------------------------------------------------

class TestOtpRepository:
    def test_create_only_names_known_columns(self):
        captured = {}

        def fake(query, params):
            captured["q"], captured["p"] = query, params
            return {"id": uuid.uuid4(), "email": "a@x.com"}

        with patch("app.db.repositories.otp_repository.pg_fetch_one", side_effect=fake):
            OtpRepository().create({
                "email": "a@x.com", "purpose": "signup", "code_hash": "h",
                "expires_at": "2030-01-01T00:00:00+00:00",
                "injected); drop table otp_codes; --": "x",
            })

        assert "drop table" not in _norm(captured["q"])
        assert set(captured["p"]) == {"email", "purpose", "code_hash", "expires_at"}

    def test_create_normalizes_uuid_to_str(self):
        row = {"id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
               "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
        with patch("app.db.repositories.otp_repository.pg_fetch_one", return_value=row):
            rec = OtpRepository().create({"email": "a@x.com", "purpose": "signup",
                                          "code_hash": "h", "expires_at": "2030-01-01T00:00:00+00:00"})
        assert rec["id"] == "22222222-2222-2222-2222-222222222222"
        assert isinstance(rec["created_at"], str)

    def test_create_with_no_writable_columns_raises(self):
        with pytest.raises(ValueError):
            OtpRepository().create({"not_a_column": 1})

    def test_get_latest_active_filters_and_orders(self):
        captured = {}
        with patch("app.db.repositories.otp_repository.pg_fetch_one",
                   side_effect=lambda q, p: captured.update(q=q, p=p) or None):
            assert OtpRepository().get_latest_active("a@x.com", "signup") is None

        q = _norm(captured["q"])
        assert "used_at is null" in q
        assert "order by created_at desc" in q
        assert "limit 1" in q
        assert captured["p"] == {"email": "a@x.com", "purpose": "signup"}

    def test_invalidate_active_uses_db_clock(self):
        """now() server-side rather than an application timestamp — removes the
        client/server clock-skew the PostgREST implementation had."""
        captured = {}
        with patch("app.db.repositories.otp_repository.pg_execute",
                   side_effect=lambda q, p: captured.update(q=q, p=p)):
            OtpRepository().invalidate_active("a@x.com", "signup")

        q = _norm(captured["q"])
        assert "set used_at = now()" in q
        assert "used_at is null" in q

    def test_mark_used_uses_db_clock(self):
        captured = {}
        with patch("app.db.repositories.otp_repository.pg_execute",
                   side_effect=lambda q, p: captured.update(q=q)):
            OtpRepository().mark_used("otp-1")
        assert "now()" in _norm(captured["q"])


# ---------------------------------------------------------------------------
# RefreshTokenRepository
# ---------------------------------------------------------------------------

class TestRefreshTokenRepository:
    def test_create_only_names_known_columns(self):
        captured = {}

        def fake(query, params):
            captured["q"], captured["p"] = query, params
            return {"id": uuid.uuid4()}

        with patch("app.db.repositories.refresh_token_repository.pg_fetch_one", side_effect=fake):
            RefreshTokenRepository().create({
                "user_id": "u1", "token_hash": "th", "family_id": "fam",
                "expires_at": "2030-01-01T00:00:00+00:00", "bogus": "x",
            })

        assert "bogus" not in captured["q"]
        assert set(captured["p"]) == {"user_id", "token_hash", "family_id", "expires_at"}

    def test_get_by_hash_miss_returns_none(self):
        with patch("app.db.repositories.refresh_token_repository.pg_fetch_one", return_value=None):
            assert RefreshTokenRepository().get_by_hash("nope") is None

    def test_revoke_family_only_touches_unrevoked(self):
        """Re-revoking a family must not overwrite the original revoked_at —
        that timestamp is the forensic record of when reuse was detected."""
        captured = {}
        with patch("app.db.repositories.refresh_token_repository.pg_execute",
                   side_effect=lambda q, p: captured.update(q=q, p=p)):
            RefreshTokenRepository().revoke_family("fam-1")

        q = _norm(captured["q"])
        assert "revoked_at is null" in q
        assert "family_id = %s" in q

    def test_revoke_all_for_user_only_touches_unrevoked(self):
        captured = {}
        with patch("app.db.repositories.refresh_token_repository.pg_execute",
                   side_effect=lambda q, p: captured.update(q=q)):
            RefreshTokenRepository().revoke_all_for_user("u1")
        assert "revoked_at is null" in _norm(captured["q"])


# ---------------------------------------------------------------------------
# AuthAuditRepository
# ---------------------------------------------------------------------------

class TestAuthAuditRepository:
    def test_writes_expected_payload(self):
        captured = {}
        with patch("app.db.repositories.auth_audit_repository.pg_execute",
                   side_effect=lambda q, p: captured.update(q=q, p=p)):
            AuthAuditRepository().log(event="login", success=True, user_id="u1",
                                      ip="1.2.3.4", user_agent="ua")

        assert "insert into auth_audit" in _norm(captured["q"])
        assert captured["p"] == {"event": "login", "success": True, "user_id": "u1",
                                 "ip": "1.2.3.4", "user_agent": "ua"}

    def test_never_raises_when_the_write_fails(self):
        """Audit logging must never break the auth flow it is observing — this
        is the property that let a malformed user_id be passed safely."""
        with patch("app.db.repositories.auth_audit_repository.pg_execute",
                   side_effect=RuntimeError("db down")):
            AuthAuditRepository().log(event="login", success=False)  # must not raise

    def test_never_raises_on_malformed_user_id(self):
        import psycopg2
        with patch("app.db.repositories.auth_audit_repository.pg_execute",
                   side_effect=psycopg2.Error("invalid input syntax for type uuid")):
            AuthAuditRepository().log(event="login", success=True, user_id="not-a-uuid")


# ---------------------------------------------------------------------------
# No Supabase left anywhere in the runtime
# ---------------------------------------------------------------------------

class TestSupabaseFullyRemoved:
    def test_app_imports_without_loading_supabase(self):
        """The `supabase` package is no longer a dependency. If any module
        re-introduces an import of it, this fails rather than the container
        crash-looping on a missing package after the next clean build."""
        import sys
        import app.main  # noqa: F401

        assert not [m for m in sys.modules if m == "supabase" or m.startswith("supabase.")]
        assert not [m for m in sys.modules if m == "postgrest" or m.startswith("postgrest.")]

    def test_db_package_exports_no_supabase_helpers(self):
        import app.db as db

        for name in ("get_supabase_client", "get_auth_supabase_client",
                     "reset_supabase_client", "SupabaseClientError"):
            assert not hasattr(db, name), f"app.db still exports {name}"

    def test_supabase_module_is_gone(self):
        with pytest.raises(ImportError):
            import app.db.supabase  # noqa: F401
