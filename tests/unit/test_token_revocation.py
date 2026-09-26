"""Access tokens are revocable and carry no authority the account lacks (C-04)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core import auth
from app.core.auth import UserState, get_current_user, require_role
from app.core.security import create_access_token

KEY = "test-signing-key-that-is-at-least-32-bytes-long"

# Captured at import, before the autouse user_states fixture stubs it.
_REAL_LOAD_USER_STATE = auth._load_user_state


@pytest.fixture(autouse=True)
def _jwt_signing_key(monkeypatch):
    from app.core import security

    monkeypatch.setattr(security.settings, "jwt_signing_key", KEY, raising=False)


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _call(token: str):
    return get_current_user(credentials=_bearer(token))


# --- get_current_user checks the account ------------------------------------

def test_token_minted_before_a_version_bump_is_rejected_as_expired(user_states):
    token = create_access_token(user_id="u1", persona="general", token_version=0)
    user_states["u1"] = UserState(token_version=1, persona="general", active=True)

    with pytest.raises(HTTPException) as exc:
        _call(token)
    assert exc.value.status_code == 401
    assert exc.value.detail["code"] == "token_expired"   # the client refreshes


def test_token_without_a_version_claim_is_rejected(user_states):
    """Tokens minted before this change (including the ones committed in k6/) stop working."""
    now = datetime.now(timezone.utc)
    legacy = jwt.encode(
        {"sub": "u1", "persona": "admin", "iss": "ai4ia-backend", "iat": now, "exp": now + timedelta(days=30)},
        KEY, algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        _call(legacy)
    assert exc.value.detail["code"] == "token_expired"


def test_current_version_is_accepted(user_states):
    user_states["u1"] = UserState(token_version=3, persona="expert", active=True)
    user = _call(create_access_token(user_id="u1", persona="expert", token_version=3))
    assert (user.id, user.persona) == ("u1", "expert")


@pytest.mark.parametrize("state", [None, UserState(token_version=0, persona="general", active=False)])
def test_deleted_or_deactivated_account_is_rejected(user_states, state):
    user_states["u1"] = state
    with pytest.raises(HTTPException) as exc:
        _call(create_access_token(user_id="u1", persona="general"))
    assert exc.value.status_code == 401


def test_persona_comes_from_the_account_not_the_token(user_states):
    """A demoted admin keeps an admin-claim token; the account row decides."""
    user_states["u1"] = UserState(token_version=0, persona="general", active=True)
    user = _call(create_access_token(user_id="u1", persona="admin"))

    assert user.persona == "general"
    with pytest.raises(HTTPException) as exc:
        require_role("admin")(user)
    assert exc.value.status_code == 403


def test_database_outage_is_a_503_not_a_sign_out(monkeypatch):
    def _down(user_id):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(auth, "_load_user_state", _down)
    with pytest.raises(HTTPException) as exc:
        _call(create_access_token(user_id="u1", persona="general"))
    assert exc.value.status_code == 503


def test_user_state_is_cached_until_forgotten(monkeypatch):
    calls = []
    monkeypatch.setattr(auth, "_load_user_state", lambda uid: calls.append(uid) or UserState(0, "general", True))

    auth.user_state("u1")
    auth.user_state("u1")
    assert calls == ["u1"]

    auth.forget_user_state("u1")
    auth.user_state("u1")
    assert calls == ["u1", "u1"]


def test_cached_state_expires(monkeypatch):
    calls = []
    monkeypatch.setattr(auth, "_load_user_state", lambda uid: calls.append(uid) or UserState(0, "general", True))
    clock = [1000.0]
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock[0])

    auth.user_state("u1")
    clock[0] += auth._USER_STATE_TTL + 1
    auth.user_state("u1")
    assert len(calls) == 2


# --- what bumps the version -------------------------------------------------

def test_revoke_access_tokens_bumps_and_forgets(monkeypatch):
    from app.services import auth_service

    users = MagicMock()
    forgotten = []
    monkeypatch.setattr(auth, "forget_user_state", forgotten.append)
    with patch.object(auth_service, "get_user_repository", return_value=users):
        auth_service.revoke_access_tokens("u1")
    users.bump_token_version.assert_called_once_with("u1")
    assert forgotten == ["u1"]


class TestAdminActionsRevoke:
    @pytest.fixture
    def svc(self, monkeypatch):
        import app.services.admin_user_service as svc

        self.users = MagicMock()
        self.users.get_by_id.return_value = {"id": "u1", "persona": "admin", "email_id": "a@x.com",
                                             "username": "a", "status": "active"}
        self.revoked = []
        monkeypatch.setattr(svc, "get_user_repository", lambda: self.users)
        monkeypatch.setattr(svc, "get_refresh_token_repository", MagicMock)
        monkeypatch.setattr(svc, "get_auth_audit_repository", MagicMock)
        monkeypatch.setattr(svc, "revoke_access_tokens", self.revoked.append)
        monkeypatch.setattr(svc, "to_admin_record", lambda u: u)
        return svc

    def test_demotion_revokes(self, svc):
        svc.update_user("u1", first_name=None, last_name=None, email=None, persona="general")
        assert self.revoked == ["u1"]

    def test_unchanged_persona_does_not_revoke(self, svc):
        svc.update_user("u1", first_name="New", last_name=None, email=None, persona="admin")
        assert self.revoked == []

    def test_deactivation_revokes(self, svc):
        svc.deactivate_user("u1")
        assert self.revoked == ["u1"]

    def test_admin_password_reset_revokes(self, svc):
        svc.admin_reset_password("u1", "N3w-password-long")
        assert self.revoked == ["u1"]


def test_password_reset_revokes_access_tokens(monkeypatch):
    from app.services import auth_service

    users = MagicMock()
    users.get_by_email.return_value = {"id": "u1"}
    revoked = []
    monkeypatch.setattr(auth_service, "revoke_access_tokens", revoked.append)
    with patch.object(auth_service, "verify_otp"), \
         patch.object(auth_service, "get_user_repository", return_value=users), \
         patch("app.db.repositories.get_refresh_token_repository", return_value=MagicMock()), \
         patch.object(auth_service.session_service, "end_all_sessions_for_user"), \
         patch.object(auth_service, "send_password_changed_notice", create=True):
        auth_service.reset_password(email="a@x.com", code="123456", new_password="N3w-password-long")
    assert revoked == ["u1"]


# --- tokens are minted with the account's version ---------------------------

def test_refresh_mints_the_current_version_and_refuses_deactivated_accounts():
    from app.services import auth_service

    def _run(user):
        tokens = MagicMock()
        tokens.get_by_hash.return_value = {
            "id": "t1", "family_id": "f1", "user_id": "u1", "revoked_at": None, "used_at": None,
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        }
        tokens.mark_used.return_value = True
        users = MagicMock()
        users.get_by_id.return_value = user
        with patch.object(auth_service, "get_refresh_token_repository", return_value=tokens), \
             patch.object(auth_service, "get_user_repository", return_value=users), \
             patch.object(auth_service, "get_auth_audit_repository", return_value=MagicMock()), \
             patch.object(auth_service, "_issue_refresh_token", return_value=("rt2", "f1")), \
             patch.object(auth_service.session_service, "touch_session"), \
             patch.object(auth_service, "create_access_token", return_value="at") as mint:
            return auth_service.refresh_access_token(refresh_token="rt"), mint, tokens

    user = {"id": "u1", "username": "a", "persona": "user", "status": "active", "token_version": 4}
    _, mint, _ = _run(user)
    assert mint.call_args.kwargs["token_version"] == 4

    with pytest.raises(HTTPException) as exc:
        _run({**user, "status": "deactivated"})
    assert exc.value.detail["code"] == "account_deactivated"


def test_bump_token_version_is_one_atomic_update():
    from app.db.repositories.user_repository import UserRepository

    captured = {}
    with patch("app.db.repositories.user_repository.pg_fetch_one",
               side_effect=lambda q, p: captured.update(q=" ".join(q.split()).lower()) or {"token_version": 2}):
        assert UserRepository().bump_token_version("u1") == 2
    assert "token_version = token_version + 1" in captured["q"]
    assert "returning token_version" in captured["q"]


def test_long_access_token_lifetime_is_flagged_at_start_up():
    from app.core.settings import Settings

    warnings = Settings(access_token_ttl_seconds=2592000).validate_deployment_env()
    assert any("ACCESS_TOKEN_TTL_SECONDS" in w for w in warnings)
    assert not any("ACCESS_TOKEN_TTL_SECONDS" in w for w in Settings(access_token_ttl_seconds=900).validate_deployment_env())


def test_token_newer_than_the_cache_is_re_read_not_rejected(monkeypatch):
    """Another replica bumped the version and minted n+1; this one still caches n."""
    versions = iter([0, 1])
    monkeypatch.setattr(auth, "_load_user_state", lambda uid: UserState(next(versions), "general", True))
    auth.user_state("u1")  # caches version 0

    user = _call(create_access_token(user_id="u1", persona="general", token_version=1))
    assert user.id == "u1"


def test_load_started_before_a_revocation_does_not_cache_stale_state(monkeypatch):
    def _load(uid):
        auth.forget_user_state(uid)  # a revocation lands while this load runs
        return UserState(0, "general", True)

    monkeypatch.setattr(auth, "_load_user_state", _load)
    auth.user_state("u1")
    assert "u1" not in auth._user_state_cache


def test_real_account_row_mapping(monkeypatch):
    """The fixture stubs the loader everywhere; check the real one once."""
    rows = {
        "legacy": {"id": "legacy", "persona": None, "status": "active"},
        "admin": {"id": "admin", "persona": "admin", "status": "active", "token_version": 7},
        "gone": {"id": "gone", "persona": "user", "status": "deactivated", "token_version": 2},
        "locked": {"id": "locked", "persona": "researcher", "status": "locked"},
    }
    users = MagicMock()
    users.get_by_id.side_effect = rows.get
    with patch("app.db.repositories.get_user_repository", return_value=users):
        assert _REAL_LOAD_USER_STATE("legacy") == UserState(0, "", True)
        assert _REAL_LOAD_USER_STATE("admin") == UserState(7, "admin", True)
        assert _REAL_LOAD_USER_STATE("gone") == UserState(2, "general", False)
        assert _REAL_LOAD_USER_STATE("locked") == UserState(0, "expert", True)  # a login lockout keeps sessions
        assert _REAL_LOAD_USER_STATE("missing") is None
