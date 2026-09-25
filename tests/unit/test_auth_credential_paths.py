"""Credential-path hardening: reset/verify limits (C-05), atomic OTP and
refresh use (C-16), non-consuming reset verification (C-26), client IP behind
proxies (C-13), JWT key checks (C-06)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core import rate_limit
from app.core.rate_limit import TokenBucket
from app.services import auth_service

EMAIL = "someone@example.com"


@pytest.fixture(autouse=True)
def _local_limiters(monkeypatch):
    """Fresh in-process limiters for each test; never a real Redis."""
    monkeypatch.setattr(rate_limit, "_get_redis", lambda: None)
    for name, rate, per in [
        ("_otp_request_email_ip_limiter", 1, 3600.0),
        ("_otp_request_email_limiter", 2, 3600.0),
        ("_otp_request_ip_limiter", 3, 3600.0),
        ("_otp_verify_email_ip_limiter", 100, 3600.0),
        ("_otp_verify_email_limiter", 3, 3600.0),
        ("_otp_verify_ip_limiter", 100, 3600.0),
        ("_otp_daily_email_ip_limiter", 1, 86400.0),
        ("_otp_daily_email_limiter", 1, 86400.0),
        ("_otp_verify_daily_email_ip_limiter", 100, 86400.0),
        ("_otp_verify_daily_limiter", 100, 86400.0),
    ]:
        monkeypatch.setattr(auth_service, name, TokenBucket(rate=rate, per=per, name=name))


def _otp_record(**overrides):
    record = {
        "id": "otp-1",
        "user_id": "user-1",
        "attempts": 0,
        "code_hash": "hash",
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }
    record.update(overrides)
    return record


# --- forgot-password (C-05) -------------------------------------------------

class TestForgotPassword:
    def test_per_email_limit_returns_429_with_retry_after(self):
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users):
            auth_service.forgot_password(email=EMAIL)
            auth_service.forgot_password(email=EMAIL)
            with pytest.raises(HTTPException) as exc:
                auth_service.forgot_password(email=EMAIL)
        assert exc.value.status_code == 429
        assert int(exc.value.headers["Retry-After"]) > 0

    def test_limit_ignores_address_case(self):
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users):
            auth_service.forgot_password(email=EMAIL)
            auth_service.forgot_password(email=EMAIL.upper())
            with pytest.raises(HTTPException):
                auth_service.forgot_password(email="Someone@Example.com")

    def test_known_and_unknown_addresses_are_limited_alike(self):
        """The limit applies before the lookup, so a 429 reveals nothing."""
        users = MagicMock()
        users.get_by_email.side_effect = lambda e: {"id": "u1"} if e == "real@example.com" else None
        with patch.object(auth_service, "get_user_repository", return_value=users), \
             patch.object(auth_service, "_issue_otp"):
            for email in ("real@example.com", "ghost@example.com"):
                auth_service.forgot_password(email=email)
                auth_service.forgot_password(email=email)
                with pytest.raises(HTTPException) as exc:
                    auth_service.forgot_password(email=email)
                assert exc.value.status_code == 429

    def test_attacker_ip_hits_its_own_limit_before_the_owners(self):
        """The tight tier is per e-mail+IP, so one attacker can't lock the owner out."""
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users):
            auth_service.forgot_password(email=EMAIL, ip="198.51.100.66")
            with pytest.raises(HTTPException):
                auth_service.forgot_password(email=EMAIL, ip="198.51.100.66")
            auth_service.forgot_password(email=EMAIL, ip="203.0.113.1")  # the owner, elsewhere

    def test_proxy_address_does_not_use_the_ip_tiers(self):
        """If IP resolution collapses to a proxy, per-IP keys would be shared by everyone."""
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users):
            for i in range(5):
                auth_service.forgot_password(email=f"p{i}@example.com", ip="10.0.0.5")

    def test_resend_uses_the_same_request_limits(self):
        """C-05: /resend-otp must not be a way around the forgot-password limits."""
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users), \
             patch.object(auth_service, "_otp_resend_limiter", TokenBucket(rate=100, per=60.0, name="r")):
            auth_service.resend_otp(email=EMAIL, purpose="password_reset")
            auth_service.resend_otp(email=EMAIL, purpose="password_reset")
            with pytest.raises(HTTPException) as exc:
                auth_service.resend_otp(email=EMAIL, purpose="password_reset")
        assert exc.value.status_code == 429

    def test_per_ip_limit_spans_addresses(self):
        users = MagicMock()
        users.get_by_email.return_value = None
        with patch.object(auth_service, "get_user_repository", return_value=users):
            for i in range(3):
                auth_service.forgot_password(email=f"a{i}@example.com", ip="203.0.113.9")
            with pytest.raises(HTTPException) as exc:
                auth_service.forgot_password(email="a9@example.com", ip="203.0.113.9")
        assert exc.value.status_code == 429

    def test_daily_code_cap_is_silent_for_reset(self):
        """Hitting the daily cap looks like success, so it can't reveal the account."""
        users = MagicMock()
        users.get_by_email.return_value = {"id": "u1"}
        otp_repo = MagicMock()
        with patch.object(auth_service, "get_user_repository", return_value=users), \
             patch.object(auth_service, "get_otp_repository", return_value=otp_repo), \
             patch.object(auth_service, "send_otp_email"):
            auth_service.forgot_password(email=EMAIL)      # issues the one code allowed
            auth_service.forgot_password(email=EMAIL)      # capped, but no error
        assert otp_repo.create.call_count == 1


def test_one_outsider_cannot_use_up_the_owners_daily_codes(monkeypatch):
    """The daily code cap is tiered too: an attacker's IP hits its own cap first."""
    monkeypatch.setattr(auth_service, "_otp_daily_email_limiter", TokenBucket(rate=3, per=86400.0, name="d"))
    otp_repo = MagicMock()
    with patch.object(auth_service, "get_otp_repository", return_value=otp_repo), \
         patch.object(auth_service, "send_otp_email"):
        auth_service._issue_otp(email=EMAIL, purpose="password_reset", user_id="u1", ip="198.51.100.66")
        with pytest.raises(HTTPException):
            auth_service._issue_otp(email=EMAIL, purpose="password_reset", user_id="u1", ip="198.51.100.66")
        auth_service._issue_otp(email=EMAIL, purpose="password_reset", user_id="u1", ip="203.0.113.1")
    assert otp_repo.create.call_count == 2


def test_daily_guess_cap_bounds_guesses_however_many_ips(monkeypatch):
    monkeypatch.setattr(auth_service, "_otp_verify_daily_limiter", TokenBucket(rate=2, per=86400.0, name="v"))
    monkeypatch.setattr(auth_service, "_otp_verify_email_limiter", TokenBucket(rate=100, per=3600.0, name="e"))
    repo = MagicMock()
    repo.get_latest_active.return_value = _otp_record()
    repo.claim_attempt.return_value = 1
    with patch.object(auth_service, "get_otp_repository", return_value=repo), \
         patch.object(auth_service, "get_user_repository", return_value=MagicMock()), \
         patch.object(auth_service, "get_auth_audit_repository", return_value=MagicMock()), \
         patch.object(auth_service, "verify_otp_code", return_value=False):
        for ip in ("198.51.100.1", "198.51.100.2"):
            with pytest.raises(HTTPException) as exc:
                auth_service.verify_otp(email=EMAIL, code="000000", purpose="password_reset", ip=ip)
            assert exc.value.status_code == 400
        with pytest.raises(HTTPException) as exc:
            auth_service.verify_otp(email=EMAIL, code="000000", purpose="password_reset", ip="198.51.100.3")
    assert exc.value.status_code == 429


def test_daily_code_cap_is_a_429_for_other_purposes():
    with patch.object(auth_service, "get_otp_repository", return_value=MagicMock()), \
         patch.object(auth_service, "send_otp_email"):
        auth_service._issue_otp(email=EMAIL, purpose="password_change", user_id="u1")
        with pytest.raises(HTTPException) as exc:
            auth_service._issue_otp(email=EMAIL, purpose="password_change", user_id="u1")
    assert exc.value.status_code == 429


# --- verify-otp (C-05, C-16, C-26) -----------------------------------------

class TestVerifyOtp:
    def _run(self, otp_repo, *, code_ok=True, **kwargs):
        with patch.object(auth_service, "get_otp_repository", return_value=otp_repo), \
             patch.object(auth_service, "get_user_repository", return_value=MagicMock()), \
             patch.object(auth_service, "get_auth_audit_repository", return_value=MagicMock()), \
             patch.object(auth_service, "verify_otp_code", return_value=code_ok):
            return auth_service.verify_otp(email=EMAIL, code="123456", purpose="password_reset", **kwargs)

    def _repo(self, record=None, claim=1, mark=True):
        repo = MagicMock()
        repo.get_latest_active.return_value = record or _otp_record()
        repo.claim_attempt.return_value = claim
        repo.mark_used.return_value = mark
        return repo

    def test_guess_limit_holds_across_fresh_codes(self):
        """Each new code used to bring five new guesses; the per-email limit does not reset."""
        for i in range(3):
            with pytest.raises(HTTPException) as exc:
                self._run(self._repo(_otp_record(id=f"otp-{i}")), code_ok=False)
            assert exc.value.status_code == 400
        with pytest.raises(HTTPException) as exc:
            self._run(self._repo(_otp_record(id="otp-new")), code_ok=False)
        assert exc.value.status_code == 429
        assert "Retry-After" in exc.value.headers

    def test_attempt_is_claimed_atomically_before_checking(self):
        repo = self._repo(claim=None)
        with pytest.raises(HTTPException) as exc:
            self._run(repo)
        assert exc.value.status_code == 429
        repo.claim_attempt.assert_called_once_with("otp-1", auth_service.settings.otp_max_attempts)

    def test_code_redeemed_by_a_parallel_request_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            self._run(self._repo(mark=False))
        assert exc.value.status_code == 400

    def test_check_without_consuming_leaves_the_code_usable(self):
        repo = self._repo()
        assert self._run(repo, consume=False) == {"email": EMAIL}
        repo.mark_used.assert_not_called()


def test_verify_otp_route_does_not_consume_reset_codes():
    """C-26: /verify-otp with purpose=password_reset only checks the code."""
    from app.api import auth as auth_routes

    request = MagicMock()
    request.client.host = "203.0.113.5"
    request.headers = {}
    payload = MagicMock(email=EMAIL, code="123456", purpose="password_reset")
    with patch.object(auth_routes.auth_service, "verify_otp", return_value={"email": EMAIL}) as verify:
        auth_routes.verify_otp(payload, request, MagicMock())
    assert verify.call_args.kwargs["consume"] is False


# --- refresh rotation (C-16) ------------------------------------------------

def _refresh(used_at=None, claimed=True, user_agent="Firefox"):
    tokens = MagicMock()
    tokens.get_by_hash.return_value = {
        "id": "t1", "family_id": "f1", "user_id": "u1", "revoked_at": None, "used_at": used_at,
        "user_agent": "Firefox",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }
    tokens.mark_used.return_value = claimed
    with patch.object(auth_service, "get_refresh_token_repository", return_value=tokens), \
         patch.object(auth_service, "get_user_repository", return_value=MagicMock()), \
         patch.object(auth_service, "get_auth_audit_repository", return_value=MagicMock()), \
         patch.object(auth_service.session_service, "end_session"):
        with pytest.raises(HTTPException) as exc:
            auth_service.refresh_access_token(refresh_token="rt", user_agent=user_agent)
    return exc.value, tokens


def test_parallel_refresh_losing_the_race_keeps_the_session():
    """Two tabs refreshing at once: the loser gets a retryable 401, nothing is revoked."""
    error, tokens = _refresh(used_at=None, claimed=False)
    assert error.status_code == 401
    assert error.detail["code"] == "refresh_superseded"
    tokens.revoke_family.assert_not_called()


def test_token_reused_just_after_rotation_keeps_the_session():
    just_now = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    error, tokens = _refresh(used_at=just_now)
    assert error.detail["code"] == "refresh_superseded"
    tokens.revoke_family.assert_not_called()


def test_reuse_from_another_browser_inside_the_window_revokes_the_family():
    """Inside the grace window only the same browser is let off; a different one is a thief."""
    just_now = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    error, tokens = _refresh(used_at=just_now, user_agent="curl/8.0")
    assert error.status_code == 401
    tokens.revoke_family.assert_called_once_with("f1")


def test_token_replayed_later_revokes_the_family():
    long_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    error, tokens = _refresh(used_at=long_ago)
    assert error.status_code == 401
    tokens.revoke_family.assert_called_once_with("f1")
    tokens.mark_used.assert_not_called()


# --- client IP (C-13) -------------------------------------------------------

def _request(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "client": (peer, 1234), "headers": headers})


@pytest.mark.parametrize("peer, forwarded, expected", [
    ("198.51.100.7", None, "198.51.100.7"),                        # direct, untrusted peer
    ("198.51.100.7", "1.2.3.4", "198.51.100.7"),                   # untrusted peer can't forward
    ("127.0.0.1", "198.51.100.7", "198.51.100.7"),                 # sidecar in front
    ("127.0.0.1", "198.51.100.7, 10.1.2.3", "198.51.100.7"),       # sidecar + ingress hop
    ("127.0.0.1", "6.6.6.6, 198.51.100.7, 10.1.2.3", "198.51.100.7"),  # spoofed left entry ignored
    ("10.1.2.3", None, "10.1.2.3"),                                # internal, nothing forwarded
])
def test_client_ip(peer, forwarded, expected):
    from app.core.client_ip import client_ip

    assert client_ip(_request(peer, forwarded)) == expected


# --- JWT key (C-06) ---------------------------------------------------------

class TestJwtKey:
    def test_empty_key_rejects_tokens_instead_of_erroring(self, monkeypatch):
        from app.core import security

        monkeypatch.setattr(security.settings, "jwt_signing_key", "")
        with pytest.raises(security.TokenInvalidError):
            security.decode_access_token("a.b.c")
        with pytest.raises(security.TokenInvalidError):
            security.create_access_token(user_id="u1", persona="admin")

    def test_short_key_is_a_startup_problem(self, monkeypatch):
        from app.core import security

        monkeypatch.setattr(security.settings, "jwt_signing_key", "short-key")
        assert "shorter than" in security.jwt_key_problem()

    @pytest.mark.parametrize("key", ["", "k" * 32])
    def test_empty_or_long_key_is_not_a_startup_problem(self, monkeypatch, key):
        from app.core import security

        monkeypatch.setattr(security.settings, "jwt_signing_key", key)
        assert security.jwt_key_problem() is None

    def test_malformed_token_is_invalid_not_a_server_error(self, monkeypatch):
        from app.core import security

        monkeypatch.setattr(security.settings, "jwt_signing_key", "k" * 32)
        with pytest.raises(security.TokenInvalidError):
            security.decode_access_token("not-a-jwt")
