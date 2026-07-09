"""Unit tests for app.core.security — password hashing, JWT, refresh tokens, OTP."""
from __future__ import annotations

import time

import pytest

from app.core.security import (
    TokenExpiredError,
    TokenInvalidError,
    create_access_token,
    decode_access_token,
    generate_otp_code,
    generate_refresh_token,
    hash_otp_code,
    hash_password,
    hash_refresh_token,
    parse_timestamp,
    verify_legacy_plaintext_password,
    verify_otp_code,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify_roundtrip(self):
        h = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", h)

    def test_wrong_password_rejected(self):
        h = hash_password("correct horse battery staple")
        assert not verify_password("wrong password", h)

    def test_hash_is_not_plaintext(self):
        h = hash_password("mypassword")
        assert h != "mypassword"
        assert h.startswith("$argon2id$")

    def test_same_password_hashes_differently_each_time(self):
        """Argon2 salts each hash — two hashes of the same password must differ."""
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2
        assert verify_password("samepassword", h1)
        assert verify_password("samepassword", h2)

    def test_verify_rejects_garbage_hash(self):
        assert not verify_password("anything", "not-a-real-hash")


class TestLegacyPlaintextCompare:
    def test_matching_plaintext_accepted(self):
        assert verify_legacy_plaintext_password("secret123", "secret123")

    def test_mismatched_plaintext_rejected(self):
        assert not verify_legacy_plaintext_password("secret123", "other")

    def test_empty_stored_plaintext_rejected(self):
        assert not verify_legacy_plaintext_password("secret123", "")


class TestAccessTokens:
    def test_create_and_decode_roundtrip(self):
        token = create_access_token(user_id="user-abc", persona="admin")
        claims = decode_access_token(token)
        assert claims["sub"] == "user-abc"
        assert claims["persona"] == "admin"

    def test_tampered_token_rejected(self):
        token = create_access_token(user_id="user-abc", persona="general")
        tampered = token[:-1] + ("x" if token[-1] != "x" else "y")
        with pytest.raises(TokenInvalidError):
            decode_access_token(tampered)

    def test_garbage_token_rejected(self):
        with pytest.raises(TokenInvalidError):
            decode_access_token("not.a.jwt")

    def test_expired_token_rejected(self, monkeypatch):
        from app.core import security as security_module

        monkeypatch.setattr(security_module.settings, "access_token_ttl_seconds", 0, raising=False)
        token = create_access_token(user_id="user-abc", persona="general")
        time.sleep(1.1)
        with pytest.raises(TokenExpiredError):
            decode_access_token(token)

    def test_wrong_signing_key_rejected(self, monkeypatch):
        from app.core import security as security_module

        token = create_access_token(user_id="user-abc", persona="general")
        monkeypatch.setattr(security_module.settings, "jwt_signing_key", "a-different-key", raising=False)
        with pytest.raises(TokenInvalidError):
            decode_access_token(token)


class TestRefreshTokens:
    def test_generate_is_random_each_time(self):
        a = generate_refresh_token()
        b = generate_refresh_token()
        assert a != b

    def test_hash_is_deterministic(self):
        token = generate_refresh_token()
        assert hash_refresh_token(token) == hash_refresh_token(token)

    def test_hash_differs_for_different_tokens(self):
        a, b = generate_refresh_token(), generate_refresh_token()
        assert hash_refresh_token(a) != hash_refresh_token(b)

    def test_hash_is_not_the_raw_token(self):
        token = generate_refresh_token()
        assert hash_refresh_token(token) != token


class TestOtpCodes:
    def test_generated_code_is_six_digits(self):
        code = generate_otp_code()
        assert len(code) == 6
        assert code.isdigit()

    def test_verify_correct_code(self):
        code = generate_otp_code()
        h = hash_otp_code(code)
        assert verify_otp_code(code, h)

    def test_verify_rejects_wrong_code(self):
        code = generate_otp_code()
        h = hash_otp_code(code)
        wrong = "000000" if code != "000000" else "111111"
        assert not verify_otp_code(wrong, h)

    def test_hash_does_not_reveal_code(self):
        code = "123456"
        h = hash_otp_code(code)
        assert code not in h


class TestParseTimestamp:
    def test_parses_full_precision_microseconds(self):
        dt = parse_timestamp("2026-07-08T08:11:15.069960+00:00")
        assert dt.year == 2026 and dt.month == 7 and dt.day == 8

    def test_parses_short_precision_microseconds(self):
        """Postgres/Supabase can emit fewer than 6 fractional digits
        (e.g. '.06996'), which stdlib fromisoformat() rejects pre-3.11."""
        dt = parse_timestamp("2026-07-08T08:11:15.06996+00:00")
        assert dt.microsecond == 69960

    def test_parses_z_suffix(self):
        dt = parse_timestamp("2026-07-08T08:11:15Z")
        assert dt.tzinfo is not None

    def test_naive_timestamp_gets_utc_tzinfo(self):
        dt = parse_timestamp("2026-07-08T08:11:15")
        assert dt.tzinfo is not None
