from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from dateutil import parser as _dateutil_parser

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

from app.core.settings import settings

# Common composition policy: minimum length + at least one of each character
# class (uppercase, lowercase, digit, special). Argon2id + rate limiting +
# lockout already defend against brute force; this is the baseline users
# expect and auditors check for.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
_UPPERCASE_RE = re.compile(r"[A-Z]")
_LOWERCASE_RE = re.compile(r"[a-z]")
_DIGIT_RE = re.compile(r"[0-9]")
_SPECIAL_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]""")

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]{3,32}$")


def normalize_identifier(value: str) -> str:
    """Canonical form for a username or email: trimmed + lowercased, so
    lookups and uniqueness checks aren't sensitive to case or incidental
    whitespace (e.g. autofill/copy-paste padding). Existing accounts created
    before this was enforced are matched case-insensitively at the query
    layer (see UserRepository) rather than requiring a backfill."""
    return value.strip().lower()


def validate_username_format(username: str) -> str:
    if not USERNAME_RE.match(username):
        raise ValueError("Username may only contain letters, numbers, '.', '_', '-'")
    return username


def validate_password_strength(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters")
    if not _UPPERCASE_RE.search(password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not _LOWERCASE_RE.search(password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not _DIGIT_RE.search(password):
        raise ValueError("Password must contain at least one number")
    if not _SPECIAL_RE.search(password):
        raise ValueError("Password must contain at least one special character")
    return password


def parse_timestamp(value: str) -> datetime:
    """Parse a Postgres/Supabase timestamptz string. Postgres emits a
    variable number of fractional-second digits (e.g. '.06996' — 5 digits,
    not 6), which datetime.fromisoformat() rejects on Python < 3.11.
    dateutil handles any fractional precision and both 'Z' and '+00:00'
    offsets."""
    dt = _dateutil_parser.isoparse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_password_hasher: PasswordHasher | None = None


def _get_password_hasher() -> PasswordHasher:
    """Constructed lazily (not at import time) so tests that monkeypatch
    settings after import still take effect."""
    global _password_hasher
    if _password_hasher is None:
        _password_hasher = PasswordHasher(
            time_cost=settings.argon2_time_cost,
            memory_cost=settings.argon2_memory_cost_kib,
            parallelism=settings.argon2_parallelism,
        )
    return _password_hasher


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return _get_password_hasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _get_password_hasher().verify(password_hash, password)
    except (VerifyMismatchError, InvalidHash):
        return False


def verify_legacy_plaintext_password(password: str, stored_plaintext: str) -> bool:
    """Constant-time compare for the pre-migration plaintext `password` column.
    Only used once, to authenticate a legacy row before it is upgraded to a
    proper Argon2id hash."""
    return hmac.compare_digest(password, stored_plaintext)


# ---------------------------------------------------------------------------
# Access tokens (JWT)
# ---------------------------------------------------------------------------

class TokenError(Exception):
    """Base for access-token verification failures."""


class TokenExpiredError(TokenError):
    pass


class TokenInvalidError(TokenError):
    pass


def _signing_key() -> str:
    """The JWT key. Refuses an empty one here rather than relying on the JWT
    library to, since an empty HMAC key would let anyone sign tokens."""
    if not settings.jwt_signing_key:
        raise TokenInvalidError("JWT_SIGNING_KEY is not set")
    return settings.jwt_signing_key


def jwt_key_problem() -> str | None:
    """Why the configured key is unsafe to start with, or None. A missing key
    only disables sign-in; a short one can be brute-forced from any token."""
    key = settings.jwt_signing_key
    if key and len(key.encode("utf-8")) < settings.jwt_min_key_bytes:
        return (
            f"JWT_SIGNING_KEY is shorter than {settings.jwt_min_key_bytes} bytes; "
            "use a longer random key"
        )
    return None


def create_access_token(*, user_id: str, persona: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "persona": persona,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + timedelta(seconds=settings.access_token_ttl_seconds),
    }
    return jwt.encode(payload, _signing_key(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    key = _signing_key()
    try:
        return jwt.decode(
            token,
            key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("access token expired") from exc
    except jwt.PyJWTError as exc:
        # Any other library error (bad signature, bad key, malformed) is a
        # rejected token, never a 500.
        raise TokenInvalidError("access token invalid") from exc


# ---------------------------------------------------------------------------
# Refresh tokens — opaque random string; only the SHA-256 hash is stored.
# ---------------------------------------------------------------------------

def generate_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# OTP codes — 6-digit numeric; only the SHA-256 hash is stored.
# ---------------------------------------------------------------------------

def generate_otp_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def verify_otp_code(code: str, code_hash: str) -> bool:
    return hmac.compare_digest(hash_otp_code(code), code_hash)
