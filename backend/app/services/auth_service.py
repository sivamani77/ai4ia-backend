from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

import uuid

from app.core import metrics as _metrics
from app.core.rate_limit import TokenBucket
from app.core.security import (
    create_access_token,
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
from app.core.settings import settings
from app.db.repositories import (
    get_auth_audit_repository,
    get_otp_repository,
    get_refresh_token_repository,
    get_user_repository,
)
from app.db.repositories.user_repository import DuplicateUserError
from app.services import session_service
from app.services.mail_service import (
    MailDeliveryError,
    send_email_changed_notice,
    send_otp_email,
    send_password_changed_notice,
)

logger = logging.getLogger("ai4ia.services.auth")

# persona (UI) <-> DB `persona` column mapping — matches the convention the
# frontend already established (data.persona === "user" -> "general", etc.)
_PERSONA_TO_DB = {"general": "user", "expert": "researcher"}
_DB_TO_PERSONA = {"user": "general", "researcher": "expert", "admin": "admin"}


_otp_resend_limiter = TokenBucket(rate=1, per=60.0)
_login_limiter = TokenBucket(rate=5, per=60.0)


def _to_ui_persona(db_persona: str | None) -> str:
    return _DB_TO_PERSONA.get(db_persona or "", db_persona or "")


def to_public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "email": user.get("email_id"),
        "persona": _to_ui_persona(user.get("persona")),
    }


# ---------------------------------------------------------------------------
# Signup + OTP
# ---------------------------------------------------------------------------

def signup(*, username: str, password: str, first_name: str, last_name: str, email: str, persona: str) -> str:
    users = get_user_repository()
    audit = get_auth_audit_repository()

    if users.username_exists(username):
        _metrics.auth_events_total.labels(event="signup", outcome="failure").inc()
        raise HTTPException(status_code=409, detail="Username already taken")
    if users.email_exists(email):
        _metrics.auth_events_total.labels(event="signup", outcome="failure").inc()
        raise HTTPException(status_code=409, detail="Email already registered")

    password_hash = hash_password(password)
    try:
        created = users.create_user(
            {
                "username": username,
                "password_hash": password_hash,
                # Legacy plaintext column is NOT NULL in the existing schema but
                # unused for new accounts — password_hash is authoritative.
                "password": "",
                "first_name": first_name,
                "last_name": last_name,
                "email_id": email,
                "persona": _PERSONA_TO_DB[persona],
                "status": "pending_verification",
                "email_verified": False,
            }
        )
    except DuplicateUserError as exc:
        _metrics.auth_events_total.labels(event="signup", outcome="failure").inc()
        detail = "Username already taken" if exc.field == "username" else "Email already registered"
        raise HTTPException(status_code=409, detail=detail)
    audit.log(event="signup", success=True, user_id=created["id"])
    _metrics.auth_events_total.labels(event="signup", outcome="success").inc()

    _issue_otp(email=email, purpose="signup", user_id=created["id"])
    return created["id"]


def _issue_otp(*, email: str, purpose: str, user_id: str | None) -> None:
    otp_repo = get_otp_repository()
    otp_repo.invalidate_active(email, purpose)

    code = generate_otp_code()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=settings.otp_ttl_minutes)).isoformat()
    otp_repo.create(
        {
            "email": email,
            "user_id": user_id,
            "purpose": purpose,
            "code_hash": hash_otp_code(code),
            "expires_at": expires_at,
        }
    )

    try:
        send_otp_email(email, code, purpose)
    except MailDeliveryError:
        # Surface as a 502 rather than silently pretending the OTP was sent —
        # a stuck user with no way to get a code is worse than a loud error.
        raise HTTPException(status_code=502, detail="Failed to send verification email. Please try again.")


def resend_otp(*, email: str, purpose: str) -> None:
    if not _otp_resend_limiter.allow(f"{email}:{purpose}"):
        _metrics.rate_limit_rejections_total.labels(limiter="otp_resend").inc()
        _metrics.auth_events_total.labels(event="otp_resend", outcome="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")

    users = get_user_repository()
    user = users.get_by_email(email)
    user_id = user["id"] if user else None

    if purpose == "signup" and user is not None and user.get("status") == "active":
        # Already verified — nothing to resend, but don't reveal account
        # state to an unauthenticated caller beyond a generic message.
        _metrics.auth_events_total.labels(event="otp_resend", outcome="noop").inc()
        return
    if purpose == "password_reset" and user is None:
        # No account enumeration — behave as if it was sent.
        _metrics.auth_events_total.labels(event="otp_resend", outcome="noop").inc()
        return

    _issue_otp(email=email, purpose=purpose, user_id=user_id)
    _metrics.auth_events_total.labels(event="otp_resend", outcome="success").inc()


def issue_token_pair(
    user_id: str, *, user_agent: str | None = None, ip: str | None = None
) -> tuple[dict, str, str, str]:
    """Issue a fresh access + refresh token pair for an already-authenticated
    user_id — used right after signup verification, where no password check
    is needed again."""
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")
    persona = _to_ui_persona(user.get("persona"))
    access_token = create_access_token(user_id=user_id, persona=persona)
    refresh_token, session_id = _issue_refresh_token(user_id=user_id, user_agent=user_agent, ip=ip)
    session_service.create_session(
        session_id=session_id, user_id=user_id, user_agent=user_agent, ip=ip
    )
    return to_public_user(user), access_token, refresh_token, session_id


def verify_otp(*, email: str, code: str, purpose: str) -> dict:
    otp_repo = get_otp_repository()
    users = get_user_repository()
    audit = get_auth_audit_repository()

    record = otp_repo.get_latest_active(email, purpose)
    if record is None:
        _metrics.auth_events_total.labels(event="otp_verify", outcome="failure").inc()
        raise HTTPException(status_code=400, detail="No active verification code for this email")

    expires_at = parse_timestamp(record["expires_at"])
    if datetime.now(timezone.utc) >= expires_at:
        _metrics.auth_events_total.labels(event="otp_verify", outcome="failure").inc()
        raise HTTPException(status_code=400, detail="Verification code expired. Please request a new one.")

    if record["attempts"] >= settings.otp_max_attempts:
        _metrics.auth_events_total.labels(event="otp_verify", outcome="failure").inc()
        raise HTTPException(status_code=429, detail="Too many attempts. Please request a new code.")

    if not verify_otp_code(code, record["code_hash"]):
        otp_repo.increment_attempts(record["id"], record["attempts"] + 1)
        audit.log(event="otp_failed", success=False, user_id=record.get("user_id"))
        _metrics.auth_events_total.labels(event="otp_verify", outcome="failure").inc()
        raise HTTPException(status_code=400, detail="Incorrect verification code")

    otp_repo.mark_used(record["id"])
    audit.log(event="otp_verified", success=True, user_id=record.get("user_id"))
    _metrics.auth_events_total.labels(event="otp_verify", outcome="success").inc()

    if purpose == "signup":
        user_id = record.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=400, detail="No pending signup for this email")
        users.mark_verified(user_id)
        user = users.get_by_id(user_id)
        return to_public_user(user)

    # purpose == "password_reset": caller still needs to supply the new
    # password in a follow-up call — this just proves the code was valid.
    return {"email": email}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _issue_refresh_token(
    *, user_id: str, family_id: str | None = None, user_agent: str | None = None, ip: str | None = None
) -> tuple[str, str]:
    """Returns (refresh_token, family_id). `family_id` is reused as the
    session id — see app/services/session_service.py — one identifier per
    login that survives refresh rotation."""
    token = generate_refresh_token()
    resolved_family_id = family_id or str(uuid.uuid4())
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=settings.refresh_token_ttl_seconds)
    ).isoformat()
    get_refresh_token_repository().create(
        {
            "user_id": user_id,
            "token_hash": hash_refresh_token(token),
            "family_id": resolved_family_id,
            "expires_at": expires_at,
            "user_agent": user_agent,
            "ip": ip,
        }
    )
    return token, resolved_family_id


def login(
    *, username: str, password: str, client_key: str, user_agent: str | None = None, ip: str | None = None
) -> tuple[dict, str, str, str]:
    if not _login_limiter.allow(client_key):
        _metrics.rate_limit_rejections_total.labels(limiter="login").inc()
        _metrics.auth_events_total.labels(event="login", outcome="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Too many login attempts. Please wait and try again.")

    users = get_user_repository()
    audit = get_auth_audit_repository()

    user = users.get_by_username(username)
    if user is None and "@" in username:
        user = users.get_by_email(username)
    if user is None:
        audit.log(event="login", success=False)
        _metrics.auth_events_total.labels(event="login", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.get("locked_until"):
        locked_until = parse_timestamp(user["locked_until"])
        if datetime.now(timezone.utc) < locked_until:
            audit.log(event="login", success=False, user_id=user["id"])
            _metrics.auth_events_total.labels(event="login", outcome="failure").inc()
            raise HTTPException(status_code=423, detail="Account temporarily locked. Please try again later.")

    password_hash = user.get("password_hash")
    if password_hash:
        ok = verify_password(password, password_hash)
    else:
        legacy_plaintext = user.get("password")
        ok = bool(legacy_plaintext) and verify_legacy_plaintext_password(password, legacy_plaintext)
        if ok:
            users.upgrade_legacy_password(user["id"], hash_password(password))

    if not ok:
        _record_failed_attempt(user)
        audit.log(event="login", success=False, user_id=user["id"])
        _metrics.auth_events_total.labels(event="login", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if user.get("status") == "pending_verification":
        _metrics.auth_events_total.labels(event="login", outcome="failure").inc()
        raise HTTPException(
            status_code=403,
            detail={"code": "email_not_verified", "message": "Please verify your email before logging in."},
        )

    if user.get("status") == "deactivated":
        audit.log(event="login", success=False, user_id=user["id"])
        _metrics.auth_events_total.labels(event="login", outcome="failure").inc()
        raise HTTPException(
            status_code=403,
            detail={"code": "account_deactivated", "message": "This account has been deactivated."},
        )

    users.reset_login_attempts(user["id"])
    audit.log(event="login", success=True, user_id=user["id"])
    _metrics.auth_events_total.labels(event="login", outcome="success").inc()

    persona = _to_ui_persona(user.get("persona"))
    access_token = create_access_token(user_id=user["id"], persona=persona)
    refresh_token, session_id = _issue_refresh_token(user_id=user["id"], user_agent=user_agent, ip=ip)
    session_service.create_session(
        session_id=session_id, user_id=user["id"], user_agent=user_agent, ip=ip
    )
    return to_public_user(user), access_token, refresh_token, session_id


def _record_failed_attempt(user: dict) -> None:
    users = get_user_repository()
    failed_count = user.get("failed_login_count", 0) + 1
    locked_until = None
    if failed_count >= settings.login_max_failed_attempts:
        locked_until = (
            datetime.now(timezone.utc) + timedelta(seconds=settings.login_lockout_seconds)
        ).isoformat()
    users.record_failed_login(user["id"], failed_count=failed_count, locked_until=locked_until)


# ---------------------------------------------------------------------------
# Forgot / reset password
# ---------------------------------------------------------------------------

def forgot_password(*, email: str) -> None:
    users = get_user_repository()
    user = users.get_by_email(email)
    if user is None:
        _metrics.auth_events_total.labels(event="forgot_password", outcome="noop").inc()
        return  # no account enumeration
    _issue_otp(email=email, purpose="password_reset", user_id=user["id"])
    _metrics.auth_events_total.labels(event="forgot_password", outcome="success").inc()


def reset_password(*, email: str, code: str, new_password: str) -> None:
    from app.db.repositories import get_refresh_token_repository

    verify_otp(email=email, code=code, purpose="password_reset")

    users = get_user_repository()
    user = users.get_by_email(email)
    if user is None:
        _metrics.auth_events_total.labels(event="password_reset", outcome="failure").inc()
        raise HTTPException(status_code=400, detail="No account for this email")

    users.update_user(user["id"], {"password_hash": hash_password(new_password), "password": ""})
    users.reset_login_attempts(user["id"])

    get_refresh_token_repository().revoke_all_for_user(user["id"])
    session_service.end_all_sessions_for_user(user["id"], reason="password_reset")
    get_auth_audit_repository().log(event="password_reset", success=True, user_id=user["id"])
    _metrics.auth_events_total.labels(event="password_reset", outcome="success").inc()

    try:
        send_password_changed_notice(email)
    except MailDeliveryError:
        logger.warning("Password changed notice failed to send to %s", email)


# ---------------------------------------------------------------------------
# Self-service profile — name / password / email
# ---------------------------------------------------------------------------

def update_profile(user_id: str, *, first_name: str | None, last_name: str | None) -> dict:
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    updates: dict = {}
    if first_name is not None:
        updates["first_name"] = first_name
    if last_name is not None:
        updates["last_name"] = last_name

    if updates:
        users.update_user(user_id, updates)

    get_auth_audit_repository().log(event="profile_update", success=True, user_id=user_id)
    return to_public_user(users.get_by_id(user_id))


def request_password_change_otp(user_id: str) -> None:
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    email = user.get("email_id")
    if not email:
        raise HTTPException(status_code=400, detail="Account has no email on file")

    if not _otp_resend_limiter.allow(f"{email}:password_change"):
        _metrics.rate_limit_rejections_total.labels(limiter="otp_resend").inc()
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")

    _issue_otp(email=email, purpose="password_change", user_id=user_id)


def change_password(user_id: str, *, current_password: str, code: str, new_password: str) -> None:
    from app.db.repositories import get_refresh_token_repository

    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    email = user.get("email_id")
    if not email:
        raise HTTPException(status_code=400, detail="Account has no email on file")

    if not verify_password(current_password, user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    verify_otp(email=email, code=code, purpose="password_change")

    users.update_user(user_id, {"password_hash": hash_password(new_password), "password": ""})
    users.reset_login_attempts(user_id)

    get_refresh_token_repository().revoke_all_for_user(user_id)
    session_service.end_all_sessions_for_user(user_id, reason="password_change")
    get_auth_audit_repository().log(event="password_change", success=True, user_id=user_id)

    try:
        send_password_changed_notice(email)
    except MailDeliveryError:
        logger.warning("Password changed notice failed to send to %s", email)


def request_email_change_otp(user_id: str, *, new_email: str) -> None:
    """Issues TWO codes: one to the current email (proves the request is
    really coming from the account holder, not just a hijacked session) and
    one to the new email (proves they control it). Both are required to
    confirm in change_email()."""
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    current_email = user.get("email_id")
    if not current_email:
        raise HTTPException(status_code=400, detail="Account has no email on file")
    if new_email == current_email:
        raise HTTPException(status_code=400, detail="This is already your email")
    if users.email_exists(new_email):
        raise HTTPException(status_code=409, detail="Email already registered")

    if not _otp_resend_limiter.allow(f"{current_email}:email_change_old"):
        _metrics.rate_limit_rejections_total.labels(limiter="otp_resend").inc()
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")
    _issue_otp(email=current_email, purpose="email_change_old", user_id=user_id)

    if not _otp_resend_limiter.allow(f"{new_email}:email_change"):
        _metrics.rate_limit_rejections_total.labels(limiter="otp_resend").inc()
        raise HTTPException(status_code=429, detail="Please wait before requesting another code")
    _issue_otp(email=new_email, purpose="email_change", user_id=user_id)


def change_email(user_id: str, *, new_email: str, current_email_code: str, new_email_code: str) -> dict:
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")

    old_email = user.get("email_id")
    if not old_email:
        raise HTTPException(status_code=400, detail="Account has no email on file")

    # Re-check at confirm time too — closes the same request-then-confirm
    # race window that create_user's DuplicateUserError backstops for signup.
    if users.email_exists(new_email):
        raise HTTPException(status_code=409, detail="Email already registered")

    verify_otp(email=old_email, code=current_email_code, purpose="email_change_old")
    verify_otp(email=new_email, code=new_email_code, purpose="email_change")

    try:
        users.update_user(user_id, {"email_id": new_email})
    except DuplicateUserError:
        raise HTTPException(status_code=409, detail="Email already registered")

    get_auth_audit_repository().log(event="email_change", success=True, user_id=user_id)

    try:
        send_email_changed_notice(old_email, new_email)
    except MailDeliveryError:
        logger.warning("Email changed notice failed to send to %s", old_email)

    return to_public_user(users.get_by_id(user_id))


# ---------------------------------------------------------------------------
# Refresh / logout
# ---------------------------------------------------------------------------

def refresh_access_token(
    *, refresh_token: str, user_agent: str | None = None, ip: str | None = None
) -> tuple[dict, str, str, str]:
    tokens = get_refresh_token_repository()
    users = get_user_repository()
    audit = get_auth_audit_repository()

    token_hash = hash_refresh_token(refresh_token)
    record = tokens.get_by_hash(token_hash)
    if record is None:
        _metrics.auth_events_total.labels(event="refresh", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Invalid session. Please log in again.")

    if record.get("revoked_at"):
        _metrics.auth_events_total.labels(event="refresh", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Session revoked. Please log in again.")

    expires_at = parse_timestamp(record["expires_at"])
    if datetime.now(timezone.utc) >= expires_at:
        _metrics.auth_events_total.labels(event="refresh", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")

    if record.get("used_at"):
        # This token was already rotated once — someone is replaying an old
        # token, meaning the whole family may be compromised. Kill it all.
        tokens.revoke_family(record["family_id"])
        session_service.end_session(record["family_id"], reason="reuse_detected")
        audit.log(event="refresh_reuse_detected", success=False, user_id=record["user_id"])
        _metrics.auth_events_total.labels(event="refresh", outcome="failure").inc()
        _metrics.auth_refresh_reuse_detected_total.inc()
        raise HTTPException(status_code=401, detail="Session invalid. Please log in again.")

    tokens.mark_used(record["id"])

    user = users.get_by_id(record["user_id"])
    if user is None:
        _metrics.auth_events_total.labels(event="refresh", outcome="failure").inc()
        raise HTTPException(status_code=401, detail="Account no longer exists")

    persona = _to_ui_persona(user.get("persona"))
    access_token = create_access_token(user_id=user["id"], persona=persona)
    new_refresh_token, session_id = _issue_refresh_token(
        user_id=user["id"], family_id=record["family_id"], user_agent=user_agent, ip=ip
    )
    # No model warm-up here by design — refresh happens on every silent
    # token renewal (page loads, tab focus), not just on a real return visit,
    # and firing a warm-up on every one of those would defeat the point of
    # scale-to-zero. Only touch last_seen_at; app/services/model_gate.py
    # decides readiness by probing the hosts directly when a feature is used.
    session_service.touch_session(session_id)
    audit.log(event="refresh", success=True, user_id=user["id"])
    _metrics.auth_events_total.labels(event="refresh", outcome="success").inc()
    return to_public_user(user), access_token, new_refresh_token, session_id


def logout(*, refresh_token: str) -> None:
    tokens = get_refresh_token_repository()
    record = tokens.get_by_hash(hash_refresh_token(refresh_token))
    if record is None:
        _metrics.auth_events_total.labels(event="logout", outcome="noop").inc()
        return
    tokens.revoke(record["id"])
    session_service.end_session(record["family_id"], reason="logout")
    get_auth_audit_repository().log(event="logout", success=True, user_id=record["user_id"])
    _metrics.auth_events_total.labels(event="logout", outcome="success").inc()
