from __future__ import annotations

import logging

from fastapi import HTTPException

from app.core.security import hash_password
from app.db.repositories import (
    get_auth_audit_repository,
    get_event_repository,
    get_job_repository,
    get_refresh_token_repository,
    get_session_repository,
    get_user_repository,
)
from app.db.repositories.user_repository import DuplicateUserError
from app.services.auth_service import _PERSONA_TO_DB, _to_ui_persona, to_public_user

logger = logging.getLogger("ai4ia.services.admin_users")

# UI persona -> DB persona column value. Admin CAN self-assign "admin" here
# (unlike self-service signup) since only an admin can reach these endpoints.
_ADMIN_PERSONA_TO_DB = {**_PERSONA_TO_DB, "admin": "admin"}


def to_admin_record(user: dict) -> dict:
    return {
        "id": user["id"],
        "username": user["username"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "email": user.get("email_id"),
        "persona": _to_ui_persona(user.get("persona")),
        "status": user.get("status"),
        "email_verified": bool(user.get("email_verified")),
        "failed_login_count": user.get("failed_login_count", 0),
        "locked_until": user.get("locked_until"),
        "created_at": user.get("created_at"),
        "updated_at": user.get("updated_at"),
    }


def list_users(
    *,
    limit: int,
    offset: int,
    status: str | None,
    persona: str | None,
    search: str | None,
) -> tuple[list[dict], int]:
    users = get_user_repository()
    db_persona = _ADMIN_PERSONA_TO_DB.get(persona) if persona else None
    rows, total = users.list_all(
        limit=limit, offset=offset, status=status, persona=db_persona, search=search
    )
    return [to_admin_record(r) for r in rows], total


def get_user(user_id: str) -> dict:
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return to_admin_record(user)


def create_user(
    *, username: str, password: str, first_name: str, last_name: str, email: str, persona: str
) -> dict:
    users = get_user_repository()
    if users.username_exists(username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if users.email_exists(email):
        raise HTTPException(status_code=409, detail="Email already registered")

    try:
        created = users.create_user(
            {
                "username": username,
                "password_hash": hash_password(password),
                "password": "",
                "first_name": first_name,
                "last_name": last_name,
                "email_id": email,
                "persona": _ADMIN_PERSONA_TO_DB[persona],
                # Admin-created accounts are active immediately — no OTP dance;
                # the admin is vouching for the email out-of-band.
                "status": "active",
                "email_verified": True,
            }
        )
    except DuplicateUserError as exc:
        detail = "Username already taken" if exc.field == "username" else "Email already registered"
        raise HTTPException(status_code=409, detail=detail)
    get_auth_audit_repository().log(event="admin_create_user", success=True, user_id=created["id"])
    return to_admin_record(created)


def update_user(
    user_id: str,
    *,
    first_name: str | None,
    last_name: str | None,
    email: str | None,
    persona: str | None,
) -> dict:
    users = get_user_repository()
    existing = users.get_by_id(user_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="User not found")

    updates: dict = {}
    if first_name is not None:
        updates["first_name"] = first_name
    if last_name is not None:
        updates["last_name"] = last_name
    if email is not None and email != existing.get("email_id"):
        if users.email_exists(email):
            raise HTTPException(status_code=409, detail="Email already registered")
        updates["email_id"] = email
    if persona is not None:
        updates["persona"] = _ADMIN_PERSONA_TO_DB[persona]

    if updates:
        users.update_user(user_id, updates)

    get_auth_audit_repository().log(event="admin_update_user", success=True, user_id=user_id)
    return to_admin_record(users.get_by_id(user_id))


def admin_reset_password(user_id: str, new_password: str) -> None:
    users = get_user_repository()
    if users.get_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    users.update_user(user_id, {"password_hash": hash_password(new_password), "password": ""})
    users.reset_login_attempts(user_id)
    get_refresh_token_repository().revoke_all_for_user(user_id)
    get_auth_audit_repository().log(event="admin_reset_password", success=True, user_id=user_id)


def unlock_user(user_id: str) -> dict:
    users = get_user_repository()
    if users.get_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    users.unlock(user_id)
    get_auth_audit_repository().log(event="admin_unlock_user", success=True, user_id=user_id)
    return to_admin_record(users.get_by_id(user_id))


def deactivate_user(user_id: str) -> dict:
    """Soft-delete: blocks login, keeps every row (account + job history)
    intact and reversible via reactivate_user."""
    users = get_user_repository()
    if users.get_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    users.deactivate(user_id)
    get_refresh_token_repository().revoke_all_for_user(user_id)
    get_auth_audit_repository().log(event="admin_deactivate_user", success=True, user_id=user_id)
    return to_admin_record(users.get_by_id(user_id))


def reactivate_user(user_id: str) -> dict:
    users = get_user_repository()
    if users.get_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    users.reactivate(user_id)
    get_auth_audit_repository().log(event="admin_reactivate_user", success=True, user_id=user_id)
    return to_admin_record(users.get_by_id(user_id))


def hard_delete_user(user_id: str) -> dict:
    """Full cascade delete — 'right to be forgotten'. Removes the account
    AND everything attributable to it. Both groups now live in the same
    Postgres database (DATABASE_URL), but they are still deleted in two
    steps because only the second group is covered by FK cascades:

      pipeline tables: the user's jobs (+ S3/local export artifacts),
                       job_events, llm_call_traces, agent_decision_records,
                       prompt_resolutions, llm_call_payloads
      auth tables:     user_sessions (deleted explicitly — that table has no
                       FK to app_users, so nothing cascades to it and its
                       stored ip/user_agent would outlive the account), then
                       the app_users row (cascades refresh_tokens/otp_codes
                       and nulls auth_audit.user_id via FK ON DELETE rules)

    Insights are deleted by owner. Insights made before owners were
    recorded, and never backfilled, have no owner and are not covered.

    Order matters: delete children before the app_users row, since once the
    account is gone there is no way to look up "this user's jobs" again.
    """
    users = get_user_repository()
    user = users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    jobs_repo = get_job_repository()
    events_repo = get_event_repository()

    from app.db.repositories.llm_trace_repository import LLMTraceRepository
    from app.db.repositories.decision_record_repository import DecisionRecordRepository
    from app.db.repositories.insight_repository import InsightRepository
    from app.services import processing_records

    llm_traces = LLMTraceRepository()
    decision_records = DecisionRecordRepository()

    user_jobs = jobs_repo.list_by_user(user_id)
    deleted_storage_objects = 0

    for job in user_jobs:
        job_id = job.get("id") or job.get("job_id")
        if not job_id:
            continue
        deleted_storage_objects += _delete_job_storage(job_id)
        llm_traces.delete_by_job(job_id)
        processing_records.delete_for_job(job_id)
        events_repo.delete_by_job(job_id)
        jobs_repo.delete_job(job_id)

    llm_traces.delete_by_user(user_id)
    decision_records.delete_by_user(user_id)
    InsightRepository().delete_by_user(user_id)
    processing_records.delete_for_user(user_id)

    # Must precede delete_user: user_sessions has no FK to app_users, so this
    # is the only thing that removes the session rows' ip/user_agent.
    deleted_sessions = get_session_repository().delete_all_for_user(user_id)

    users.delete_user(user_id)  # cascades refresh_tokens/otp_codes/auth_audit (auth project FK)

    logger.info(
        "Hard-deleted user %s: %d jobs, %d storage objects, %d sessions",
        user_id, len(user_jobs), deleted_storage_objects, deleted_sessions,
    )

    return {
        "deleted_jobs": len(user_jobs),
        "deleted_storage_objects": deleted_storage_objects,
        "deleted_sessions": deleted_sessions,
    }


def _delete_job_storage(job_id: str) -> int:
    """Best-effort cleanup of a job's export/asset files. S3 or local,
    depending on the active storage backend. Failures are logged, not
    raised — a missing file shouldn't block the rest of the account
    deletion."""
    from app.core.settings import settings

    if settings.ai4ia_storage_backend == "s3":
        try:
            from app.api.admin import _list_keys, _delete_keys

            keys = [k for k in _list_keys("exports/") if job_id in k]
            keys += [k for k in _list_keys("generated_assets/") if job_id in k]
            return _delete_keys(keys)
        except Exception:
            logger.exception("Failed to delete S3 storage for job %s", job_id)
            return 0

    try:
        import shutil
        from pathlib import Path
        from app.core.settings import settings as _settings

        removed = 0
        for base_dir in (_settings.ai4ia_local_exports_dir, _settings.ai4ia_local_assets_dir):
            job_dir = Path(base_dir) / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
                removed += 1
        return removed
    except Exception:
        logger.exception("Failed to delete local storage for job %s", job_id)
        return 0
