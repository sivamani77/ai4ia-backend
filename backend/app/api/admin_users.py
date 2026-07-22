"""
Admin user management API. Every route here requires persona=admin
(enforced by require_role("admin") on this router's include in main.py).

GET    /api/admin/users                — list/search/filter all accounts
GET    /api/admin/users/{user_id}       — one account's full detail
GET    /api/admin/users/{user_id}/generations — everything this user has generated
POST   /api/admin/users                 — create an account directly (no OTP)
PATCH  /api/admin/users/{user_id}       — edit name/email/persona
POST   /api/admin/users/{user_id}/reset-password  — force-set a password
POST   /api/admin/users/{user_id}/unlock          — clear a lockout
POST   /api/admin/users/{user_id}/deactivate      — soft-delete (reversible)
POST   /api/admin/users/{user_id}/reactivate      — undo deactivate
DELETE /api/admin/users/{user_id}                 — hard delete, full cascade
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.schemas.admin_users import (
    AdminCreateUserRequest,
    AdminDeleteUserResponse,
    AdminResetPasswordRequest,
    AdminUpdateUserRequest,
    AdminUserGenerationsResponse,
    AdminUserListResponse,
    AdminUserRecord,
)
from app.services import admin_user_service

router = APIRouter(prefix="/admin/users", tags=["admin-users"])


@router.get("", response_model=AdminUserListResponse)
def list_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="pending_verification | active | locked | deactivated"),
    persona: str | None = Query(None, description="general | expert | admin"),
    search: str | None = Query(None, description="Matches username, email, or name"),
):
    users, total = admin_user_service.list_users(
        limit=limit, offset=offset, status=status, persona=persona, search=search
    )
    return AdminUserListResponse(
        users=[AdminUserRecord(**u) for u in users], total=total, limit=limit, offset=offset
    )


@router.get("/{user_id}", response_model=AdminUserRecord)
def get_user(user_id: str):
    return AdminUserRecord(**admin_user_service.get_user(user_id))


@router.get("/{user_id}/generations", response_model=AdminUserGenerationsResponse)
def get_user_generations(user_id: str):
    """Every video-generation job this user has created, any status
    (queued/running/failed/completed) — prompt in, rendered media out.
    For admin monitoring/debugging of what a specific user is generating."""
    from app.services import job_store
    from app.services.admin_content_service import job_to_generation_entry

    user = AdminUserRecord(**admin_user_service.get_user(user_id))

    jobs = job_store.list_by_user(user_id)
    generations = [job_to_generation_entry(job) for job in jobs]

    return AdminUserGenerationsResponse(user=user, generations=generations, total=len(generations))


@router.post("", response_model=AdminUserRecord, status_code=201)
def create_user(payload: AdminCreateUserRequest):
    created = admin_user_service.create_user(
        username=payload.username,
        password=payload.password,
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        persona=payload.persona,
    )
    return AdminUserRecord(**created)


@router.patch("/{user_id}", response_model=AdminUserRecord)
def update_user(user_id: str, payload: AdminUpdateUserRequest):
    updated = admin_user_service.update_user(
        user_id,
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        persona=payload.persona,
    )
    return AdminUserRecord(**updated)


@router.post("/{user_id}/reset-password")
def admin_reset_password(user_id: str, payload: AdminResetPasswordRequest):
    admin_user_service.admin_reset_password(user_id, payload.new_password)
    return {"message": "Password reset. All of this user's sessions have been signed out."}


@router.post("/{user_id}/unlock", response_model=AdminUserRecord)
def unlock_user(user_id: str):
    return AdminUserRecord(**admin_user_service.unlock_user(user_id))


@router.post("/{user_id}/deactivate", response_model=AdminUserRecord)
def deactivate_user(user_id: str):
    return AdminUserRecord(**admin_user_service.deactivate_user(user_id))


@router.post("/{user_id}/reactivate", response_model=AdminUserRecord)
def reactivate_user(user_id: str):
    return AdminUserRecord(**admin_user_service.reactivate_user(user_id))


@router.delete("/{user_id}", response_model=AdminDeleteUserResponse)
def delete_user(user_id: str, hard: bool = Query(False, description="true = permanent full-cascade delete")):
    if not hard:
        result = admin_user_service.deactivate_user(user_id)
        return AdminDeleteUserResponse(status="deactivated", user_id=user_id, mode="soft")

    cascade = admin_user_service.hard_delete_user(user_id)
    return AdminDeleteUserResponse(
        status="deleted",
        user_id=user_id,
        mode="hard",
        deleted_jobs=cascade["deleted_jobs"],
        deleted_storage_objects=cascade["deleted_storage_objects"],
    )
