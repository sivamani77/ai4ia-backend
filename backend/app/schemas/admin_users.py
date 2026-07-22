from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.security import validate_password_strength


class AdminUserRecord(BaseModel):
    id: str
    username: str
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    persona: str
    status: str
    email_verified: bool
    failed_login_count: int
    locked_until: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AdminUserListResponse(BaseModel):
    users: list[AdminUserRecord]
    total: int
    limit: int
    offset: int


class AdminCreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: str
    persona: str = Field(..., description="general | expert | admin")

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        return validate_password_strength(v)

    @field_validator("persona")
    @classmethod
    def _validate_persona(cls, v: str) -> str:
        if v not in ("general", "expert", "admin"):
            raise ValueError("persona must be 'general', 'expert', or 'admin'")
        return v


class AdminUpdateUserRequest(BaseModel):
    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    email: str | None = None
    persona: str | None = None

    @field_validator("persona")
    @classmethod
    def _validate_persona(cls, v: str | None) -> str | None:
        if v is not None and v not in ("general", "expert", "admin"):
            raise ValueError("persona must be 'general', 'expert', or 'admin'")
        return v


class AdminResetPasswordRequest(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        return validate_password_strength(v)


class AdminDeleteUserResponse(BaseModel):
    status: str
    user_id: str
    mode: str
    deleted_jobs: int = 0
    deleted_storage_objects: int = 0


class AdminGenerationEntry(BaseModel):
    """One video-generation job belonging to a user, any status — includes
    the prompt that produced it plus whatever media has rendered so far,
    for admin monitoring/debugging."""

    job_id: str | None = None
    user_id: str | None = None
    status: str | None = None
    prompt: str | None = None
    target_duration: int | None = None
    current_stage: str | None = None
    progress_percent: int | None = None
    message: str | None = None
    error: Any = None
    created_at: str | None = None
    completed_at: str | None = None
    video: dict[str, Any] | None = None
    srt: dict[str, Any] | None = None
    vtt: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    thumbnail: dict[str, Any] | None = None
    other_assets: list[dict[str, Any]] = Field(default_factory=list)


class AdminUserGenerationsResponse(BaseModel):
    user: AdminUserRecord
    generations: list[AdminGenerationEntry]
    total: int
