from __future__ import annotations

import re

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.security import validate_password_strength

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]{3,32}$")


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    persona: str = Field(..., description="general | expert — admin cannot self-assign")

    @field_validator("username")
    @classmethod
    def _validate_username(cls, v: str) -> str:
        if not _USERNAME_RE.match(v):
            raise ValueError("Username may only contain letters, numbers, '.', '_', '-'")
        return v

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        return validate_password_strength(v)

    @field_validator("persona")
    @classmethod
    def _validate_persona(cls, v: str) -> str:
        if v not in ("general", "expert"):
            raise ValueError("persona must be 'general' or 'expert'")
        return v


class SignupResponse(BaseModel):
    message: str
    email: str


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    purpose: str = Field(..., description="signup | password_reset")


class ResendOtpRequest(BaseModel):
    email: EmailStr
    purpose: str = Field(..., description="signup | password_reset")


class LoginRequest(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    id: str
    username: str
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    persona: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, v: str) -> str:
        return validate_password_strength(v)


class MessageResponse(BaseModel):
    message: str
