from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.security import normalize_identifier, validate_password_strength, validate_username_format


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
        return validate_username_format(normalize_identifier(v))

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return normalize_identifier(v)

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

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return normalize_identifier(v)


class ResendOtpRequest(BaseModel):
    email: EmailStr
    purpose: str = Field(..., description="signup | password_reset")

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return normalize_identifier(v)


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email")
    password: str

    @field_validator("username")
    @classmethod
    def _strip_username(cls, v: str) -> str:
        # Kept case-sensitive here (not normalize_identifier's lowercasing):
        # existing accounts predate the lowercase-only rule, and lookup is
        # case-insensitive at the query layer, so trimming is all this needs.
        return v.strip()


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
    # Refresh-token family_id, reused as the session id (see
    # app/services/session_service.py). Frontend stores this and sends it
    # back as X-Session-Id for endpoints that report model warm-up status.
    session_id: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return normalize_identifier(v)


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    new_password: str

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return normalize_identifier(v)

    @field_validator("new_password")
    @classmethod
    def _validate_new_password(cls, v: str) -> str:
        return validate_password_strength(v)


class MessageResponse(BaseModel):
    message: str
