"""
User authentication API.

POST /api/auth/signup            — register (frontend runs reCAPTCHA itself)
POST /api/auth/verify-otp        — confirm signup or password-reset OTP
POST /api/auth/resend-otp        — request a fresh OTP
POST /api/auth/login             — username/password -> access token + refresh cookie
POST /api/auth/refresh           — rotate refresh cookie -> new access token
POST /api/auth/logout            — revoke refresh token, clear cookie
POST /api/auth/forgot-password   — request a password-reset OTP
POST /api/auth/reset-password    — OTP + new password -> revokes all sessions
GET  /api/auth/me                — current account, from a verified access token
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.core.auth import CurrentUser, get_current_user
from app.core.settings import settings
from app.db.repositories import get_user_repository
from app.schemas.auth import (
    AuthTokenResponse,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    ResendOtpRequest,
    ResetPasswordRequest,
    SignupRequest,
    SignupResponse,
    UserPublic,
    VerifyOtpRequest,
)
from app.services import auth_service, model_gate

router = APIRouter(prefix="/auth", tags=["auth"])

_VALID_PURPOSES = {"signup", "password_reset"}


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        max_age=settings.refresh_token_ttl_seconds,
        path="/api/auth",
        domain=settings.refresh_cookie_domain or None,
        secure=settings.refresh_cookie_secure,
        httponly=True,
        samesite="lax",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path="/api/auth",
        domain=settings.refresh_cookie_domain or None,
    )


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _require_valid_purpose(purpose: str) -> None:
    if purpose not in _VALID_PURPOSES:
        raise HTTPException(status_code=400, detail="purpose must be 'signup' or 'password_reset'")


@router.post("/signup", response_model=SignupResponse, status_code=201)
def signup(payload: SignupRequest):
    auth_service.signup(
        username=payload.username,
        password=payload.password,
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        persona=payload.persona,
    )
    return SignupResponse(message="Check your email for a verification code", email=payload.email)


@router.post("/verify-otp")
def verify_otp(payload: VerifyOtpRequest, request: Request, response: Response):
    _require_valid_purpose(payload.purpose)
    result = auth_service.verify_otp(email=payload.email, code=payload.code, purpose=payload.purpose)

    if payload.purpose == "signup":
        user, access_token, refresh_token, session_id = auth_service.issue_token_pair(
            result["id"],
            user_agent=request.headers.get("user-agent"),
            ip=_client_ip(request),
        )
        _set_refresh_cookie(response, refresh_token)
        model_gate.trigger_warmup(session_id)
        return AuthTokenResponse(
            access_token=access_token, user=UserPublic(**user), session_id=session_id
        )

    return MessageResponse(message="Code verified")


@router.post("/resend-otp", response_model=MessageResponse)
def resend_otp(payload: ResendOtpRequest):
    _require_valid_purpose(payload.purpose)
    auth_service.resend_otp(email=payload.email, purpose=payload.purpose)
    return MessageResponse(message="If an account exists, a new code has been sent")


@router.post("/login", response_model=AuthTokenResponse)
def login(payload: LoginRequest, request: Request, response: Response):
    user, access_token, refresh_token, session_id = auth_service.login(
        username=payload.username,
        password=payload.password,
        client_key=f"{payload.username}:{_client_ip(request)}",
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
    )
    _set_refresh_cookie(response, refresh_token)
    model_gate.trigger_warmup(session_id)
    return AuthTokenResponse(
        access_token=access_token, user=UserPublic(**user), session_id=session_id
    )


@router.post("/refresh", response_model=AuthTokenResponse)
def refresh(request: Request, response: Response):
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No session found. Please log in.")

    # No model_gate.trigger_warmup here by design — refresh fires on every
    # silent token renewal (page load, tab focus), not just a real return
    # visit. See the note in auth_service.refresh_access_token.
    user, access_token, new_refresh_token, session_id = auth_service.refresh_access_token(
        refresh_token=refresh_token,
        user_agent=request.headers.get("user-agent"),
        ip=_client_ip(request),
    )
    _set_refresh_cookie(response, new_refresh_token)
    return AuthTokenResponse(
        access_token=access_token, user=UserPublic(**user), session_id=session_id
    )


@router.post("/logout", response_model=MessageResponse)
def logout(request: Request, response: Response):
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if refresh_token:
        auth_service.logout(refresh_token=refresh_token)
    _clear_refresh_cookie(response)
    return MessageResponse(message="Logged out")


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest):
    auth_service.forgot_password(email=payload.email)
    return MessageResponse(message="If an account exists, a reset code has been sent")


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest):
    auth_service.reset_password(email=payload.email, code=payload.code, new_password=payload.new_password)
    return MessageResponse(message="Password updated. Please log in.")


@router.get("/me", response_model=UserPublic)
def me(current_user: CurrentUser = Depends(get_current_user)):
    users = get_user_repository()
    user = users.get_by_id(current_user.id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return UserPublic(**auth_service.to_public_user(user))
