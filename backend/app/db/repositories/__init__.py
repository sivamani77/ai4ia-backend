from functools import lru_cache

from app.db.repositories.auth_audit_repository import AuthAuditRepository
from app.db.repositories.event_repository import EventRepository
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.otp_repository import OtpRepository
from app.db.repositories.refresh_token_repository import RefreshTokenRepository
from app.db.repositories.user_repository import UserRepository


@lru_cache(maxsize=1)
def get_job_repository() -> JobRepository:
    return JobRepository()


@lru_cache(maxsize=1)
def get_event_repository() -> EventRepository:
    return EventRepository()


@lru_cache(maxsize=1)
def get_user_repository() -> UserRepository:
    return UserRepository()


@lru_cache(maxsize=1)
def get_otp_repository() -> OtpRepository:
    return OtpRepository()


@lru_cache(maxsize=1)
def get_auth_audit_repository() -> AuthAuditRepository:
    return AuthAuditRepository()


@lru_cache(maxsize=1)
def get_refresh_token_repository() -> RefreshTokenRepository:
    return RefreshTokenRepository()


__all__ = [
    "AuthAuditRepository",
    "EventRepository",
    "JobRepository",
    "OtpRepository",
    "RefreshTokenRepository",
    "UserRepository",
    "get_auth_audit_repository",
    "get_event_repository",
    "get_job_repository",
    "get_otp_repository",
    "get_refresh_token_repository",
    "get_user_repository",
]
