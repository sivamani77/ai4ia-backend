from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import TokenExpiredError, TokenInvalidError, decode_access_token
from app.core.settings import settings

BEARER_CHALLENGE = 'Bearer realm="ai4ia"'
_EXPIRED_CHALLENGE = BEARER_CHALLENGE + ', error="invalid_token", error_description="The access token expired"'
_INVALID_CHALLENGE = BEARER_CHALLENGE + ', error="invalid_token"'

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(key: str = Security(_api_key_header)) -> None:
    if not settings.api_key:
        return
    if key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


@dataclass(frozen=True)
class CurrentUser:
    id: str
    persona: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> CurrentUser:
    """Required auth dependency — fails closed with 401 if the bearer token
    is missing, malformed, expired, or invalid. user_id always comes from a
    verified token, never from anything client-supplied."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": BEARER_CHALLENGE})

    try:
        claims = decode_access_token(credentials.credentials)
    except TokenExpiredError:
        raise HTTPException(
            status_code=401,
            detail={"code": "token_expired", "message": "Access token expired"},
            headers={"WWW-Authenticate": _EXPIRED_CHALLENGE},
        )
    except TokenInvalidError:
        raise HTTPException(
            status_code=401, detail="Invalid access token", headers={"WWW-Authenticate": _INVALID_CHALLENGE}
        )

    return CurrentUser(id=claims["sub"], persona=claims.get("persona", ""))


def require_role(*allowed_personas: str):
    def _dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.persona not in allowed_personas:
            raise HTTPException(status_code=403, detail="You do not have permission to perform this action")
        return user

    return _dependency
