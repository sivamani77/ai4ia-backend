from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import TokenExpiredError, TokenInvalidError, decode_access_token
from app.core.settings import settings

BEARER_CHALLENGE = 'Bearer realm="ai4ia"'
_EXPIRED_CHALLENGE = BEARER_CHALLENGE + ', error="invalid_token", error_description="The access token expired"'
_INVALID_CHALLENGE = BEARER_CHALLENGE + ', error="invalid_token"'

logger = logging.getLogger("ai4ia.core.auth")

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


@dataclass(frozen=True)
class UserState:
    """What a token is checked against: its version, the persona to act as,
    and whether the account may still sign in."""
    token_version: int
    persona: str
    active: bool


# Each replica caches user state briefly, so a revocation (version bump,
# demotion, deactivation) takes effect everywhere within this many seconds
# without a database read on every request.
_USER_STATE_TTL = 15.0
_USER_STATE_PRUNE_AT = 10_000  # entries; expired ones are dropped past this
_user_state_cache: dict[str, tuple[UserState | None, float]] = {}
# Bumped by forget_user_state, so a load that started before a revocation
# cannot store the stale state it read.
_user_state_generation: dict[str, int] = {}
_user_state_lock = threading.Lock()


def _load_user_state(user_id: str) -> UserState | None:
    from app.db.repositories import get_user_repository
    from app.services.auth_service import _to_ui_persona

    user = get_user_repository().get_by_id(user_id)
    if user is None:
        return None
    return UserState(
        token_version=int(user.get("token_version") or 0),
        persona=_to_ui_persona(user.get("persona")),
        active=user.get("status") != "deactivated",
    )


def user_state(user_id: str) -> UserState | None:
    now = time.monotonic()
    with _user_state_lock:
        cached = _user_state_cache.get(user_id)
        generation = _user_state_generation.get(user_id, 0)
    if cached is not None and now - cached[1] < _USER_STATE_TTL:
        return cached[0]
    state = _load_user_state(user_id)
    with _user_state_lock:
        if _user_state_generation.get(user_id, 0) == generation:
            _user_state_cache[user_id] = (state, now)
        if len(_user_state_cache) > _USER_STATE_PRUNE_AT:
            for key in [k for k, (_, at) in _user_state_cache.items() if now - at >= _USER_STATE_TTL]:
                _user_state_cache.pop(key, None)
    return state


def forget_user_state(user_id: str) -> None:
    """Drop the cached state so this replica sees a change immediately."""
    with _user_state_lock:
        _user_state_cache.pop(user_id, None)
        _user_state_generation[user_id] = _user_state_generation.get(user_id, 0) + 1


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

    stale = HTTPException(
        # Revoked by a password, persona or account change, or minted before
        # versions existed. Same code as expiry, so the client refreshes.
        status_code=401,
        detail={"code": "token_expired", "message": "Access token is no longer valid"},
        headers={"WWW-Authenticate": _EXPIRED_CHALLENGE},
    )
    version = claims.get("ver")
    if not isinstance(version, int):
        raise stale

    try:
        state = user_state(claims["sub"])
        if state is not None and version > state.token_version:
            # Minted after a change this replica has not seen yet; re-read
            # rather than reject a token that is newer than the cache.
            forget_user_state(claims["sub"])
            state = user_state(claims["sub"])
    except Exception:  # noqa: BLE001
        # A database outage must not look like a signed-out user.
        logger.exception("Could not load user state for token check")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again.")

    if state is None or not state.active:
        raise HTTPException(
            status_code=401, detail="Account is not active", headers={"WWW-Authenticate": _INVALID_CHALLENGE}
        )
    if version != state.token_version:
        raise stale

    # Persona comes from the account, not the token, so a demotion applies at once.
    return CurrentUser(id=claims["sub"], persona=state.persona)


def require_role(*allowed_personas: str):
    def _dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.persona not in allowed_personas:
            raise HTTPException(status_code=403, detail="You do not have permission to perform this action")
        return user

    return _dependency
