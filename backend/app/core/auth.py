from __future__ import annotations

import re

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.core.settings import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


async def require_api_key(key: str = Security(_api_key_header)) -> None:
    if not settings.api_key:
        return
    if key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def extract_session_id(request: Request) -> str:
    session_id = request.headers.get("X-Session-ID", "")
    if not session_id:
        return ""
    if not _UUID4_RE.match(session_id):
        raise HTTPException(status_code=400, detail="X-Session-ID must be a valid UUID4")
    return session_id
