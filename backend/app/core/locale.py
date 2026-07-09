"""Request-locale resolution for response translation.

A client opts into translation by sending `?lang=fr` (optionally
`&region=CA`) or an `Accept-Language` header. Query params win over the
header. The result is a normalized BCP-47 tag such as "fr" or "fr-CA",
or None when no (supported) locale was requested — None means the
response passes through untranslated.
"""
from __future__ import annotations

import re

from starlette.requests import Request

# Late-bound settings access: modules may be imported while tests have
# app.core.settings.settings patched, so never bind the instance itself.
from app.core import settings as _settings_module

SOURCE_LANGUAGE = "en"

_LANG_RE = re.compile(r"^[a-zA-Z]{2,3}$")
_REGION_RE = re.compile(r"^[a-zA-Z]{2}$")


def _normalize(lang: str | None, region: str | None) -> str | None:
    if not lang:
        return None
    lang = lang.strip().lower()
    if not _LANG_RE.match(lang):
        return None
    if lang == SOURCE_LANGUAGE:
        return None
    if lang not in _settings_module.settings.translation_supported_languages:
        return None
    if region:
        region = region.strip().upper()
        if _REGION_RE.match(region):
            return f"{lang}-{region}"
    return lang


def _parse_accept_language(header: str) -> tuple[str | None, str | None]:
    """Return (lang, region) of the highest-q entry, ignoring wildcards."""
    best: tuple[float, str, str | None] | None = None
    for part in header.split(","):
        piece = part.strip()
        if not piece or piece.startswith("*"):
            continue
        q = 1.0
        if ";" in piece:
            piece, _, params = piece.partition(";")
            piece = piece.strip()
            m = re.search(r"q=([0-9.]+)", params)
            if m:
                try:
                    q = float(m.group(1))
                except ValueError:
                    continue
        subtags = piece.split("-")
        lang = subtags[0]
        region = subtags[1] if len(subtags) > 1 and _REGION_RE.match(subtags[1]) else None
        if best is None or q > best[0]:
            best = (q, lang, region)
    if best is None:
        return None, None
    return best[1], best[2]


def resolve_locale(request: Request) -> str | None:
    """Resolve the requested target locale for this request, or None."""
    if not _settings_module.settings.translation_enabled:
        return None

    lang = request.query_params.get("lang") or request.query_params.get("language")
    region = request.query_params.get("region")
    if lang:
        return _normalize(lang, region)

    header = request.headers.get("accept-language")
    if header:
        h_lang, h_region = _parse_accept_language(header)
        return _normalize(h_lang, h_region)

    return None
