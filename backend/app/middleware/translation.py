"""Response-translation middleware.

When a request resolves to a target locale (see app.core.locale), buffer
the JSON response, translate the values of allowlisted prose keys via
translation_service, and re-emit the body. Everything else — IDs, enums,
URLs, timestamps, numbers, non-JSON responses, errors — passes through
untouched. Fail-open: any problem returns the original response.

The X-Translation header reports what happened: "full", "partial"
(provider failed for some strings), or "skipped" (body too large or not
translatable).
"""
from __future__ import annotations

import asyncio
import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.locale import resolve_locale
from app.core import settings as _settings_module

logger = logging.getLogger("ai4ia.middleware.translation")

# JSON keys whose string values are human-readable prose. Anything not
# listed here is never touched. Extend per-route findings here (Phase 3).
TRANSLATABLE_KEYS = frozenset({
    # generic prose
    "title",
    "description",
    "summary",
    "narration",
    "text",
    "quote",
    "headline",
    "subtitle",
    "tagline",
    "insight",
    "rationale",
    "reasoning",
    "explanation",
    "message",
    "caption",
    "overview",
    "takeaway",
    "question",
    "answer",
    "conclusion",
    "statement",
    # insights / theme synthesis / compare
    "card_text",
    "key_takeaway",
    "closing_takeaway",
    "why_it_matters",
    "how_it_works",
    "opening_hook",
    "opening_context",
    "synthesis",
    "synthesis_narration",
    "supporting_quote",
    "evidence_gap_narration",
    "key_points",
    "main_ideas",
    # relay of ideas
    "relay_narrative",
    "relay_title",
    "role_in_relay",
    "bridge",
    # storyline / video pipeline
    "logline",
    "voiceover_text",
    "quote_text",
    "video_overview",
    "video_title",
    "visual_title",
    "video_mode_rationale",
    "relevance_reason",
    "planning_note",
    "planning_notes",
    "low_evidence_brief",
    "low_evidence_speaker_brief",
    "theme_hint",
})

# Endpoints that must never be translated (operational/diagnostic surfaces).
EXCLUDED_PATH_SUBSTRINGS = (
    "/health",
    "/admin",
    "/provenance",
)


def collect_strings(node: object, out: list[str], parent_key: str | None = None) -> None:
    """Depth-first collection of translatable string values.

    A string is translatable when its own dict key is allowlisted, or when
    it sits inside a list whose key is allowlisted (e.g. key_points: [...]).
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str):
                if key in TRANSLATABLE_KEYS and value.strip():
                    out.append(value)
            else:
                collect_strings(value, out, key)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                if parent_key in TRANSLATABLE_KEYS and item.strip():
                    out.append(item)
            else:
                collect_strings(item, out, parent_key)


def apply_translations(node: object, mapping: dict[str, str], parent_key: str | None = None) -> object:
    """Return a copy of `node` with allowlisted string values replaced."""
    if isinstance(node, dict):
        result = {}
        for key, value in node.items():
            if isinstance(value, str) and key in TRANSLATABLE_KEYS:
                result[key] = mapping.get(value, value)
            else:
                result[key] = apply_translations(value, mapping, key)
        return result
    if isinstance(node, list):
        return [
            mapping.get(item, item)
            if isinstance(item, str) and parent_key in TRANSLATABLE_KEYS
            else apply_translations(item, mapping, parent_key)
            for item in node
        ]
    return node


class TranslationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        locale = resolve_locale(request)
        if locale is None:
            return await call_next(request)

        if any(part in request.url.path for part in EXCLUDED_PATH_SUBSTRINGS):
            return await call_next(request)

        response = await call_next(request)

        if response.status_code != 200:
            return response
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        def _rebuild(new_body: bytes, translation_state: str) -> Response:
            headers = dict(response.headers)
            headers["content-length"] = str(len(new_body))
            headers["x-translation"] = translation_state
            return Response(
                content=new_body,
                status_code=response.status_code,
                headers=headers,
                media_type="application/json",
            )

        if len(body) > _settings_module.settings.translation_max_body_bytes:
            return _rebuild(body, "skipped")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return _rebuild(body, "skipped")

        texts: list[str] = []
        collect_strings(payload, texts)
        if not texts:
            return _rebuild(body, "skipped")

        try:
            from app.services.translation_service import translate_batch

            translated, complete = await asyncio.to_thread(
                translate_batch, texts, locale
            )
            mapping = dict(zip(texts, translated))
            new_payload = apply_translations(payload, mapping)
            new_body = json.dumps(new_payload, ensure_ascii=False).encode("utf-8")
            return _rebuild(new_body, "full" if complete else "partial")
        except Exception:  # noqa: BLE001
            logger.exception("translation middleware failed; returning original body")
            return _rebuild(body, "skipped")
