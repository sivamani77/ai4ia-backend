"""Evidence some intents add to the storyline payload, built once per job in stage 07."""
from __future__ import annotations

from typing import Any


def speaker_panel_only(speaker: str | None) -> bool:
    """True when every registry video the speaker appears in is a panel."""
    from app.services.speaker_registry import get_registry

    videos = get_registry().videos_of(speaker) if speaker else []
    return bool(videos) and all(v.kind == "panel" for _, v in videos)


def build_intent_context(profile: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    from app.pipeline.storyline_gen.corpus_facts import build_corpus_facts, format_facts_for_prompt
    from app.services.speaker_registry import get_registry

    registry = get_registry()
    context: dict[str, Any] = {}
    if profile.get("intent_type") == "archive_overview":
        facts = build_corpus_facts(records, registry_speaker_count=registry.recorded_speaker_count())
        context["archive_facts"] = format_facts_for_prompt(facts)
    if profile.get("speaker_panel_only"):
        titles = {}
        for r in records:
            titles.setdefault(str(r.get("video_id") or ""), r.get("video_title") or r.get("source_video"))
        context["panel_context"] = [{"video": titles.get(vid) or vid, "panel": v.names}
                                    for vid, v in registry.videos_of(profile["requested_speaker"])]
    if profile.get("theme_hint") or profile.get("chapter_hint"):
        context["requested_facet"] = {k: profile[k] for k in ("theme_hint", "chapter_hint") if profile.get(k)}
    return context
