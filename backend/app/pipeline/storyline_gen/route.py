"""Apply the intent router's route to the stage 07 prompt profile (docs/design/intent-layer-design.md §7.5.5).

Intent selection runs several times and the first matching intent wins, so the
route sets and clears every field it reads: a leftover speaker or title would
otherwise take precedence.
"""
from __future__ import annotations

import re
from typing import Any

from app.pipeline.storyline_gen.prompt_intent import PROMPT_TOPIC_CONFIG, detect_prompt_intent
from app.pipeline.storyline_gen.text_utils import normalize_name, stem_term


def _chapter_name(number: int | None, records: list[dict[str, Any]]) -> str | None:
    if number is None:
        return None
    pattern = re.compile(rf"\s*Ch\s*0*{number}\b")
    return next((str(r["primary_chapter"]).strip() for r in records
                 if r.get("primary_chapter") and pattern.match(str(r["primary_chapter"]))), None)


def _facet_holds(topic_terms: list[str], facet: str | None) -> bool:
    """Every topic word is part of the theme or chapter name ("health" in "AI in Health, ...")."""
    facet_stems = {stem_term(t) for t in re.findall(r"[a-z0-9]+", normalize_name(facet or ""))}
    return bool(topic_terms) and all(stem_term(t) in facet_stems for t in topic_terms)


def _set_evidence(profile: dict[str, Any]) -> None:
    """Scope, evidence and strictness for the intent the route chose, as build_prompt_profile sets them."""
    topic_terms = profile.get("topic_terms") or []
    if profile["intent_type"] == "archive_overview":
        return
    has_subject = bool(topic_terms or profile.get("theme_hint") or profile.get("chapter_hint"))
    profile["scope"] = "specific_topic" if has_subject else "general"
    if profile.get("requested_speaker"):
        profile.update(evidence_requirement="speaker_source_evidence", strictness="high")
        return
    strictness = [(PROMPT_TOPIC_CONFIG.get(t) or {}).get("strictness") for t in topic_terms]
    profile["strictness"] = "high" if "high" in strictness else "low" if "low" in strictness else "medium"
    profile["evidence_requirement"] = ("broad_theme_evidence" if profile["intent_type"] == "theme_video"
                                       else "direct_topic_evidence")


def profile_for_route(
    prompt: str,
    known_speakers: list[str],
    records: list[dict[str, Any]],
    route: dict[str, Any] | None,
    target_duration_seconds: int | None = None,
    requested_speaker: str | None = None,
) -> dict[str, Any]:
    """The prompt profile stage 07 plans with: the rules profile, or the route applied over it."""
    rules = detect_prompt_intent(prompt, known_speakers, records, target_duration_seconds=target_duration_seconds,
                                 requested_speaker=requested_speaker)
    if not route or route.get("method") != "router":
        return rules
    intent = route.get("intent")
    # Intents whose data step is not built yet keep today's behaviour (§7.5.5).
    if intent == "overview" and route.get("year"):
        return rules
    if intent == "counting":
        intent = "topic"
    speakers = route.get("speakers") or []
    speaker = speakers[0] if intent in ("speaker", "comparison") and speakers else None
    if speaker is None and requested_speaker and intent in ("speaker", "comparison") \
            and requested_speaker not in (route.get("excluded_speakers") or []):
        # The router reads a person but its name did not validate: use the one the resolver found.
        # When it reads a topic, a name in the text (quoted, excluded, injected) stays out.
        speaker, intent = requested_speaker, "speaker"

    profile = detect_prompt_intent(prompt, known_speakers, records, target_duration_seconds=target_duration_seconds,
                                   requested_speaker=speaker, topic_terms=route.get("topic_terms") or [])
    profile["requested_speaker"] = speaker
    profile["speaker_required"] = speaker is not None

    source_video = None
    if intent == "title":
        source_video = route.get("source_video")
    elif intent in ("topic", "theme_chapter", "overview", "chronological"):
        source_video = rules.get("requested_source_video")  # an exact title match is a fact
    if source_video != profile.get("requested_source_video"):
        profile["requested_source_video"] = source_video
        profile["title_match_ratio"] = rules.get("title_match_ratio", 0.0) if source_video else 0.0
        profile["title_matched_terms"] = rules.get("title_matched_terms", []) if source_video else []

    topic_terms = profile.get("topic_terms") or []
    if intent == "theme_chapter":
        chapter = _chapter_name(route.get("chapter"), records)
        profile["chapter_hint"] = chapter
        profile["theme_hint"] = route.get("theme") or profile.get("theme_hint")
        profile["intent_type"] = "chapter_video" if chapter else "theme_video"
    elif intent == "overview":
        profile.update(intent_type="archive_overview", scope="archive_overview",
                       evidence_requirement="corpus_facts", topic_terms=[], expanded_terms=[])
    elif speaker:
        profile["intent_type"] = "speaker_topic_summary"
    elif source_video:
        profile["intent_type"] = "topic_video"
    elif rules.get("intent_type") in ("theme_video", "chapter_video") and (
            _facet_holds(topic_terms, rules.get("theme_hint")) or _facet_holds(topic_terms, rules.get("chapter_hint"))):
        # The rules found the facet by name, and the router's topic sits inside it.
        profile.update(intent_type=rules["intent_type"], theme_hint=rules.get("theme_hint"),
                       chapter_hint=rules.get("chapter_hint"))
    else:
        profile["intent_type"] = "topic_video"

    _set_evidence(profile)
    for term in route.get("exclude_terms") or []:
        if term not in profile["negative_terms"]:
            profile["negative_terms"].append(term)
    profile["excluded_speakers"] = list(route.get("excluded_speakers") or [])
    if route.get("language", "en") != "en":
        # The archive is embedded in English: search with the router's English topic.
        profile["retrieval_query"] = " ".join(filter(None, [speaker, route.get("topic_text")])) or prompt
    profile["route"] = {k: route.get(k) for k in ("method", "intent", "year", "language", "aggregate", "count_unit")}
    return profile


def drop_excluded_speakers(candidates: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Fragments by a speaker the prompt asked to leave out ("but not Cordel Green")."""
    excluded = set(profile.get("excluded_speakers") or [])
    if not excluded:
        return candidates
    from app.services.speaker_registry import get_registry

    registry = get_registry()
    return [c for c in candidates if (registry.canonical(str(c.get("speaker") or "")) or c.get("speaker")) not in excluded]
