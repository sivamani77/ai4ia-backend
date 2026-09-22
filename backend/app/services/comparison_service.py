"""Comparison, theme synthesis, and thematic relay.

All three reuse the citation-first contract from insight_service: the LLM may
only cite from a numbered evidence pool, every claim carries fragment_refs, and
we validate refs after generation. The relay is a thematic idea-thread ACROSS
SPEAKERS (not across years — this archive is a single edition), so it embodies
the "Relay of Ideas" with the axis being voices building on a theme.
"""
from datetime import datetime, timezone
from typing import Any

from app.core.config import INSIGHT_MODEL
from app.services import embedding_index, evidence_gate, fragment_store, guardrail_service
from app.services.insight_service import (
    DECLINE_TITLE,
    OUTCOME_GENERATED,
    OUTCOME_INSUFFICIENT_EVIDENCE,
    _truncate,
    _validate,
)
from app.services.llm_service import generate_json

_DECLINE_REASON = (
    "No fragments in the archive matched this closely enough to support an "
    "evidence-grounded answer."
)


def _declined_relay(topic: str, diagnostics: dict) -> dict:
    """Same shape as a generated relay, every collection empty. No LLM call."""
    return {
        "topic": topic,
        "outcome": OUTCOME_INSUFFICIENT_EVIDENCE,
        "decline_reason": _DECLINE_REASON,
        "decline_diagnostics": diagnostics,
        "relay_title": DECLINE_TITLE,
        "relay_narrative": None,
        "relay_steps": [],
        "step_count": 0,
        "distinct_speakers": 0,
        "edition": "AI4IA 2025",
        "models": {"llm": None},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_explain": {
            "how_it_works": (
                "Semantic search found nothing close enough to this topic, so no "
                "relay was built and the LLM was never called."
            ),
            "retrieval": {"method": "semantic_cosine_similarity", **diagnostics},
            "generation": {"model": None, "skipped": True},
        },
    }


def _declined_theme(theme: str, diagnostics: dict) -> dict:
    """Same shape as a generated theme synthesis, every collection empty."""
    return {
        "theme": theme,
        "outcome": OUTCOME_INSUFFICIENT_EVIDENCE,
        "decline_reason": _DECLINE_REASON,
        "decline_diagnostics": diagnostics,
        "title": DECLINE_TITLE,
        "summary": None,
        "key_points": [],
        "representative_clips": [],
        "unsupported_claims": [],
        "evidence_pool": [],
        "models": {"llm": None},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_explain": {
            "how_it_works": (
                "Neither the theme facet nor the semantic fallback found anything "
                "close enough, so no synthesis was attempted and the LLM was "
                "never called."
            ),
            "retrieval": {"method": "theme_facet_match → semantic_fallback", **diagnostics},
            "generation": {"model": None, "skipped": True},
        },
    }


def _clip(p: dict) -> dict:
    return {
        "fragment_id": p.get("fragment_id"), "speaker": p.get("speaker"),
        "speaker_attribution": p.get("speaker_attribution"),
        "edition": p.get("edition"),
        "primary_theme": p.get("primary_theme"),
        "video_stream_url": p.get("stream_url"),
        "start": p.get("start"), "end": p.get("end"), "duration": p.get("duration"),
        "text": _truncate(p.get("text", ""), 400),
    }


def _pool_from_fragments(frags: list[dict]) -> list[dict]:
    """Number a fragment list [1..N] as a citation pool (used by speaker compare)."""
    pool = []
    for i, f in enumerate(frags, start=1):
        pool.append({
            "ref": i, "fragment_id": f.get("fragment_id"), "speaker": f.get("speaker"),
            "speaker_attribution": f.get("speaker_attribution"), "edition": f.get("edition"),
            "primary_theme": f.get("primary_theme"), "primary_chapter": f.get("primary_chapter"),
            "start": f.get("start"), "end": f.get("end"), "duration": f.get("duration"),
            "stream_url": fragment_store.stream_url_for(f.get("source_video")), "text": f.get("text"),
        })
    return pool


def _format_pool(pool: list[dict]) -> str:
    lines = []
    for p in pool:
        meta = " | ".join(filter(None, [p.get("speaker"), p.get("primary_theme")]))
        lines.append(f"[{p['ref']}] ({meta}) {_truncate(p['text'])}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- speakers
_COMPARE_PROMPT = """You are comparing how different AI4IA speakers address a topic.

Topic: {topic}

Numbered evidence fragments (each tagged with its speaker). Use ONLY these.
Every statement MUST cite supporting fragment numbers via "fragment_refs".
Do NOT invent. A claim with no supporting fragment must be omitted.

EVIDENCE:
{evidence}

Speakers being compared: {speakers}

Return JSON exactly:
{{
  "per_speaker": [{{"speaker": "<name>", "position": "<their stance in 1-2 sentences>",
                    "fragment_refs": [1]}}],
  "agreements": [{{"statement": "<shared view>", "fragment_refs": [1,4]}}],
  "tensions":   [{{"statement": "<disagreement/contrast>", "fragment_refs": [2,5]}}],
  "synthesis": "<2-3 sentence cross-speaker takeaway>"
}}"""


def compare_speakers(speakers: list[str], topic: str | None = None,
                     limit_per_speaker: int = 6, model: str | None = None,
                     user_id: str | None = None) -> dict:
    # Vet the caller's own words only — the default topic below is ours.
    if topic:
        guardrail_service.assert_query_allowed(topic, surface="compare_speakers", user_id=user_id)
    topic = topic or "their overall perspective on AI"
    pool: list[dict] = []
    missing: list[str] = []
    for name in speakers:
        frags = fragment_store.by_facet("speaker", name)
        if not frags:
            missing.append(name)
            continue
        # If a topic is given, rank this speaker's fragments by similarity to it.
        if topic:
            try:
                ranked = embedding_index.search(topic, top_k=limit_per_speaker,
                                                filters={"speaker": name})
                frags = ranked or frags[:limit_per_speaker]
            except Exception:  # noqa: BLE001
                frags = frags[:limit_per_speaker]
        else:
            frags = frags[:limit_per_speaker]
        pool.extend(frags)

    if not pool:
        raise ValueError(f"No fragments found for speakers: {speakers}")

    numbered = _pool_from_fragments(pool)
    valid = {p["ref"] for p in numbered}
    raw = generate_json(
        _COMPARE_PROMPT.format(topic=topic, evidence=_format_pool(numbered),
                               speakers=", ".join(speakers)),
        model=model or INSIGHT_MODEL, temperature=0.2,
    )

    agreements, a_bad = _validate(raw.get("agreements"), valid)
    tensions, t_bad = _validate(raw.get("tensions"), valid)
    per_speaker, _ = _validate(raw.get("per_speaker"), valid)

    total_claims = len(agreements) + len(tensions) + len(a_bad) + len(t_bad)
    supported_claims = len(agreements) + len(tensions)
    coverage = round(supported_claims / total_claims, 3) if total_claims else 0.0

    return {
        "topic": topic,
        "speakers_requested": speakers,
        "speakers_missing": missing,
        "per_speaker": per_speaker,
        "agreements": agreements,
        "tensions": tensions,
        "synthesis": raw.get("synthesis"),
        "unsupported_claims": a_bad + t_bad,
        "evidence_pool": numbered,
        "models": {"llm": model or INSIGHT_MODEL},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_explain": {
            "how_it_works": (
                "For each requested speaker, their fragments are retrieved from the fragment store. "
                "If a topic is given, fragments are re-ranked by semantic similarity to that topic "
                "using nomic-embed-text:v1.5. All speakers' fragments are merged into one numbered "
                "evidence pool. The LLM (openai/gpt-oss-120b) must cite fragment numbers for every "
                "agreement, tension, and per-speaker position it identifies. Claims without valid "
                "fragment references are dropped."
            ),
            "retrieval": {
                "method": "fragment_store_by_speaker + semantic_reranking_if_topic",
                "embedding_model": "nomic-embed-text:v1.5",
                "limit_per_speaker": limit_per_speaker,
                "total_fragments_in_pool": len(numbered),
                "speakers_found": [s for s in speakers if s not in missing],
                "speakers_missing": missing,
            },
            "generation": {
                "model": model or INSIGHT_MODEL,
                "temperature": 0.2,
                "citation_contract": "LLM may only cite numbered fragments from the evidence pool",
            },
            "validation": {
                "total_claims_generated": total_claims,
                "supported_claims": supported_claims,
                "unsupported_claims_dropped": len(a_bad) + len(t_bad),
                "evidence_coverage_score": coverage,
            },
        },
    }


# ----------------------------------------------------------------------------- theme
_THEME_PROMPT = """You are synthesizing how the AI4IA archive treats a theme.

Theme: {theme}

Numbered evidence fragments. Use ONLY these. Every statement MUST cite
supporting fragment numbers via "fragment_refs". Do NOT invent.

EVIDENCE:
{evidence}

Return JSON exactly:
{{
  "title": "<evocative title>",
  "summary": "<3-4 sentence synthesis>",
  "key_points": [{{"point": "<idea>", "fragment_refs": [1,2]}}],
  "representative_clips": [{{"ref": 1, "reason": "<why>"}}]
}}"""


def synthesize_theme(theme: str, limit: int = 20, model: str | None = None,
                     user_id: str | None = None) -> dict:
    guardrail_service.assert_query_allowed(theme, surface="synthesis_theme", user_id=user_id)
    frags = embedding_index.search(theme, top_k=limit, filters={"theme": theme})
    if not frags:  # fall back to semantic-only if the theme string isn't an exact facet
        frags = embedding_index.search(theme, top_k=limit)
    sufficient, gate = evidence_gate.evidence_is_sufficient(frags, surface="synthesis_theme")
    if not sufficient:
        return _declined_theme(theme, gate)

    numbered = _pool_from_fragments(frags)
    valid = {p["ref"] for p in numbered}
    raw = generate_json(
        _THEME_PROMPT.format(theme=theme, evidence=_format_pool(numbered)),
        model=model or INSIGHT_MODEL, temperature=0.2,
    )
    key_points, kp_bad = _validate(raw.get("key_points"), valid)
    rep = []
    for r in raw.get("representative_clips") or []:
        ref = r.get("ref")
        if isinstance(ref, int) and ref in valid:
            p = next(x for x in numbered if x["ref"] == ref)
            rep.append({**_clip(p), "reason": r.get("reason")})

    total_kp = len(key_points) + len(kp_bad)
    kp_coverage = round(len(key_points) / total_kp, 3) if total_kp else 0.0

    return {
        "theme": theme,
        "outcome": OUTCOME_GENERATED,
        "title": raw.get("title") or theme,
        "summary": raw.get("summary"),
        "key_points": key_points,
        "representative_clips": rep,
        "unsupported_claims": kp_bad,
        "evidence_pool": numbered,
        "models": {"llm": model or INSIGHT_MODEL},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_explain": {
            "how_it_works": (
                "Fragments are retrieved by exact theme facet match first; if that returns nothing, "
                "a pure semantic search on the theme string is used as fallback. "
                "The numbered pool is fed to openai/gpt-oss-120b which must cite fragment numbers for "
                "every key point. Uncited key points are dropped. Representative clips are resolved "
                "from the pool by their ref number."
            ),
            "retrieval": {
                "method": "theme_facet_match → semantic_fallback",
                "embedding_model": "nomic-embed-text:v1.5",
                "fragments_in_pool": len(numbered),
                "limit_requested": limit,
            },
            "generation": {
                "model": model or INSIGHT_MODEL,
                "temperature": 0.2,
                "citation_contract": "LLM may only cite numbered fragments from the evidence pool",
            },
            "validation": {
                "key_points_generated": total_kp,
                "key_points_supported": len(key_points),
                "key_points_dropped": len(kp_bad),
                "evidence_coverage_score": kp_coverage,
                "representative_clips_resolved": len(rep),
            },
        },
    }


# ----------------------------------------------------------------------------- relay
_RELAY_PROMPT = """You are building a "Relay of Ideas": an ordered narrative thread on a
topic where each step is a DIFFERENT speaker's clip, sequenced so the idea
develops — introduction, then development, then a tension or counterpoint, then
a synthesis. This is a relay across VOICES, not across time.

Topic: {topic}

Numbered candidate clips (each from a speaker). Use ONLY these. Pick 3-6 that
form the best thread. Each step MUST reference exactly one clip number.

CLIPS:
{evidence}

Return JSON exactly:
{{
  "relay_title": "<title>",
  "relay_steps": [{{"ref": 1, "role_in_relay": "opening|development|tension|synthesis",
                    "bridge": "<1 sentence connecting this voice to the thread>"}}],
  "relay_narrative": "<short paragraph describing the through-line>"
}}"""


def build_relay(topic: str, limit: int = 20, model: str | None = None,
                user_id: str | None = None) -> dict:
    guardrail_service.assert_query_allowed(topic, surface="relay", user_id=user_id)
    frags = embedding_index.search(topic, top_k=limit)
    sufficient, gate = evidence_gate.evidence_is_sufficient(frags, surface="relay")
    if not sufficient:
        return _declined_relay(topic, gate)

    # Prefer diversity of speakers in the candidate pool.
    seen, diverse = set(), []
    for f in frags:
        sp = f.get("speaker")
        if sp not in seen:
            seen.add(sp)
            diverse.append(f)
    pool_frags = (diverse + [f for f in frags if f not in diverse])[:limit]

    numbered = _pool_from_fragments(pool_frags)
    valid = {p["ref"] for p in numbered}
    raw = generate_json(
        _RELAY_PROMPT.format(topic=topic, evidence=_format_pool(numbered)),
        model=model or INSIGHT_MODEL, temperature=0.3,
    )

    steps = []
    for s in raw.get("relay_steps") or []:
        ref = s.get("ref")
        if isinstance(ref, int) and ref in valid:
            p = next(x for x in numbered if x["ref"] == ref)
            steps.append({**_clip(p), "role_in_relay": s.get("role_in_relay"),
                          "bridge": s.get("bridge")})

    return {
        "topic": topic,
        "outcome": OUTCOME_GENERATED,
        "relay_title": raw.get("relay_title") or topic,
        "relay_narrative": raw.get("relay_narrative"),
        "relay_steps": steps,
        "step_count": len(steps),
        "distinct_speakers": len({s["speaker"] for s in steps if s.get("speaker")}),
        "edition": "AI4IA 2025",
        "models": {"llm": model or INSIGHT_MODEL},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_explain": {
            "how_it_works": (
                "A semantic search retrieves the top-K fragments most similar to the topic. "
                "The candidate pool is deduplicated by speaker so no voice dominates — each unique "
                "speaker appears at most once in the first pass. The LLM then selects 3-6 clips "
                "and assigns each a role in the relay thread (opening / development / tension / "
                "synthesis) plus a one-sentence bridge explaining how that voice advances the idea. "
                "Only clips whose ref resolves to a real pool entry are included in relay_steps."
            ),
            "retrieval": {
                "method": "semantic_cosine_similarity + speaker_diversity_dedup",
                "embedding_model": "nomic-embed-text:v1.5",
                "fragments_retrieved": len(frags),
                "unique_speakers_in_pool": len(seen),
                "pool_size_after_dedup": len(pool_frags),
            },
            "generation": {
                "model": model or INSIGHT_MODEL,
                "temperature": 0.3,
                "relay_roles": ["opening", "development", "tension", "synthesis"],
                "target_steps": "3-6",
            },
            "validation": {
                "steps_selected_by_llm": len(raw.get("relay_steps") or []),
                "steps_resolved_to_real_clips": len(steps),
                "distinct_speakers_in_relay": len({s["speaker"] for s in steps if s.get("speaker")}),
            },
        },
    }
