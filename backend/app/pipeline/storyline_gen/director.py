from typing import Any

import numpy as np

from app.pipeline.storyline_gen.candidate_scoring import (
    chapter_coverage,
    theme_coverage,
)


def choose_video_mode(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    requested_speaker = (intent or {}).get("requested_speaker")
    exact_clip_count = int(metrics.get("exact_clip_count") or 0)
    related_clip_count = int(metrics.get("related_clip_count") or 0)
    usable_speaker_clip_count = int(metrics.get("usable_speaker_clip_count") or 0)
    unique_speaker_count = int(metrics.get("unique_speaker_count") or 0)
    unique_source_video_count = int(metrics.get("unique_source_video_count") or 0)
    theme_count = len(metrics.get("theme_coverage") or [])
    total_usable = exact_clip_count + related_clip_count

    distinct_non_null_years = len(set(
        y for y in (metrics.get("video_years") or []) if y is not None
    ))
    # Fall back to unique_source_video_count when speaker field is absent from records
    effective_voice_count = unique_speaker_count if unique_speaker_count > 0 else unique_source_video_count
    if (
        not requested_speaker
        and distinct_non_null_years >= 2
        and exact_clip_count >= 3
        and effective_voice_count >= 2
    ):
        return "chronological_relay"

    if requested_speaker:
        usable_speaker_visual_count = int(metrics.get("usable_speaker_visual_count") or 0)
        total_speaker_usable = usable_speaker_clip_count + usable_speaker_visual_count
        if total_speaker_usable == 0:
            return "no_evidence"
        if total_speaker_usable <= 2:
            return "low_evidence_speaker_brief"
        return "speaker_led_theme_summary"

    if exact_clip_count == 0:
        return "low_evidence_brief"
    if exact_clip_count >= 4 and usable_speaker_clip_count >= 3:
        return "speaker_led_documentary"
    if exact_clip_count >= 3 and unique_speaker_count >= 3 and theme_count <= 4:
        return "relay_of_ideas"
    if exact_clip_count >= 2 and related_clip_count >= 4:
        return "theme_summary_explainer"
    if exact_clip_count >= 1 and related_clip_count >= 3:
        return "narration_led_explainer"
    if total_usable < 3 or unique_source_video_count < 2:
        return "low_evidence_brief"
    return "narration_led_explainer"


def evidence_strength_from_metrics(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    requested_speaker = (intent or {}).get("requested_speaker")
    exact_clip_count = int(metrics.get("exact_clip_count") or 0)
    strong_related_count = int(metrics.get("strong_related_count") or 0)
    usable_speaker_clip_count = int(metrics.get("usable_speaker_clip_count") or 0)

    if requested_speaker:
        usable_speaker_visual_count = int(metrics.get("usable_speaker_visual_count") or 0)
        total_speaker_usable = usable_speaker_clip_count + usable_speaker_visual_count
        if total_speaker_usable >= 4:
            return "high"
        if total_speaker_usable >= 2:
            return "medium"
        if total_speaker_usable == 1:
            return "low"
        return "none"

    if exact_clip_count >= 4 and usable_speaker_clip_count >= 3:
        return "high"
    if exact_clip_count >= 2 or strong_related_count >= 4:
        return "medium"
    return "low"


def get_selected_candidate_cap(target_duration_seconds: int, video_mode: str, evidence_strength: str) -> int:
    if evidence_strength == "low" or video_mode in ("low_evidence_brief", "low_evidence_speaker_brief", "no_evidence"):
        return 5
    if target_duration_seconds <= 60:
        return 10
    if target_duration_seconds <= 120:
        return 16
    if target_duration_seconds <= 180:
        return 24
    # Long targets need a deep candidate pool so the duration-fill ladder has
    # real content to draw from (~1 candidate per 12s of target).
    return max(30, target_duration_seconds // 12)


def fallback_risk_from_metrics(metrics: dict[str, Any]) -> str:
    total_usable = int(metrics.get("exact_clip_count") or 0) + int(metrics.get("related_clip_count") or 0)
    unique_source_video_count = int(metrics.get("unique_source_video_count") or 0)

    if total_usable >= 6 and unique_source_video_count >= 4:
        return "low"
    if total_usable >= 3:
        return "medium"
    return "high"


def build_director_plan(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> dict[str, Any]:
    requested_speaker = (intent or {}).get("requested_speaker")
    video_mode = choose_video_mode(metrics, intent=intent)
    evidence_strength = evidence_strength_from_metrics(metrics, intent=intent)

    structure_map = {
        "chronological_relay": [
            {
                "role": "temporal_context",
                "section_type": "narration",
                "narration_guidance": "Establish the historical or multi-year arc — describe how the topic has evolved across conference years.",
                "evidence_expectation": "clips or narration grounded in the earliest year represented in the candidate pool",
            },
            {
                "role": "human_problem",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A speaker articulates the core human challenge or gap that drives the relay of ideas across time.",
                "evidence_expectation": "exact-match soundbite from any year that names the problem clearly",
            },
            {
                "role": "risk_escalation",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A voice from a later year escalates the stakes — what has worsened, spread, or gone unaddressed.",
                "evidence_expectation": "strong-related or exact clip from a more recent year than the human_problem clip",
            },
            {
                "role": "solution_direction",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A speaker proposes or describes a concrete direction, policy, or intervention pointing toward resolution.",
                "evidence_expectation": "exact or strong-related clip with forward-looking or prescriptive language",
            },
            {
                "role": "synthesis",
                "section_type": "narration",
                "narration_guidance": "Narrator closes the relay — summarising what the sequence of voices reveals about progress, persistence of the problem, or urgency for action.",
                "evidence_expectation": "narration only; may reference multiple clips from the relay by year",
            },
        ],
        "speaker_led_documentary": [
            "opening_hook",
            "exact_evidence_clip",
            "speaker_voice_two",
            "bridge_narration",
            "speaker_voice_three",
            "closing_takeaway",
        ],
        "relay_of_ideas": [
            "opening_context",
            "idea_one",
            "idea_two",
            "tension_point",
            "synthesis_narration",
            "closing_takeaway",
        ],
        "theme_summary_explainer": [
            "opening_context",
            "related_theme_clip",
            "explanation_bridge",
            "exact_evidence_clip",
            "related_governance_clip",
            "closing_takeaway",
        ],
        "narration_led_explainer": [
            "opening_context",
            "exact_evidence_clip",
            "related_theme_clip",
            "synthesis_narration",
            "related_supporting_clip",
            "closing_takeaway",
        ],
        "low_evidence_brief": [
            "opening_context",
            "available_evidence_clip",
            "evidence_gap_narration",
            "related_context",
            "closing_takeaway",
        ],
        "speaker_led_theme_summary": [
            "opening_context",
            "exact_evidence_clip",
            "exact_evidence_clip",
            "bridge_narration",
            "exact_evidence_clip",
            "closing_takeaway",
        ],
        "low_evidence_speaker_brief": [
            "opening_context",
            "available_evidence_clip",
            "evidence_gap_narration",
            "closing_takeaway",
        ],
        "no_evidence": [
            "opening_context",
            "evidence_gap_narration",
            "closing_takeaway",
        ],
    }

    notes = []
    exact_clip_count = metrics.get("exact_clip_count", 0)
    usable_soundbite_count = metrics.get("usable_speaker_clip_count", 0)
    usable_visual_count = metrics.get("usable_speaker_visual_count", 0)
    total_speaker_usable = usable_soundbite_count + usable_visual_count
    if requested_speaker and total_speaker_usable == 0:
        notes.append(f"No archive clips were found for {requested_speaker}. No unrelated speakers were used as filler.")
    elif requested_speaker and total_speaker_usable <= 2:
        notes.append(f"Limited archive evidence was found for {requested_speaker}. The video uses only available {requested_speaker} clips and does not include unrelated speakers.")
    elif exact_clip_count == 0:
        notes.append("No direct evidence clips found for this prompt. Direct topic evidence is limited or absent.")
    elif exact_clip_count <= 2:
        notes.append("Exact prompt evidence was limited, so the plan leans on related theme and chapter evidence with narration bridges.")
    if usable_soundbite_count == 0 and usable_visual_count == 0:
        notes.append("No speaker clips are approved for soundbite or visual use. Narration and b-roll will carry the video.")
    elif usable_soundbite_count == 0 and usable_visual_count > 0:
        notes.append(f"{usable_visual_count} speaker visual(s) are approved for silent/muted use under narration. No high-quality soundbites available.")
    elif metrics.get("unique_speaker_count", 0) < 3:
        notes.append("Speaker diversity is limited, so narration carries more of the connective explanation.")
    if not notes:
        notes.append("The candidate pool is diverse enough to support a multi-voice documentary structure.")

    return {
        "enabled": True,
        "video_mode": video_mode,
        "evidence_strength": evidence_strength,
        "exact_clip_count": metrics.get("exact_clip_count", 0),
        "related_clip_count": metrics.get("related_clip_count", 0),
        "strong_related_count": metrics.get("strong_related_count", 0),
        "usable_speaker_clip_count": usable_soundbite_count,
        "usable_speaker_visual_count": usable_visual_count,
        "rejected_clip_count": metrics.get("rejected_clip_count", 0),
        "unique_speaker_count": metrics.get("unique_speaker_count", 0),
        "unique_source_video_count": metrics.get("unique_source_video_count", 0),
        "theme_coverage": metrics.get("theme_coverage", []),
        "chapter_coverage": metrics.get("chapter_coverage", []),
        "average_similarity": metrics.get("average_similarity"),
        "average_quote_strength": metrics.get("average_quote_strength"),
        "recommended_structure": structure_map[video_mode],
        "fallback_risk": fallback_risk_from_metrics(metrics),
        "planning_notes": notes,
        "requested_speaker": requested_speaker,
    }


def compute_evidence_metrics(
    selected: list[dict[str, Any]],
    supporting: list[dict[str, Any]] | None = None,
    context: list[dict[str, Any]] | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    supporting = supporting or []
    context = context or []
    rejected = rejected or []

    exact_records = [r for r in selected if r.get("relevance_role") == "exact"]
    strong_related = [r for r in selected if r.get("relevance_role") == "strong_related"]
    weak_in_selected = [r for r in selected if r.get("relevance_role") == "weak_related"]
    usable_speaker_clips = [r for r in selected if r.get("speaker_soundbite_allowed") is True or r.get("speaker_clip_allowed") is True]
    usable_speaker_visuals = [r for r in selected if r.get("speaker_visual_allowed") is True]
    semantic_only_matches = [r for r in selected if r.get("semantic_only_match") is True]
    context_promoted = [r for r in selected if r.get("promoted_from_context") is True]

    usable_records = selected + supporting

    speakers = {str(r.get("speaker")) for r in selected if r.get("speaker")}
    sources = {str(r.get("source_video")) for r in selected if r.get("source_video")}
    video_years = [r.get("video_year") for r in selected]
    similarities = [float(r.get("semantic_score") or 0.0) for r in usable_records]
    quote_strengths = [float(r.get("quote_strength") or 0.0) for r in usable_records if r.get("quote_strength") is not None]

    return {
        "exact_clip_count": len(exact_records),
        "related_clip_count": len(strong_related) + len(weak_in_selected),
        "strong_related_count": len(strong_related),
        "weak_related_count": len(weak_in_selected),
        "supporting_candidate_count": len(supporting),
        "contextual_count": len(context),
        "rejected_clip_count": len(rejected),
        "usable_speaker_clip_count": len(usable_speaker_clips),
        "usable_speaker_visual_count": len(usable_speaker_visuals),
        "semantic_only_match_count": len(semantic_only_matches),
        "context_promoted_count": len(context_promoted),
        "unique_speaker_count": len(speakers),
        "unique_source_video_count": len(sources),
        "video_years": video_years,
        "theme_coverage": theme_coverage(selected),
        "chapter_coverage": chapter_coverage(selected),
        "average_similarity": round(float(np.mean(similarities)), 6) if similarities else None,
        "average_quote_strength": round(float(np.mean(quote_strengths)), 6) if quote_strengths else None,
    }


def bucket_candidates(
    classified: list[dict[str, Any]],
    intent: dict[str, Any],
    max_low_evidence_weak: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    requested_speaker = intent.get("requested_speaker")

    selected: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for c in classified:
        role = c.get("relevance_role")
        fva = c.get("final_video_allowed", False)
        use_as = c.get("use_as", "")
        if role in ("exact", "strong_related") and fva:
            selected.append(c)
        elif role == "weak_related" and use_as == "speaker_visual" and fva:
            # Speaker-matched weak clip: allowed as a visual under narration
            selected.append(c)
        elif role == "weak_related":
            supporting.append(c)
        elif role == "contextual":
            context.append(c)
        else:
            rejected.append(c)

    # If no selected candidates, promote top weak_related as narration_support
    # (low-evidence mode). If that still isn't enough to fill
    # max_low_evidence_weak slots, fall back further to the best-scoring
    # context (contextual-role, semantic-only) candidates rather than leaving
    # `selected` empty -- a candidate with no keyword overlap but a high
    # semantic_score is still better last-resort evidence than nothing.
    if not selected:
        promoted_from_supporting = supporting[:max_low_evidence_weak]
        remaining_slots = max_low_evidence_weak - len(promoted_from_supporting)
        promoted_from_context: list[dict[str, Any]] = []
        if remaining_slots > 0 and context:
            context_sorted = sorted(context, key=lambda c: float(c.get("semantic_score") or 0.0), reverse=True)
            promoted_from_context = context_sorted[:remaining_slots]

        for c in promoted_from_supporting + promoted_from_context:
            c = dict(c)
            c["final_video_allowed"] = True
            c["use_as"] = "narration_support"
            c["speaker_soundbite_allowed"] = False
            c["speaker_visual_allowed"] = False
            c["speaker_clip_allowed"] = False
            c["promoted_from_context"] = c.get("relevance_role") == "contextual"
            selected.append(c)

        supporting = supporting[max_low_evidence_weak:]
        if promoted_from_context:
            promoted_context_ids = {c.get("fragment_id") for c in promoted_from_context}
            context = [c for c in context if c.get("fragment_id") not in promoted_context_ids]

    # Speaker prompt hard enforcement: remove non-matching from selected
    if requested_speaker:
        bad = [c for c in selected if not c.get("speaker_match")]
        if bad:
            for c in bad:
                c2 = dict(c)
                c2["final_video_allowed"] = False
                c2["use_as"] = "reject"
                rejected.append(c2)
            selected = [c for c in selected if c.get("speaker_match")]

    return {
        "selected_candidates": selected,
        "supporting_candidates": supporting,
        "context_candidates": context,
        "rejected_candidates": rejected,
    }
