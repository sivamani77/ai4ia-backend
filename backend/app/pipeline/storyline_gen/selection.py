from typing import Any

from app.core.settings import settings
from app.pipeline.timeline_budget import (
    classify_timeline_segment_type,
    normalize_candidate_time_bounds,
    timeline_card_duration,
    timeline_segment_count_for_duration,
)
from app.pipeline.storyline_gen.candidate_scoring import CRYPTO_TERMS
from app.pipeline.storyline_gen.moment_repair import (
    find_best_subclip_window,
    select_best_moment_for_section,
)
from app.pipeline.storyline_gen.retrieval import utility_boost
from app.pipeline.storyline_gen.text_utils import (
    normalize_name,
    parse_list_field,
    truncate_text,
)


MAX_SOURCE_VIDEO_OCCURRENCES = settings.max_uses_per_source_video
MAX_SPEAKER_OCCURRENCES = settings.max_uses_per_speaker


def sort_final_video_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(c: dict[str, Any]) -> tuple:
        role = c.get("relevance_role", "")
        role_priority = 0
        if role == "exact":
            role_priority = 3
        elif role == "strong_related":
            role_priority = 2
        elif role == "weak_related":
            role_priority = 1
        return (
            role_priority,
            float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or 0),
            1 if c.get("speaker_soundbite_allowed") or c.get("speaker_clip_allowed") else 0,
            1 if c.get("speaker_visual_allowed") else 0,
            1 if c.get("use_as") in ("speaker_soundbite", "speaker_visual") else 0,
            float(c.get("selection_score") or 0),
            float(c.get("semantic_score") or 0),
            float(c.get("quote_strength") or 0),
            float(c.get("clarity_score") or 0),
        )
    return sorted(candidates, key=sort_key, reverse=True)


def apply_diversity_cap(
    candidates: list[dict[str, Any]],
    max_per_source: int = 2,
    max_per_speaker: int = 2,
    requested_speaker: str | None = None,
) -> list[dict[str, Any]]:
    source_counts: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    result = []
    for c in candidates:
        source = str(c.get("source_video") or "unknown")
        speaker = str(c.get("speaker") or "unknown")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        # Speaker cap only applies when no speaker is specifically requested
        if not requested_speaker and speaker != "unknown" and speaker_counts.get(speaker, 0) >= max_per_speaker:
            continue
        result.append(c)
        source_counts[source] = source_counts.get(source, 0) + 1
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    # Ensure at least 1 candidate survives when input is non-empty
    if not result and candidates:
        result = [candidates[0]]
    return result


def dedupe_by_fragment_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        deduped.append(candidate)
    return deduped


def estimate_section_count(target_duration_seconds: int) -> int:
    """Estimate how many narrative sections the documentary will have.

    The actual section structure is decided downstream by the director/LLM,
    but selection needs an estimate up front to size the speaker reservation.
    Mirrors the ~3-6 section role sequences used by build_director_plan.
    """
    if target_duration_seconds <= 90:
        return 3
    if target_duration_seconds <= 180:
        return 4
    if target_duration_seconds <= 300:
        return 5
    return 6


def speakers_per_section_for_duration(target_duration_seconds: int) -> int:
    """Adaptive speaker density: short videos get 1 real speaker voice per
    section (narration-driven, tighter); longer videos allow up to 2 per
    section (more speaker evidence, richer)."""
    return 1 if target_duration_seconds <= 180 else 2


def is_speaker_eligible(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
    )


def select_documentary_candidates(
    selected: list[dict[str, Any]],
    target_duration_seconds: int,
    cap: int,
    requested_speaker: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Speaker-first selection.

    Real speaker voices are structurally required for a documentary, so we
    reserve slots for the best speaker-eligible candidates BEFORE the general
    relevance cap runs — otherwise a narration-only clip that scores marginally
    higher on topic relevance crowds out a genuine speaker soundbite, which is
    exactly how earlier runs collapsed to 1-2 usable speakers despite a rich
    pool.

    Returns (selected, diagnostics).
    """
    ranked = sort_final_video_candidates(selected)

    # Enforce diversity ONCE across the whole pool (≤2 per source, ≤2 per
    # speaker) so the same talk/person can't dominate — whether it lands in the
    # reserved speaker tier or the fill tier.
    diverse = apply_diversity_cap(
        ranked,
        max_per_source=2,
        max_per_speaker=2,
        requested_speaker=requested_speaker,
    )

    n_sections = estimate_section_count(target_duration_seconds)
    per_section = speakers_per_section_for_duration(target_duration_seconds)
    # Reserve generously (up to 2x the structural need) so the duration
    # allocation in Phase B has speaker overflow to enrich sections with —
    # never below the cap, though.
    speaker_target = min(cap, n_sections * per_section * 2)

    speaker_pool = [c for c in diverse if is_speaker_eligible(c)]
    reserved_speakers = speaker_pool[:speaker_target]
    reserved_ids = {str(c.get("fragment_id")) for c in reserved_speakers}

    remaining = [c for c in diverse if str(c.get("fragment_id")) not in reserved_ids]
    fill_slots = max(0, cap - len(reserved_speakers))
    filler = remaining[:fill_slots]

    combined = sort_final_video_candidates(reserved_speakers + filler)

    diagnostics = {
        "speaker_eligible_in_pool": len(speaker_pool),
        "speaker_slots_reserved": len(reserved_speakers),
        "speaker_target": speaker_target,
        "estimated_sections": n_sections,
        "speakers_per_section": per_section,
        "fill_slots_used": len(filler),
        "final_selected": len(combined),
    }
    return combined, diagnostics


# Duration floor helpers (probe-clamp + fill-to-target) live in the shared
# app.pipeline.timeline_budget module — imported at the top of this file.


def build_export_timeline_from_candidates(
    selected_candidates: list[dict],
    prompt_profile: dict,
    target_duration_seconds: int,
    title: str | None = None,
    video_mode: str | None = None,
) -> list[dict]:
    safe_candidates = [
        dict(candidate)
        for candidate in selected_candidates
        if candidate.get("final_video_allowed") is True and candidate.get("fragment_id")
    ]
    safe_candidates = dedupe_by_fragment_id(safe_candidates)
    if not safe_candidates:
        return []

    if video_mode == "chronological_relay":
        ranked_candidates = sorted(
            safe_candidates,
            key=lambda c: (
                c.get("video_year") is None,
                c.get("video_year") or 9999,
                -float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or 0.0),
            ),
        )
    else:
        def sort_key(candidate: dict[str, Any]) -> tuple:
            segment_type = classify_timeline_segment_type(candidate)
            type_priority = {
                "speaker_clip": 4,
                "speaker_visual": 3,
                "archive_visual": 2,
                "narration_broll": 1,
            }.get(segment_type, 0)
            role_priority = {
                "exact": 3,
                "strong_related": 2,
                "weak_related": 1,
            }.get(str(candidate.get("relevance_role") or ""), 0)
            return (
                type_priority,
                role_priority,
                float(candidate.get("final_prompt_score") or candidate.get("prompt_relevance_score") or 0.0),
                float(candidate.get("selection_score") or 0.0),
                float(candidate.get("semantic_score") or 0.0),
            )

        ranked_candidates = sorted(safe_candidates, key=sort_key, reverse=True)

    desired_segment_count = min(timeline_segment_count_for_duration(target_duration_seconds), len(ranked_candidates))
    chosen: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    if video_mode == "chronological_relay":
        # Per-year cap: prevent one over-represented year from filling all slots.
        # Each distinct year gets at most ceil(desired / distinct_year_count) slots
        # in the first pass, then backfill relaxes the year cap (source cap stays).
        distinct_years = sorted(set(
            c.get("video_year") for c in ranked_candidates
            if c.get("video_year") is not None
        ))
        n_years = max(1, len(distinct_years))
        # ceiling division — every year gets at least 1 guaranteed slot
        year_cap = max(2, -(-desired_segment_count // n_years))
        year_counts: dict[int, int] = {}

        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            source_video = str(candidate.get("source_video") or "unknown")
            year = candidate.get("video_year")
            if source_counts.get(source_video, 0) >= 2:
                continue
            if year is not None and year_counts.get(year, 0) >= year_cap:
                continue
            chosen.append(candidate)
            source_counts[source_video] = source_counts.get(source_video, 0) + 1
            if year is not None:
                year_counts[year] = year_counts.get(year, 0) + 1

        # Backfill: relax year cap, keep source cap
        if len(chosen) < desired_segment_count:
            chosen_ids = {str(item.get("fragment_id")) for item in chosen}
            for candidate in ranked_candidates:
                if len(chosen) >= desired_segment_count:
                    break
                fragment_id = str(candidate.get("fragment_id") or "")
                if fragment_id in chosen_ids:
                    continue
                source_video = str(candidate.get("source_video") or "unknown")
                if source_counts.get(source_video, 0) >= 2:
                    continue
                chosen.append(candidate)
                chosen_ids.add(fragment_id)
                source_counts[source_video] = source_counts.get(source_video, 0) + 1
    else:
        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            source_video = str(candidate.get("source_video") or "unknown")
            if source_counts.get(source_video, 0) >= 2:
                continue
            chosen.append(candidate)
            source_counts[source_video] = source_counts.get(source_video, 0) + 1

    if len(chosen) < desired_segment_count:
        chosen_ids = {str(item.get("fragment_id")) for item in chosen}
        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            fragment_id = str(candidate.get("fragment_id") or "")
            if fragment_id in chosen_ids:
                continue
            chosen.append(candidate)
            chosen_ids.add(fragment_id)

    if not chosen:
        return []

    card_duration = timeline_card_duration(target_duration_seconds)
    content_budget = max(1, target_duration_seconds - (card_duration * 2))
    per_segment_duration = max(6.0, round(content_budget / max(1, len(chosen)), 2))
    timeline: list[dict[str, Any]] = [
        {
            "type": "title_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    ]

    used_fragment_ids: set[str] = set()
    section_roles = [
        "hook", "human_context", "context", "tension",
        "problem_or_tension", "expert_evidence", "solution_direction", "final_synthesis",
    ]
    _last_year_stamp: int | None = object()  # sentinel — never equals any real year

    for idx, candidate in enumerate(chosen):
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id:
            continue

        if video_mode == "chronological_relay":
            candidate_year = candidate.get("video_year")
            if candidate_year is not None and candidate_year != _last_year_stamp:
                timeline.append({
                    "type": "year_stamp",
                    "year": candidate_year,
                    "duration": 2.5,
                    "audio_mode": "silent",
                    "muted": True,
                    "section_role": "temporal_transition",
                })
                _last_year_stamp = candidate_year

        section_role = section_roles[idx % len(section_roles)]
        dummy_section = {"section_type": section_role, "role": section_role}

        moment = select_best_moment_for_section(
            section=dummy_section,
            candidates=[candidate],
            prompt_profile=prompt_profile,
            used_fragment_ids=used_fragment_ids,
            section_role=section_role,
            window_seconds=per_segment_duration,
        )

        if moment is None:
            # Fallback: use legacy subclip window logic
            start_value, end_value, duration_value = normalize_candidate_time_bounds(candidate)
            clip_duration = duration_value if duration_value is not None else per_segment_duration
            clip_duration = max(1.0, min(per_segment_duration, clip_duration))
            if start_value is not None:
                end_value = start_value + clip_duration if end_value is None else min(end_value, start_value + clip_duration)
                clip_duration = max(0.1, end_value - start_value)
            window_candidate = dict(candidate)
            if start_value is not None:
                window_candidate["start"] = start_value
            if end_value is not None:
                window_candidate["end"] = end_value
            legacy_window = find_best_subclip_window(window_candidate, prompt_profile, desired_duration=clip_duration)
            if legacy_window.get("adjusted"):
                start_value = legacy_window["start"]
                end_value = legacy_window["end"]
                clip_duration = max(0.1, end_value - start_value)
            legacy_seg_type = classify_timeline_segment_type(candidate)
            segment = {
                "type": legacy_seg_type,
                "segment_type": legacy_seg_type,
                "audio_mode": "source_audio" if legacy_seg_type == "speaker_clip" else "silent",
                "fragment_id": fragment_id,
                "source_video": candidate.get("source_video"),
                "video_year": candidate.get("video_year"),
                "video_upload_date": candidate.get("video_upload_date"),
                "speaker": candidate.get("speaker"),
                "speaker_attribution": candidate.get("speaker_attribution"),
                "start": round(float(start_value), 3) if start_value is not None else None,
                "end": round(float(end_value), 3) if end_value is not None else None,
                "duration": round(float(clip_duration), 3),
                "use_as": candidate.get("use_as"),
                "relevance_role": candidate.get("relevance_role"),
                "selection_reason": candidate.get("selection_reason"),
                "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
                "subclip_adjusted": legacy_window.get("adjusted", False),
                "moment_text": "",
                "moment_summary": "",
                "moment_selection_reason": "legacy_fallback_no_moment_selected",
                "section_role": section_role,
                "muted": not bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "subtitle_text": "",
                "subtitle_kind": "none",
                "speech_quality": {},
                "moment_scores": {},
            }
        else:
            used_fragment_ids.add(fragment_id)
            seg_type = moment.get("segment_type") or moment["type"]
            audio_mode = moment.get("audio_mode") or (
                "source_audio" if seg_type == "speaker_clip" else "silent"
            )
            segment = {
                "type": seg_type,
                "segment_type": seg_type,
                "audio_mode": audio_mode,
                "fragment_id": fragment_id,
                "source_video": moment.get("source_video") or candidate.get("source_video"),
                "video_year": candidate.get("video_year"),
                "video_upload_date": candidate.get("video_upload_date"),
                "speaker": moment.get("speaker") or candidate.get("speaker"),
                "speaker_attribution": moment.get("speaker_attribution") or candidate.get("speaker_attribution"),
                "start": moment["start"],
                "end": moment["end"],
                "duration": moment["duration"],
                "use_as": moment.get("use_as") or candidate.get("use_as"),
                "relevance_role": moment.get("relevance_role") or candidate.get("relevance_role"),
                "selection_reason": moment.get("selection_reason") or candidate.get("selection_reason"),
                "speaker_clip_allowed": moment.get("speaker_clip_allowed", False),
                "speaker_visual_allowed": moment.get("speaker_visual_allowed", False),
                "subclip_adjusted": moment.get("subclip_adjusted", False),
                "moment_text": moment.get("moment_text", ""),
                "moment_summary": moment.get("moment_summary", ""),
                "moment_selection_reason": moment.get("moment_selection_reason", ""),
                "section_role": moment.get("section_role", section_role),
                "muted": moment.get("muted", True),
                "subtitle_text": moment.get("subtitle_text", ""),
                "subtitle_kind": moment.get("subtitle_kind", "none"),
                "speech_quality": moment.get("speech_quality", {}),
                "moment_scores": moment.get("moment_scores", {}),
            }
        timeline.append(segment)

    timeline.append(
        {
            "type": "closing_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    )
    return timeline


def build_evidence_gap_notes(
    prompt_intent: dict[str, Any],
    prompt_terms: set[str],
    evidence_metrics: dict[str, Any],
) -> list[str]:
    notes = []
    exact = int(evidence_metrics.get("exact_clip_count") or 0)
    usable_speaker = int(evidence_metrics.get("usable_speaker_clip_count") or 0)
    requested_speaker = prompt_intent.get("requested_speaker")

    WOMEN_TERMS = {"women", "woman", "female", "girls", "empowerment", "empower"}
    CRYPTO_TERMS_SET = {"crypto", "cryptocurrency", "blockchain", "web3"}

    if exact == 0:
        notes.append(
            "No direct evidence clips were found for this prompt. "
            "The video should be narration-led and should not present adjacent material as direct evidence."
        )

    if exact == 0 and prompt_terms & WOMEN_TERMS:
        notes.append(
            "No direct women/empowerment clips were found. "
            "Adjacent gender, diversity, or inclusion clips may be used only as contextual support."
        )

    if exact == 0 and prompt_terms & CRYPTO_TERMS_SET:
        notes.append(
            "No direct cryptocurrency/blockchain evidence was found. "
            "Do not use cryptography or general AI governance clips as direct crypto evidence."
        )

    if int(evidence_metrics.get("context_promoted_count") or 0) > 0:
        notes.append(
            "Some included clips matched only on general topic similarity, not specific keywords — "
            "treat as background context, not direct evidence."
        )

    usable_speaker_visual = int(evidence_metrics.get("usable_speaker_visual_count") or 0)
    total_speaker_usable = usable_speaker + usable_speaker_visual
    if requested_speaker and total_speaker_usable <= 2:
        count_label = "No" if total_speaker_usable == 0 else f"Only {total_speaker_usable}"
        notes.append(
            f"{count_label} usable speaker clip(s) or visual(s) found for {requested_speaker}. "
            f"Do not use unrelated speakers as filler."
        )

    return notes


def build_storyline_evidence_package(profile: dict, buckets: dict) -> dict:
    selected = buckets.get("selected_candidates", [])
    supporting = buckets.get("supporting_candidates", [])
    exact_selected = [item for item in selected if item.get("relevance_role") == "exact"]

    evidence_gaps: list[str] = []
    must_not_claim: list[str] = []

    topic_terms = set(profile.get("topic_terms") or [])
    if profile.get("requested_speaker") and not exact_selected:
        evidence_gaps.append(f"Only limited archive evidence was found for {profile.get('requested_speaker')}.")
    elif not exact_selected:
        evidence_gaps.append("No direct evidence clips were found for this prompt.")
    if topic_terms & {"women", "woman", "empowerment", "empower"} and not exact_selected:
        evidence_gaps.append("No direct women/empowerment clips were found.")
        must_not_claim.append("Do not present generic inclusion/accessibility clips as direct women empowerment evidence.")
    if "crypto" in topic_terms and not exact_selected:
        evidence_gaps.append("No direct cryptocurrency/blockchain evidence was found.")
        must_not_claim.append("Do not treat cryptography or quantum encryption as cryptocurrency evidence.")
    elif "crypto" in topic_terms and not any(
        any(normalize_name(term) in {normalize_name(v) for v in CRYPTO_TERMS} for term in (item.get("direct_evidence_terms") or []) + (item.get("expanded_evidence_terms") or []))
        for item in selected
    ):
        must_not_claim.append("Do not treat cryptography or quantum encryption as cryptocurrency evidence.")
    if profile.get("requested_speaker"):
        must_not_claim.append("Do not use unrelated speakers as filler.")
    if not topic_terms and any(item.get("semantic_only_match") for item in selected):
        must_not_claim.append(
            "Some included clips were selected on general topical/semantic similarity, not an exact "
            "match to the prompt's wording — narrate them as broadly relevant perspectives, not as "
            "direct answers to the specific question asked."
        )

    return {
        "prompt_profile": profile,
        "selected_evidence": build_candidate_pack(selected),
        "supporting_context": build_candidate_pack(supporting[:10]),
        "evidence_gaps": evidence_gaps,
        "must_not_claim": must_not_claim,
    }


def _assign_relay_sections_by_year(
    recommended_structure: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For chronological_relay mode: pre-assign real fragments to each recommended
    section grouped by year so the LLM receives concrete year anchors.

    Each content section gets:
      - year_slot: the representative year for that section
      - speaker_soundbites / archive_visual_fragments: 1-2 best candidates from
        that year (for the LLM to reference and narrate around)

    Years are distributed across sections in chronological order so older years
    appear earlier and newer years appear later.
    """
    if not candidates:
        return recommended_structure

    # Bucket by year; discard year=None for slot assignment
    by_year: dict[int, list[dict[str, Any]]] = {}
    for c in candidates:
        y = c.get("video_year")
        if y is not None:
            by_year.setdefault(int(y), []).append(c)

    distinct_years = sorted(by_year.keys())
    if not distinct_years:
        return recommended_structure

    # Content sections = everything except the final synthesis/narration-only section
    # which may span multiple years and shouldn't be pinned to one year.
    content_roles = [
        s for s in recommended_structure
        if s.get("section_type") != "narration" or s.get("role") not in ("temporal_context", "synthesis")
    ]
    narration_only_roles = [
        s for s in recommended_structure
        if s not in content_roles
    ]

    # Distribute years across content sections so ALL distinct years are represented.
    # When there are more years than sections, pick evenly-spaced years so the
    # earliest and latest years are always included (guaranteeing the full temporal arc).
    # When there are more sections than years, repeat years cyclically.
    year_for_section: list[int] = []
    n_sections = len(content_roles)
    n_years = len(distinct_years)
    if n_sections >= n_years:
        # More sections than years — assign years cyclically
        for i in range(n_sections):
            year_for_section.append(distinct_years[i % n_years])
    else:
        # Fewer sections than years — pick evenly-spaced years so the arc spans
        # from the earliest to the latest year (always include first and last).
        if n_sections == 1:
            year_for_section = [distinct_years[0]]
        else:
            step = (n_years - 1) / (n_sections - 1)
            year_for_section = [distinct_years[round(i * step)] for i in range(n_sections)]

    # Build enriched section copies with real fragments attached
    used_frag_ids: set[str] = set()
    enriched: list[dict[str, Any]] = []

    content_idx = 0
    for section in recommended_structure:
        sec = dict(section)
        if section in narration_only_roles:
            # Narration-only sections: assign median year as context hint but no clips
            all_years = sorted(by_year.keys())
            sec["year_slot"] = all_years[len(all_years) // 2] if all_years else None
            sec["speaker_soundbites"] = []
            sec["archive_visual_fragments"] = []
            enriched.append(sec)
            continue

        year = year_for_section[content_idx] if content_idx < len(year_for_section) else distinct_years[-1]
        content_idx += 1
        sec["year_slot"] = year

        pool = [c for c in by_year.get(year, []) if c.get("fragment_id") not in used_frag_ids]
        soundbites = []
        visuals = []
        for c in pool[:3]:
            fid = c.get("fragment_id")
            seg_type = c.get("segment_type") or c.get("type") or ""
            entry = {
                "fragment_id": fid,
                "start": c.get("start"),
                "end": c.get("end"),
                "video_year": year,
                "speaker_name": c.get("speaker_name") or c.get("speaker") or "",
                "text": (c.get("text") or "")[:200],
            }
            if seg_type in ("speaker_clip", "speaker_visual") or c.get("speaker_name"):
                soundbites.append(entry)
            else:
                visuals.append({**entry, "usage": "evidence_visual"})
            if fid:
                used_frag_ids.add(fid)
        sec["speaker_soundbites"] = soundbites[:2]
        sec["archive_visual_fragments"] = visuals[:2]
        enriched.append(sec)

    return enriched


def select_balanced_candidates(retrieved: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    for record in retrieved:
        record["selection_score"] = round(
            float(record.get("final_prompt_score") or record.get("prompt_relevance_score") or record["semantic_score"]) + utility_boost(record),
            6,
        )

    def sort_key(record: dict[str, Any]) -> tuple[int, int, float]:
        relevance_role = record.get("relevance_role")
        priority = 0
        if relevance_role == "exact":
            priority = 4
        elif relevance_role == "strong_related":
            priority = 3
        elif relevance_role == "weak_related":
            priority = 2
        elif relevance_role == "contextual":
            priority = 1
        # "reject" stays at 0

        # Among equally-relevant candidates, prefer real speaking clips over
        # silent b-roll — utility_boost's score bonus alone is too small to
        # reliably win this tie against a marginally-more-topical b-roll clip.
        if record.get("speaker_soundbite_allowed") or record.get("speaker_clip_allowed"):
            speaker_bonus = 2
        elif record.get("speaker_visual_allowed"):
            speaker_bonus = 1
        else:
            speaker_bonus = 0

        return (priority, speaker_bonus, float(record["selection_score"]))

    # Filter out rejected candidates before selection
    retrieved = [r for r in retrieved if r.get("relevance_role") != "reject"]

    sorted_records = sorted(retrieved, key=sort_key, reverse=True)

    selected = []
    requested_speaker = next((r.get("requested_speaker") for r in retrieved if r.get("requested_speaker")), None)
    used_videos = {}
    used_speakers = {}
    theme_counts = {}
    year_counts: dict[int | None, int] = {}

    MAX_PER_YEAR_BUCKET = 3

    # First pass: prefer diversity by video and theme.
    for record in sorted_records:
        if len(selected) >= limit:
            break

        source_video = record.get("source_video") or "unknown"
        theme = record.get("primary_theme") or "unknown"

        used_count = used_videos.get(source_video, 0)
        theme_count = theme_counts.get(theme, 0)
        speaker = record.get("speaker") or "unknown"
        speaker_count = used_speakers.get(speaker, 0)
        year_key = record.get("video_year")  # int or None
        year_count = year_counts.get(year_key, 0)

        if used_count >= MAX_SOURCE_VIDEO_OCCURRENCES:
            continue

        if requested_speaker and source_video == "unknown" and record.get("speaker_match") is not True:
            continue

        if theme_count >= (2 if requested_speaker else 10):
            continue

        if speaker != "unknown" and speaker_count >= MAX_SPEAKER_OCCURRENCES:
            continue

        if year_count >= MAX_PER_YEAR_BUCKET:
            continue

        selected.append(record)
        used_videos[source_video] = used_count + 1
        theme_counts[theme] = theme_count + 1
        used_speakers[speaker] = speaker_count + 1
        year_counts[year_key] = year_count + 1

    # Second pass: fill remaining slots if diversity filters were too strict.
    if len(selected) < limit:
        selected_ids = {item.get("fragment_id") for item in selected}

        for record in sorted_records:
            if len(selected) >= limit:
                break

            if record.get("fragment_id") not in selected_ids:
                selected.append(record)
                selected_ids.add(record.get("fragment_id"))

    return selected[:limit]


def build_candidate_pack(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact candidate pack for the LLM prompt.
    Only include fields the LLM needs — trimmed to keep prompt under token budget."""
    packed = []
    for item in candidates:
        cuc = parse_list_field(item.get("clip_usage_candidates"))

        packed.append({
            "fragment_id": item.get("fragment_id"),
            "source_video": item.get("source_video"),
            "video_id": item.get("video_id"),
            "start": item.get("start"),
            "end": item.get("end"),
            "duration": item.get("duration"),
            "primary_theme": item.get("primary_theme"),
            "primary_chapter": item.get("primary_chapter"),
            "summary": truncate_text(item.get("summary", ""), 150),
            "keywords": parse_list_field(item.get("keywords"))[:6],
            "clip_usage_candidates": cuc,
            "quote_strength": item.get("quote_strength"),
            "clarity_score": item.get("clarity_score"),
            "relevance_role": item.get("relevance_role"),
            "speaker": item.get("speaker"),
            "speaker_attribution": item.get("speaker_attribution"),
            "speaker_clip_allowed": item.get("speaker_clip_allowed", False),
            "speaker_soundbite_allowed": item.get("speaker_soundbite_allowed", False),
            "speaker_visual_allowed": item.get("speaker_visual_allowed", False),
            "relevance_reason": item.get("relevance_reason", ""),
            "text": truncate_text(item.get("text", ""), 200),
            "video_year": item.get("video_year"),
            "audio_quality_status": item.get("audio_quality_status"),
            "final_video_allowed": item.get("final_video_allowed", False),
        })
    return packed
