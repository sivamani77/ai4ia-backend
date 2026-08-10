from __future__ import annotations
from pathlib import Path
from typing import Any

from app.pipeline.timeline_budget import (
    classify_timeline_segment_type,
    normalize_candidate_time_bounds,
    timeline_card_duration,
    timeline_segment_count_for_duration,
)
from app.pipeline.assets_gen.utils import log
from app.pipeline.assets_gen.voiceover import reconcile_narration_duration

# Speaker clip render bounds
PRE_ROLL_SECONDS: float = 0.25
POST_ROLL_SECONDS: float = 0.35
CLIP_EXPAND_SECONDS: float = 3.0
QUOTE_COMPLETE_THRESHOLD: float = 0.80  # quote_completeness_score >= this → complete

MOMENT_FIELDS_FROM_STORYLINE = {
    "moment_text",
    "moment_summary",
    "muted",
    "subtitle_text",
    "subtitle_kind",
    "section_role",
    "moment_selection_reason",
    "speech_quality",
    "moment_scores",
    "segment_type",
    "audio_mode",
}

INTRO_ADMIN_TERMS_08 = [
    "hello everyone",
    "good morning",
    "good afternoon",
    "thank you",
    "thanks for inviting me",
    "let me introduce",
    "next slide",
    "moving on",
    "today i will",
    "my name is",
    "it is an honor",
    "housekeeping",
    "restroom",
    "closed captioning",
    "can you hear me",
    "welcome remarks",
]


def collect_archive_fragments(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    selected_candidates = storyline.get("selected_candidates", []) or []

    for item in selected_candidates:
        if not item.get("final_video_allowed") or not item.get("fragment_id"):
            continue
        record = dict(item)
        record["asset_role"] = "selected_candidate"
        fragments.append(record)

    if fragments:
        return dedupe_manifest_fragments(fragments)

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("archive_visual_fragments", []):
            record = dict(item)
            record["section_id"] = section_id
            record["asset_role"] = "archive_visual"
            fragments.append(record)

        for item in section.get("speaker_soundbites", []):
            record = dict(item)
            record["section_id"] = section_id
            record["asset_role"] = "speaker_soundbite"
            fragments.append(record)

    return dedupe_manifest_fragments(fragments)


def dedupe_manifest_fragments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fragment in fragments:
        fragment_id = str(fragment.get("fragment_id") or "").strip()
        if not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        deduped.append(fragment)
    return deduped


def build_export_timeline_from_candidates(
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
    title: str | None = None,
) -> list[dict[str, Any]]:
    safe_candidates = dedupe_manifest_fragments(
        [dict(item) for item in selected_candidates if item.get("final_video_allowed") is True and item.get("fragment_id")]
    )
    if not safe_candidates:
        return []

    def sort_key(candidate: dict[str, Any]) -> tuple:
        type_priority = {
            "speaker_clip": 4,
            "speaker_visual": 3,
            "archive_visual": 2,
            "narration_broll": 1,
        }.get(classify_timeline_segment_type(candidate), 0)
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

    ranked = sorted(safe_candidates, key=sort_key, reverse=True)
    desired = min(timeline_segment_count_for_duration(target_duration_seconds), len(ranked))
    chosen = ranked[:desired]
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
    for candidate in chosen:
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id:
            continue
        start_value, end_value, duration_value = normalize_candidate_time_bounds(candidate)
        clip_duration = duration_value if duration_value is not None else per_segment_duration
        clip_duration = max(1.0, min(per_segment_duration, clip_duration))
        if start_value is not None:
            end_value = start_value + clip_duration if end_value is None else min(end_value, start_value + clip_duration)
            clip_duration = max(0.1, end_value - start_value)
        timeline.append(
            {
                "type": classify_timeline_segment_type(candidate),
                "fragment_id": fragment_id,
                "source_video": candidate.get("source_video"),
                "start": round(float(start_value), 3) if start_value is not None else None,
                "end": round(float(end_value), 3) if end_value is not None else None,
                "duration": round(float(clip_duration), 3),
                "use_as": candidate.get("use_as"),
                "relevance_role": candidate.get("relevance_role"),
                "selection_reason": candidate.get("selection_reason"),
                "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
            }
        )
    timeline.append(
        {
            "type": "closing_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    )
    return timeline


def collect_generated_assets(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    assets = []

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("generated_assets_needed", []):
            record = dict(item)
            record["section_id"] = section_id
            assets.append(record)

    return assets


def collect_internet_assets(storyline: dict[str, Any]) -> list[dict[str, Any]]:
    assets = []

    for section in storyline.get("sections", []):
        section_id = section.get("section_id")

        for item in section.get("internet_assets_needed", []):
            record = dict(item)
            record["section_id"] = section_id
            assets.append(record)

    return assets


def enrich_timeline_with_moment_fields(
    export_timeline: list[dict[str, Any]],
    storyline_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enrich asset manifest export_timeline with moment-level fields from the
    storyline's export_timeline. Existing moment fields are NOT overwritten.
    """
    # Build lookup by fragment_id from storyline timeline
    storyline_by_fid: dict[str, dict[str, Any]] = {}
    for item in storyline_timeline:
        fid = item.get("fragment_id")
        if fid:
            storyline_by_fid[str(fid)] = item

    enriched = []
    for item in export_timeline:
        item = dict(item)
        fid = item.get("fragment_id")
        if fid:
            source = storyline_by_fid.get(str(fid), {})
            for field in MOMENT_FIELDS_FROM_STORYLINE:
                if field not in item and field in source:
                    item[field] = source[field]

        # Phase 4Q: speaker_clip items must never receive AI voiceover
        seg_type = item.get("segment_type") or item.get("type") or ""
        if seg_type == "speaker_clip":
            item["voiceover_text"] = ""
            item["voiceover_path"] = None
            item["audio_mode"] = "source_audio"
            item["muted"] = False
            # subtitle_text must be moment_text, not narration
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("moment_text", "")

        enriched.append(item)
    return enriched


def _detect_intro_admin_in_moment(text: str) -> bool:
    """Return True if the moment text is dominated by intro/admin language."""
    if not text:
        return False
    normalized = text.lower()
    count = sum(1 for term in INTRO_ADMIN_TERMS_08 if term in normalized)
    return count >= 2


def _text_ends_complete(text: str) -> bool:
    """Return True if text ends with terminal punctuation."""
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?"


def _text_starts_mid_sentence(text: str) -> bool:
    """Return True if text likely starts mid-sentence (first char is lowercase)."""
    t = (text or "").lstrip()
    return bool(t) and t[0].islower()


def compute_quote_render_bounds(
    item: dict[str, Any],
    cand: dict[str, Any],
    fragment_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute render_start, render_end, render_duration, quote_complete for a speaker clip.

    Uses the full fragment window (start → end) + padding as the baseline.
    Attempts forward expansion using the source fragment boundary if the quote is incomplete.
    Only demotes if quote_completeness < 0.20 AND incomplete_sentence_penalty > 0.30.

    Never shrinks the validated window: the render window may only pad or expand the
    selected timeframe. Content trimming is 07b validation's job alone.
    """
    # A window already set by 07b validation is a locked contract — pass it through.
    locked_start = item.get("render_start")
    locked_end = item.get("render_end")
    if locked_start is not None and locked_end is not None and float(locked_end) > float(locked_start):
        return {
            "render_start": round(float(locked_start), 3),
            "render_end": round(float(locked_end), 3),
            "render_duration": round(float(locked_end) - float(locked_start), 3),
            "quote_complete": bool(item.get("quote_complete", True)),
            "quote_boundary_method": "locked_by_validation",
            "quote_expanded": False,
            "should_demote": False,
        }

    start = float(item.get("start") or 0.0)
    end = float(item.get("end") or 0.0)
    moment_text = (item.get("moment_text") or cand.get("moment_text") or "").strip()
    moment_scores = item.get("moment_scores") or cand.get("moment_scores") or {}

    completeness = float(moment_scores.get("quote_completeness_score") or 0.0)
    penalty = float(moment_scores.get("incomplete_sentence_penalty") or 0.0)

    text_complete = _text_ends_complete(moment_text)
    text_mid_start = _text_starts_mid_sentence(moment_text)

    is_complete = text_complete or (completeness >= QUOTE_COMPLETE_THRESHOLD and penalty < 0.05)

    # Start with the full fragment window + padding
    render_start = max(0.0, start - PRE_ROLL_SECONDS)
    render_end = end + POST_ROLL_SECONDS
    method = "complete_sentence" if is_complete else "padding_only"
    expanded = False

    # Try to expand an incomplete quote using the source fragment boundary
    if not is_complete and fragment_lookup:
        fid = str(item.get("fragment_id") or "")
        cf = fragment_lookup.get(fid) or {}
        raw_source_end = cf.get("end")
        if raw_source_end is not None:
            source_end = float(raw_source_end)
            available = source_end - end
            if available > 1.5:
                expand_by = min(available - 0.1, CLIP_EXPAND_SECONDS)
                if expand_by > 0.5:
                    render_end = end + expand_by + POST_ROLL_SECONDS
                    expanded = True
                    method = "sentence_or_segment_expansion"
                else:
                    method = "at_fragment_boundary_no_expansion"
            else:
                method = "at_fragment_boundary_no_expansion"

    # The full validated window always renders — no hard-max shrink, no artificial
    # min-duration stretch into unvalidated footage.
    render_duration = render_end - render_start

    # Demote only if quote is truly unusable (very low completeness + very high penalty)
    should_demote = (
        not text_complete
        and completeness < 0.20
        and penalty > 0.30
    )

    return {
        "render_start": round(render_start, 3),
        "render_end": round(render_end, 3),
        "render_duration": round(render_duration, 3),
        "quote_complete": is_complete,
        "quote_boundary_method": method,
        "quote_expanded": expanded,
        "should_demote": should_demote,
    }


def _speaker_clip_quality_score(item: dict[str, Any]) -> float:
    """Score a timeline fragment item for speaker clip suitability (0.0–1.0)."""
    sq = item.get("speech_quality") or {}
    ms = item.get("moment_scores") or {}
    is_usable = sq.get("is_usable_speech", True)
    speech_ratio = float(sq.get("speech_ratio") or 0.0)
    final_score = float(ms.get("final_moment_score") or 0.0)
    base = speech_ratio * 0.4 + final_score * 0.6
    # If speech quality heuristic says not usable but the candidate was explicitly approved
    # as a speaker soundbite (speaker_soundbite_allowed / speaker_clip_allowed), use a floor
    # score so the annotation decision is not overridden by the heuristic alone.
    if not is_usable:
        explicitly_approved = item.get("speaker_soundbite_allowed") or item.get("speaker_clip_allowed")
        if explicitly_approved:
            return max(base, 0.30)
        return 0.0
    return base


def build_hybrid_export_timeline(
    export_timeline: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    section_voiceover_paths: list[str],
    max_speaker_clips: int = 5,
    speaker_clip_score_threshold: float = 0.20,
    fragment_lookup: dict[str, dict[str, Any]] | None = None,
    vo_path_to_text: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a hybrid timeline mixing speaker_clip/source_audio and narration_broll/ai_voiceover.

    Algorithm:
    1. Non-card fragment items are scored for speaker clip suitability.
    2. Top-N (up to max_speaker_clips) items that pass threshold become speaker_clip.
    3. Speaker clips receive render_start/render_end/render_duration from compute_quote_render_bounds().
    4. Remaining fragment items become narration_broll, each receiving an audio_path
       from section_voiceover_paths (cycling if needed), with subtitle_text set from vo_path_to_text.
    5. All type fields (type, asset_type, kind, segment_type) are normalised.
    """
    candidate_by_fid: dict[str, dict[str, Any]] = {}
    for cand in (selected_candidates or []):
        fid = cand.get("fragment_id")
        if fid:
            candidate_by_fid[str(fid)] = cand

    # Collect eligible fragment items (non-card, has fragment_id, valid duration)
    card_types = {"title_card", "closing_card"}
    fragment_items: list[tuple[int, dict[str, Any]]] = []  # (original_index, item)
    card_items: list[tuple[int, dict[str, Any]]] = []

    for i, raw_item in enumerate(export_timeline):
        item = dict(raw_item)
        t = item.get("type") or item.get("segment_type") or ""
        if t in card_types:
            card_items.append((i, item))
        elif item.get("fragment_id"):
            fragment_items.append((i, item))
        else:
            card_items.append((i, item))  # treat non-fragment non-card as pass-through

    # Score eligible fragment items.
    # Items already typed as "speaker_clip" by 07b validation are pre-approved — don't re-score them.
    scored = []
    pre_approved_speaker_fids: set[str] = set()

    for orig_idx, item in fragment_items:
        fid = item.get("fragment_id") or ""
        cand = candidate_by_fid.get(str(fid), {})
        pre_type = item.get("type") or item.get("segment_type") or ""

        # 07b / Stage 07 approved this as a speaker clip or soundbite — honour it, bypass re-scoring
        is_storyline_soundbite = (
            pre_type == "speaker_clip"
            or cand.get("use_as") == "speaker_soundbite"
            or cand.get("speaker_soundbite_allowed") is True
            or item.get("use_as") == "speaker_soundbite"
        )
        if is_storyline_soundbite and fid:
            pre_approved_speaker_fids.add(str(fid))
            scored.append((orig_idx, item, cand, 1.0, True))
            continue

        start = item.get("start")
        end = item.get("end")
        source_video = item.get("source_video") or cand.get("source_video")
        role = item.get("relevance_role") or cand.get("relevance_role") or ""

        # Basic eligibility for speaker clip
        eligible = (
            source_video is not None
            and start is not None
            and end is not None
            and role in ("exact", "strong_related", "weak_related")
        )
        try:
            dur = float(item.get("duration") or 0.0) or (float(end or 0) - float(start or 0))
            if not (4.0 <= dur <= 90.0):
                eligible = False
        except (TypeError, ValueError):
            eligible = False

        score = _speaker_clip_quality_score(item) if eligible else 0.0
        scored.append((orig_idx, item, cand, score, eligible))

    # Select top-N as speaker clips (pre-filter: remove those flagged for demotion by bounds check)
    # Pre-approved items (from 07/07b) are always included first, then fill remaining slots from scored.
    eligible_only = [(i, item, cand, sc) for i, item, cand, sc, elig in scored if elig]
    eligible_only.sort(key=lambda x: (
        0 if str(x[1].get("fragment_id") or "") in pre_approved_speaker_fids else 1,
        -x[3]
    ))

    # Pre-compute bounds for all eligible items to check should_demote
    bounds_cache: dict[int, dict[str, Any]] = {}
    for i, item, cand, sc in eligible_only:
        b = compute_quote_render_bounds(item, cand, fragment_lookup)
        bounds_cache[i] = b

    speaker_clip_fids: set[str] = set()
    selected_for_speaker = []
    for i, item, cand, sc in eligible_only:
        fid = str(item.get("fragment_id") or "")
        is_pre_approved = fid in pre_approved_speaker_fids
        # Pre-approved items (07/07b validated) bypass score and demotion thresholds
        if not is_pre_approved:
            if sc < speaker_clip_score_threshold:
                continue
            if len(selected_for_speaker) >= max_speaker_clips:
                break
            b = bounds_cache.get(i, {})
            if b.get("should_demote"):
                log(f"  [hybrid] Demoting {item.get('fragment_id', '')[-40:]!r}: quote too incomplete")
                continue
        speaker_clip_fids.add(fid)
        selected_for_speaker.append(i)

    # Guaranteed speaker clip allocation floor:
    # Ensure at least up to 3 speaker clip slots are filled from available scored fragments
    if len(selected_for_speaker) < 3 and scored:
        remaining_candidates = [
            cand_tuple for cand_tuple in scored
            if str(cand_tuple[1].get("fragment_id") or "") not in speaker_clip_fids
            and (cand_tuple[2].get("source_video") or cand_tuple[1].get("source_video"))
        ]
        remaining_candidates.sort(key=lambda x: -x[3])
        target_fill = min(3 - len(selected_for_speaker), len(remaining_candidates))
        for idx_fill in range(target_fill):
            b_i, b_item, b_cand, _, _ = remaining_candidates[idx_fill]
            b_fid = str(b_item.get("fragment_id") or "")
            if b_fid:
                speaker_clip_fids.add(b_fid)
                selected_for_speaker.append(b_i)
                log(f"  [hybrid-repair] Guaranteed speaker allocation floor applied: promoted fragment {b_fid[-40:]!r} as speaker_clip")

    # Build voiceover cycling iterator for narration_broll items
    valid_vo_paths = [p for p in (section_voiceover_paths or []) if p and Path(p).exists()]
    vo_cycle_idx = 0

    # Diagnostics counters
    stat_complete = 0
    stat_expanded = 0
    stat_demoted = 0
    stat_zero_dur = 0

    # Rebuild timeline preserving original order
    result_by_idx: dict[int, dict[str, Any]] = {}

    for orig_idx, item, cand, score, eligible in scored:
        item = dict(item)
        fid = str(item.get("fragment_id") or "")

        if fid in speaker_clip_fids:
            # Promote to speaker_clip with computed render bounds
            bounds = bounds_cache.get(orig_idx) or compute_quote_render_bounds(item, cand, fragment_lookup)
            item["type"] = "speaker_clip"
            item["asset_type"] = "speaker_clip"
            item["kind"] = "speaker_clip"
            item["segment_type"] = "speaker_clip"
            item["audio_mode"] = "source_audio"
            item["muted"] = False
            item["voiceover_text"] = ""
            item["voiceover_path"] = None
            item["audio_path"] = None
            # Apply render bounds — keep original start/end for provenance
            item["render_start"] = bounds["render_start"]
            item["render_end"] = bounds["render_end"]
            item["render_duration"] = bounds["render_duration"]
            item["quote_complete"] = bounds["quote_complete"]
            item["quote_boundary_method"] = bounds["quote_boundary_method"]
            # Use render_duration as the operative duration
            item["duration"] = bounds["render_duration"]
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("moment_text") or ""
            # Carry over cleaned corpus metadata
            for field in (
                "source_transcript_version", "audio_quality_status",
                "transcript_quality_score", "word_timing_available",
                "text_hash", "text_truncated_for_embedding",
            ):
                if not item.get(field) and cand.get(field) is not None:
                    item[field] = cand[field]
            if bounds["quote_complete"]:
                stat_complete += 1
            if bounds.get("quote_expanded"):
                stat_expanded += 1
            log(
                f"  [hybrid] speaker_clip {fid[-44:]!r}: "
                f"render {bounds['render_start']:.2f}→{bounds['render_end']:.2f}s "
                f"({bounds['render_duration']:.2f}s) complete={bounds['quote_complete']} "
                f"method={bounds['quote_boundary_method']}"
            )
        else:
            # Demote to narration_broll
            item["type"] = "narration_broll"
            item["asset_type"] = "narration_broll"
            item["kind"] = "narration_broll"
            item["segment_type"] = "narration_broll"
            item["audio_mode"] = "ai_voiceover"
            item["muted"] = True
            b = bounds_cache.get(orig_idx, {})
            if b.get("should_demote"):
                stat_demoted += 1
            # Assign section voiceover path once (do not cycle duplicate voiceovers)
            if vo_cycle_idx < len(valid_vo_paths):
                vo_path = valid_vo_paths[vo_cycle_idx]
                item["audio_path"] = vo_path
                item["voiceover_path"] = vo_path
                # Set subtitle/voiceover text from path→text map
                if vo_path_to_text:
                    vo_text = vo_path_to_text.get(vo_path, "")
                    item["subtitle_text"] = vo_text
                    item["voiceover_text"] = vo_text
                    item["subtitle_kind"] = "voiceover"
                reconcile_narration_duration(item, item.get("audio_path"))
                vo_cycle_idx += 1
            else:
                item["audio_path"] = None
                item["voiceover_path"] = None
                item["audio_mode"] = "muted"
                item["muted"] = True
                cand_dur = float(cand.get("duration") or (float(cand.get("end") or 0) - float(cand.get("start") or 0))) if cand else 0.0
                dur = max(cand_dur if cand_dur > 1.0 else 5.0, 4.0)
                item["duration"] = dur
                item["render_duration"] = dur
            if float(item.get("duration") or 0.0) == 0.0:
                stat_zero_dur += 1

        result_by_idx[orig_idx] = item

    # Pass-through for cards and other items
    for orig_idx, item in card_items:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t in card_types:
            item["type"] = t
            item["asset_type"] = t
            item["kind"] = t
            item["segment_type"] = t
        result_by_idx[orig_idx] = item

    # Reconstruct in original order
    result = [result_by_idx[i] for i in range(len(export_timeline)) if i in result_by_idx]

    # Print diagnostics
    sc_count = sum(1 for x in result if (x.get("type") or x.get("segment_type")) == "speaker_clip")
    nb_count = sum(1 for x in result if (x.get("type") or x.get("segment_type")) == "narration_broll")
    log("")
    log("Hybrid timeline quote timing summary:")
    log(f"  speaker clips:           {sc_count}")
    log(f"  complete quotes:         {stat_complete}")
    log(f"  expanded quotes:         {stat_expanded}")
    log(f"  demoted incomplete:      {stat_demoted}")
    log(f"  narration_broll:         {nb_count}")
    log(f"  zero-duration items:     {stat_zero_dur}")
    log("")
    for x in result:
        st = x.get("segment_type") or x.get("type") or ""
        if st != "speaker_clip":
            continue
        fid = x.get("fragment_id", "")[-50:]
        log(
            f"  SC {fid!r}: "
            f"src={x.get('start'):.2f}→{x.get('end'):.2f} "
            f"render={x.get('render_start'):.2f}→{x.get('render_end'):.2f} "
            f"({x.get('render_duration'):.2f}s) "
            f"complete={x.get('quote_complete')} "
            f"speech_ratio={x.get('speech_quality', {}).get('speech_ratio', '?')} "
            f"score={x.get('moment_scores', {}).get('final_moment_score', '?')}"
        )

    return result
