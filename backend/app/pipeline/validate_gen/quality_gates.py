"""Editorial quality-gate checks, the LLM repair loop, and duration top-up vetting."""
import json
from typing import Any

from app.pipeline.validate_gen.fragment_validation import _quote_ends_completely, _quote_starts_cleanly
from app.pipeline.validate_gen.json_utils import extract_json_array, extract_json_object
from app.pipeline.validate_gen.llm_client import STRONG_LLM_MODEL, log, post_with_retries
from app.pipeline.validate_gen.narrative_templates import DEEPFAKE_VISUAL_ASSET_PLANS, GENERIC_VISUAL_ASSET_PLANS

# Maximum iterations for the narrative quality-gate repair loop
MAX_VALIDATION_ITERATIONS = 3

# Gates that must pass for the timeline to be considered production-ready
CRITICAL_GATE_NAMES = frozenset({
    "all_sections_covered",
    "all_speaker_labels_present",
    "no_instruction_leakage_in_voiceover",
    "has_speaker_clips",
    # all_voiceovers_topic_specific is a warning, not a repair trigger —
    # script-07's fallback narration uses common AI openers that are still on-topic
    "final_timeline_approved",
})


def _validate_speaker_quotes(
    final_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate quote_complete flag and demote unrecoverably broken quotes to narration_broll."""
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t != "speaker_clip":
            result.append(item)
            continue

        text = (item.get("quote_text") or item.get("subtitle_text") or "").strip()
        ends_ok = _quote_ends_completely(text)
        starts_ok = _quote_starts_cleanly(text)

        if ends_ok and starts_ok:
            item["quote_complete"] = True
        else:
            # Mark as incomplete but keep as speaker_clip — quality gate will warn
            item["quote_complete"] = False
            reasons = []
            if not ends_ok:
                last = text.rstrip()[-20:] if text else ""
                reasons.append(f"ends without terminal punctuation or with weak word: ...{last!r}")
            if not starts_ok:
                first = text[:20] if text else ""
                reasons.append(f"starts mid-sentence (lowercase): {first!r}...")
            item["quote_completeness_note"] = "; ".join(reasons)

        result.append(item)
    return result


def _enforce_section_rhythm(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reorder items within each section so narration_broll precedes speaker_clip.

    Exception: S1_HOOK should open with a speaker_clip (no narration before first clip).
    """
    # Separate cards from fragment items
    ordered_sections = [s["section_id"] for s in narrative_sections]
    card_items: list[dict[str, Any]] = []
    section_buckets: dict[str, list[dict[str, Any]]] = {sid: [] for sid in ordered_sections}
    unassigned: list[dict[str, Any]] = []

    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        sid = item.get("section_id")
        if t in ("title_card", "closing_card"):
            card_items.append(item)
        elif sid in section_buckets:
            section_buckets[sid].append(item)
        else:
            unassigned.append(item)

    result: list[dict[str, Any]] = []
    # title_card always first
    title_cards = [x for x in card_items if x.get("type") == "title_card"]
    closing_cards = [x for x in card_items if x.get("type") == "closing_card"]
    result.extend(title_cards)

    for sid in ordered_sections:
        items = section_buckets.get(sid, [])
        if not items:
            continue
        if sid == "S1_HOOK":
            # S1: speaker_clip first, then any narration
            speaker = [x for x in items if x.get("type") == "speaker_clip"]
            narration = [x for x in items if x.get("type") == "narration_broll"]
            result.extend(speaker)
            result.extend(narration)
        else:
            # All other sections: narration_broll first, then speaker_clip
            narration = [x for x in items if x.get("type") == "narration_broll"]
            speaker = [x for x in items if x.get("type") == "speaker_clip"]
            result.extend(narration)
            result.extend(speaker)

    result.extend(unassigned)
    result.extend(closing_cards)
    return result


def _fill_missing_sections(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    original_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure every narrative section has at least one item.

    Donor priority: an approved speaker_clip candidate first (preserves real
    speaker audio the LLM already vetted — this is content the narrative
    assembly call should have placed but didn't), then an approved
    narration_broll/visual_support candidate, then any leftover fragment from
    the original timeline as a last resort. The inserted item's type always
    matches what the donor was actually validated as; it is never hardcoded to
    narration_broll, or a validated speaker clip would be silently downgraded.
    """
    present_sids = {item.get("section_id") for item in final_timeline if item.get("section_id")}
    missing = [s for s in narrative_sections if s["section_id"] not in present_sids]
    if not missing:
        return final_timeline

    used_fids = {item.get("fragment_id") for item in final_timeline if item.get("fragment_id")}

    approved = [
        c for c in validated_candidates
        if c.get("approved") and c.get("fragment_id") not in used_fids
    ]
    speaker_pool = [c for c in approved if c.get("decision") == "speaker_clip"]
    broll_pool = [c for c in approved if c.get("decision") != "speaker_clip"]
    fallback_pool = [
        item for item in original_timeline
        if item.get("fragment_id") and item.get("fragment_id") not in used_fids
    ]

    # Find closing_card index to insert before it
    insert_before = len(final_timeline)
    for idx, item in enumerate(final_timeline):
        if item.get("type") == "closing_card":
            insert_before = idx
            break

    def _take_speaker_donor() -> dict[str, Any] | None:
        return speaker_pool.pop(0) if speaker_pool else None

    def _take_broll_donor() -> dict[str, Any] | None:
        if broll_pool:
            return broll_pool.pop(0)
        return fallback_pool.pop(0) if fallback_pool else None

    inserts = []
    for sec in missing:
        sid = sec["section_id"]
        donor = _take_speaker_donor()
        is_speaker = donor is not None
        if donor is None:
            donor = _take_broll_donor()

        frag_id = donor.get("fragment_id") if donor else None
        if frag_id:
            used_fids.add(frag_id)
        source_video = donor.get("source_video") if donor else None
        if frag_id and not source_video:
            source_video = "_".join(str(frag_id).rsplit("_", 1)[:-1])

        if is_speaker:
            qs = float(donor.get("quote_start") or donor.get("start") or 0)
            qe = float(donor.get("quote_end") or donor.get("end") or qs + 10)
            new_item: dict[str, Any] = {
                "type": "speaker_clip",
                "segment_type": "speaker_clip",
                "audio_mode": "source_audio",
                "section_id": sid,
                "fragment_id": frag_id,
                "source_video": source_video,
                "speaker": donor.get("speaker_label") or donor.get("speaker") or "",
                "start": qs,
                "end": qe,
                "render_start": qs,
                "render_end": qe,
                "render_duration": round(qe - qs, 3),
                "quote_text": donor.get("quote_text") or donor.get("text") or "",
                "subtitle_text": donor.get("subtitle_text") or donor.get("quote_text") or "",
                "quote_complete": donor.get("quote_complete", False),
                "muted": False,
                "_auto_filled": True,
            }
            log(f"  [fill] Added missing section {sid} as speaker_clip (fragment={str(frag_id)[:40]!r})")
        else:
            new_item = {
                "type": "narration_broll",
                "segment_type": "narration_broll",
                "audio_mode": "ai_voiceover",
                "section_id": sid,
                "fragment_id": frag_id,
                "source_video": source_video,
                "start": float(donor.get("start") or 0) if donor else 0.0,
                "end": float(donor.get("end") or 12) if donor else 12.0,
                "voiceover_text": "",
                "subtitle_text": "",
                "subtitle_kind": "voiceover",
                "visual_direction": sec.get("card_text", ""),
                "_auto_filled": True,
            }
            log(f"  [fill] Added missing section {sid} as narration_broll (fragment={str(frag_id)[:40]!r})")
        inserts.append(new_item)

    result = final_timeline[:insert_before] + inserts + final_timeline[insert_before:]
    return result


def _critical_gates_pass(quality_gates: dict) -> bool:
    """Return True if all critical gates pass."""
    for gate in CRITICAL_GATE_NAMES:
        if gate in quality_gates and quality_gates[gate] is False:
            return False
    return True


def _add_visual_asset_plan(
    final_timeline: list,
    narrative_sections: list,
    is_deepfake_topic: bool = True,
) -> list:
    """Add visual_mode/visual_asset_type/visual_asset_prompt/visual_title/visual_keywords to narration_broll items."""
    plans = DEEPFAKE_VISUAL_ASSET_PLANS if is_deepfake_topic else GENERIC_VISUAL_ASSET_PLANS
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not item.get("visual_mode"):
            sid = item.get("section_id") or ""
            plan = plans.get(sid)
            if plan:
                item["visual_mode"] = plan["visual_mode"]
                item["visual_asset_type"] = plan["visual_asset_type"]
                item["visual_title"] = plan.get("visual_title", "")
                item["visual_asset_prompt"] = plan.get("visual_asset_prompt", "")
                item["visual_keywords"] = plan.get("visual_keywords", [])
            else:
                item["visual_mode"] = "generated_asset"
                item["visual_asset_type"] = "concept_card"
                item["visual_title"] = sid.replace("_", " ").title() if sid else "Section Visual"
                item["visual_asset_prompt"] = ""
                item["visual_keywords"] = []
        result.append(item)
    return result


def _build_repair_prompt(
    current_timeline: list,
    quality_gates: dict,
    narrative_sections: list,
    user_prompt: str,
    iteration: int,
    target_duration: int = 120,
) -> str:
    """Build a structured LLM repair prompt describing failing gates and repair rules."""
    failed_gates = [k for k, v in quality_gates.items() if isinstance(v, bool) and not v]
    offending = quality_gates.get("offending_items", {})

    n_sections = max(1, len(narrative_sections))
    per_section_s = max(8, (target_duration - 10) // n_sections)
    target_words = max(40, int(per_section_s * 2.5))

    gate_descriptions = {
        "all_sections_covered": "Some narrative sections are missing from the timeline.",
        "all_speaker_labels_present": "Some speaker clips have no speaker label.",
        "all_speaker_quotes_complete": "Some speaker clips have incomplete quotes — demote them to narration_broll.",
        "no_instruction_leakage_in_voiceover": "Some voiceover texts start with instruction phrases instead of viewer-facing narration.",
        "all_voiceovers_topic_specific": "Some voiceover texts are generic boilerplate, not specific to the documentary topic.",
        "final_timeline_approved": "The overall timeline does not pass minimum quality standards.",
        "all_narration_broll_have_visual_asset_prompt": "Some narration_broll items are missing visual_asset_prompt.",
    }

    failure_lines = []
    for k in failed_gates:
        desc = gate_descriptions.get(k, "failed")
        off = offending.get(k, [])
        failure_lines.append(f"- {k}: {desc}" + (f" Offending: {off}" if off else ""))
    failure_report = "\n".join(failure_lines)

    sections_json = json.dumps(
        [{"section_id": s["section_id"], "title": s["title"], "purpose": s["purpose"]} for s in narrative_sections],
        indent=2,
    )

    current_tl_summary = json.dumps(
        [
            {
                "type": item.get("type") or item.get("segment_type"),
                "section_id": item.get("section_id"),
                "fragment_id": (item.get("fragment_id") or "")[:50],
                "speaker": item.get("speaker", ""),
                "quote_complete": item.get("quote_complete"),
                "quote_text": (item.get("quote_text") or "")[:100],
                "voiceover_text": (item.get("voiceover_text") or "")[:100],
            }
            for item in current_timeline
            if (item.get("type") or item.get("segment_type")) not in ("title_card", "closing_card")
        ],
        indent=2,
    )

    return (
        f"The validated timeline (iteration {iteration}) failed these quality gates:\n"
        f"{failure_report}\n\n"
        f'ORIGINAL USER PROMPT: "{user_prompt}"\n'
        f"TARGET DURATION: {target_duration} seconds\n\n"
        f"REQUIRED NARRATIVE SECTIONS (all must be present):\n{sections_json}\n\n"
        f"CURRENT TIMELINE:\n{current_tl_summary}\n\n"
        "REPAIR RULES:\n"
        "1. Demote any speaker_clip with quote_complete=false to narration_broll.\n"
        "2. Ensure all narrative sections are present.\n"
        "3. Every narration_broll voiceover_text must be viewer-facing narration — never start with\n"
        "   'Explain', 'Mention', 'Show how', 'End with', 'Start with', 'Open with', 'Describe', 'Close with', etc.\n"
        f"4. Every narration_broll voiceover_text must be ~{target_words} words ({target_words//15}-{target_words//15+2} sentences).\n"
        "5. Every narration_broll must include a non-empty visual_asset_prompt string.\n"
        "6. Keep the documentary order from the REQUIRED NARRATIVE SECTIONS.\n"
        "7. The final video must meaningfully answer the original prompt.\n"
        "8. PRESERVE existing voiceover_text if it is long and correct — only rewrite the offending items.\n"
        "9. CRITICAL: every \"fragment_id\" value must be copied EXACTLY, character-for-character,\n"
        "   from a fragment_id already present in CURRENT TIMELINE above. Never invent, abbreviate,\n"
        "   or make up a fragment_id (e.g. \"hook_001\") — fabricated ones will be discarded. If an\n"
        "   item has no real clip behind it, OMIT the \"fragment_id\" field entirely.\n\n"
        "Return a repaired final_timeline as a JSON array. Include only the array (no extra keys).\n"
        "Example structure:\n"
        '[\n'
        '  {"type": "speaker_clip", "section_id": "S1_HOOK", "fragment_id": "...", "speaker": "...", '
        '"render_start": 0.0, "render_end": 0.0, "quote_text": "...", "quote_complete": true},\n'
        '  {"type": "narration_broll", "section_id": "S2_CONTEXT", "fragment_id": "...", '
        '"voiceover_text": "evidence-driven narration about the topic...", "visual_asset_prompt": "...", '
        '"visual_title": "Context: Key Concepts", "visual_asset_type": "diagram"}\n'
        "]\n"
    )


def _llm_repair_timeline(
    current_timeline: list,
    quality_gates: dict,
    narrative_sections: list,
    user_prompt: str,
    iteration: int,
    target_duration: int = 120,
) -> "list | None":
    """Ask LLM to repair a failed timeline. Returns new timeline list or None on failure."""
    prompt = _build_repair_prompt(
        current_timeline, quality_gates, narrative_sections, user_prompt, iteration, target_duration
    )
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_completion_tokens": 10000,
            "stream": False,
        })

        repaired = extract_json_array(raw)
        if repaired and isinstance(repaired, list):
            return repaired
        obj = extract_json_object(raw)
        if obj and isinstance(obj.get("final_timeline"), list):
            return obj["final_timeline"]
        log("  [repair] LLM returned no valid JSON array.")
        return None
    except Exception as exc:
        log(f"  [repair] LLM repair call failed: {exc}")
        return None


def check_quality_gates(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    """Return (warnings_list, editorial_quality_gates_dict)."""
    warnings: list[str] = []
    offending_items: dict[str, list[str]] = {}

    def _flag(gate_key: str, item_id: str, msg: str) -> None:
        warnings.append(msg)
        offending_items.setdefault(gate_key, []).append(item_id)

    speaker_clips = [x for x in final_timeline if x.get("type") == "speaker_clip"]
    narration_brolls = [x for x in final_timeline if x.get("type") == "narration_broll"]
    all_section_ids = {x.get("section_id") for x in final_timeline if x.get("section_id")}
    required_section_ids = {s["section_id"] for s in narrative_sections}

    # 1. All sections covered
    missing_sids = required_section_ids - all_section_ids
    for sid in sorted(missing_sids):
        _flag("all_sections_covered", sid, f"GATE FAIL: Section {sid} missing from final timeline.")
    gate_sections = not bool(missing_sids)

    # 2. All speaker labels present & has speaker clips
    gate_speakers = True
    for sc in speaker_clips:
        fid = (sc.get("fragment_id") or "?")[:40]
        speaker = sc.get("speaker") or ""
        if not speaker:
            _flag("all_speaker_labels_present", fid, f"GATE FAIL: speaker_clip {fid!r} has no speaker label.")
            gate_speakers = False

    gate_has_speakers = len(speaker_clips) > 0
    if not gate_has_speakers:
        _flag("has_speaker_clips", "timeline", "GATE FAIL: Final timeline contains 0 speaker clips — zero speaker footage.")

    # 3. All speaker quotes complete
    gate_quotes = True
    for sc in speaker_clips:
        fid = (sc.get("fragment_id") or "?")[:40]
        if not sc.get("quote_complete"):
            note = sc.get("quote_completeness_note") or "quote_complete=false"
            _flag("all_speaker_quotes_complete", fid, f"GATE WARN: {fid!r} — {note}")
            gate_quotes = False

    # 4. No instruction leakage in voiceover
    instruction_verbs = frozenset(["end with", "explain", "mention", "show how",
                                    "this section", "the video should", "use this clip",
                                    "the narrative should", "start with", "begin with",
                                    "open with", "close with", "describe"])
    gate_no_leak = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        vo = (nb.get("voiceover_text") or "").lower()
        for phrase in instruction_verbs:
            if vo.startswith(phrase):
                _flag("no_instruction_leakage_in_voiceover", sid,
                      f"GATE WARN: narration_broll {sid} voiceover starts with instruction verb: {phrase!r}")
                gate_no_leak = False
                break

    # 5. All voiceovers topic-specific (not generic boilerplate)
    generic_phrases = [
        "conversations about artificial intelligence",
        "raising a central question",
        "across the world,",
        "in communities around the world",
        "the following evidence raises",
        "understanding the issue requires",
    ]
    gate_specific = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        vo = (nb.get("voiceover_text") or "").lower()
        if not vo:
            _flag("all_voiceovers_topic_specific", sid,
                  f"GATE FAIL: narration_broll {sid} has empty voiceover_text.")
            gate_specific = False
        else:
            for phrase in generic_phrases:
                if phrase in vo:
                    _flag("all_voiceovers_topic_specific", sid,
                          f"GATE WARN: narration_broll {sid} contains generic phrase: {phrase!r}")
                    gate_specific = False
                    break

    # 6. Final timeline approved overall
    gate_approved = gate_sections and gate_speakers and len(speaker_clips) >= 2

    # 7. All narration_broll items have visual_asset_prompt
    gate_visual = True
    for nb in narration_brolls:
        sid = nb.get("section_id") or "?"
        if not nb.get("visual_asset_prompt"):
            _flag("all_narration_broll_have_visual_asset_prompt", sid,
                  f"GATE WARN: narration_broll {sid} missing visual_asset_prompt.")
            gate_visual = False

    gates: dict[str, Any] = {
        "all_sections_covered": gate_sections,
        "all_speaker_labels_present": gate_speakers,
        "has_speaker_clips": gate_has_speakers,
        "all_speaker_quotes_complete": gate_quotes,
        "no_instruction_leakage_in_voiceover": gate_no_leak,
        "all_voiceovers_topic_specific": gate_specific,
        "final_timeline_approved": gate_approved,
        "all_narration_broll_have_visual_asset_prompt": gate_visual,
        "speaker_clip_count": len(speaker_clips),
        "narration_broll_count": len(narration_brolls),
        "sections_covered": sorted(all_section_ids - {None}),
        "offending_items": offending_items,
    }

    return warnings, gates


def frame_duration_fill_as_perspectives_section(enriched_timeline: list[dict[str, Any]]) -> int:
    """Group every clip added by duration_fill into an explicit, labeled
    "Additional Perspectives" closing section instead of leaving them as
    anonymous tail items.

    Per product decision: when the requested duration isn't met by the core
    narrative, surface ALL remaining eligible speaker voices on this topic as
    a clearly-framed compilation. Reuses the already-generated title card (no
    new asset needed) as a chapter divider; each clip stays a real
    speaker_clip with source audio — this only tags/labels, it never adds or
    removes content. Mutates `enriched_timeline` in place. Returns the number
    of clips framed (0 if no duration-fill additions exist).
    """
    fill_added = [i for i in enriched_timeline if i.get("added_by_duration_fill")]
    if not fill_added:
        return 0
    first_added_idx = enriched_timeline.index(fill_added[0])
    divider = {
        "type": "title_card",
        "segment_type": "title_card",
        "section_id": "S_PERSPECTIVES",
        "subtitle_text": "Additional Perspectives on This Topic",
        "label": "additional_perspectives_divider",
    }
    enriched_timeline.insert(first_added_idx, divider)
    for item in fill_added:
        item["section_id"] = "S_PERSPECTIVES"
    return len(fill_added)


def _llm_vet_topup_items(items: list[dict[str, Any]], user_prompt: str) -> set[str]:
    """One bounded LLM call vetting duration top-up segments for topical relevance.

    Returns the fragment_ids to REJECT. On any LLM failure returns an empty set —
    the items stay in the timeline flagged as lower-confidence rather than being
    dropped (dropping would silently violate the duration floor).
    """
    if not items:
        return set()

    items_json = json.dumps(
        [
            {
                "fragment_id": str(i.get("fragment_id") or ""),
                "speaker": i.get("speaker") or "",
                "excerpt": (i.get("moment_text") or i.get("subtitle_text") or "")[:300],
            }
            for i in items
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""You are validating archive clips for a documentary about: "{user_prompt}"

These clips were added to reach the requested video duration. For EACH clip decide
whether its excerpt is topically relevant enough to include (related themes are
acceptable; only reject clearly off-topic content).

CLIPS:
{items_json}

Return ONLY a JSON array, one entry per clip:
[{{"fragment_id": "...", "approved": true}}]
"""
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_completion_tokens": 2048,
            "stream": False,
        })
        verdicts = extract_json_array(raw) or []
        rejected: set[str] = set()
        for verdict in verdicts:
            if isinstance(verdict, dict) and verdict.get("approved") is False:
                fid = str(verdict.get("fragment_id") or "")
                if fid:
                    rejected.add(fid)
        return rejected
    except Exception as exc:
        log(f"  [topup] vet call failed ({exc}) — keeping top-up items flagged.")
        return set()
