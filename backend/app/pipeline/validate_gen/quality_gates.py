"""Editorial quality-gate checks, the LLM repair loop, and duration top-up vetting."""
import json
from typing import Any

from app.pipeline.prompt_safety import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted
from app.pipeline.storyline_gen.intents.base import agenda_rule
from app.pipeline.timeline_budget import _candidate_is_speaker_eligible
from app.pipeline.validate_gen.fragment_validation import _base_section_id, _quote_ends_completely, _quote_starts_cleanly
from app.pipeline.validate_gen.json_utils import extract_json_array, extract_json_object
from app.pipeline.validate_gen.llm_client import STRONG_LLM_MODEL, log, post_with_retries
from app.pipeline.validate_gen.narrative_templates import GENERIC_VISUAL_ASSET_PLANS

# Fixed instructional preamble for the timeline-repair LLM call — kept
# separate from the (untrusted) user_prompt/current-timeline content that
# _build_repair_prompt interpolates, so the two are sent as distinct
# system/user messages rather than one flattened string.
_REPAIR_SYSTEM = "You are repairing a documentary video timeline that failed automated quality gates." + UNTRUSTED_CONTENT_NOTICE

_TOPUP_VET_SYSTEM = (
    "You are validating archive clips for topical relevance to a documentary. "
    "These clips were added to reach the requested video duration. For EACH "
    "clip decide whether its excerpt is topically relevant enough to include "
    "(related themes are acceptable; only reject clearly off-topic content). "
    "Also reject any clip that is not on the AI4IA agenda:\n" + agenda_rule()
) + UNTRUSTED_CONTENT_NOTICE

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

# Speaker clips an approved timeline aims for, when that many were approved.
SPEAKER_CLIP_TARGET = 2
# Extra shortlist clips to check when too few speaker clips were approved.
EXTRA_SPEAKER_CHECKS = 6
# Most speaker clips a video may use when the request does not need clips.
OPTIONAL_CLIP_CAP = 2
# Gates a repair can only fix by adding speaker clips.
_SPEAKER_COUNT_GATES = frozenset({"has_speaker_clips", "final_timeline_approved"})


def _item_type(item: dict[str, Any]) -> str:
    return item.get("type") or item.get("segment_type") or ""


def count_speaker_clips(timeline: list[dict[str, Any]]) -> int:
    return sum(1 for item in timeline if _item_type(item) == "speaker_clip")


def speaker_clip_target(validated_candidates: list[dict[str, Any]]) -> int:
    """SPEAKER_CLIP_TARGET, lowered to the number of approved speaker clips."""
    approved = sum(1 for c in validated_candidates
                   if c.get("approved") and c.get("decision") == "speaker_clip")
    return min(SPEAKER_CLIP_TARGET, approved)


def unchecked_speaker_candidates(
    shortlist: list[dict[str, Any]],
    checked_fids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Speaker-eligible shortlist clips not yet checked, in stage 07's rank order."""
    picked: list[dict[str, Any]] = []
    seen = set(checked_fids)
    for candidate in shortlist:
        fid = str(candidate.get("fragment_id") or "")
        if fid and fid not in seen and _candidate_is_speaker_eligible(candidate):
            picked.append(candidate)
            seen.add(fid)
        if len(picked) >= limit:
            break
    return picked


def repair_shelf(
    timeline: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Approved clips not already placed as speaker clips, speaker clips first.
    A speaker clip the assembly placed as narration counts as available."""
    used = {str(item.get("fragment_id")) for item in timeline
            if item.get("fragment_id") and _item_type(item) == "speaker_clip"}
    shelf = [c for c in validated_candidates
             if c.get("approved") and c.get("fragment_id") and str(c["fragment_id"]) not in used]
    return sorted(shelf, key=lambda c: c.get("decision") != "speaker_clip")


def repair_can_help(quality_gates: dict[str, Any], shelf: list[dict[str, Any]]) -> bool:
    """False when the only failures need speaker clips that do not exist."""
    failed = {k for k in CRITICAL_GATE_NAMES if quality_gates.get(k) is False}
    if failed - _SPEAKER_COUNT_GATES:
        return True
    return any(c.get("decision") == "speaker_clip" for c in shelf)


def repair_is_better(
    before: tuple[list[dict[str, Any]], dict[str, Any]],
    after: tuple[list[dict[str, Any]], dict[str, Any]],
) -> bool:
    """A repair may not lose speaker clips or pass fewer critical gates."""
    def passing(gates: dict[str, Any]) -> int:
        return sum(1 for k in CRITICAL_GATE_NAMES if gates.get(k) is not False)

    (tl_before, gates_before), (tl_after, gates_after) = before, after
    return (count_speaker_clips(tl_after) >= count_speaker_clips(tl_before)
            and passing(gates_after) >= passing(gates_before))


_OPENING_WORDS = ("HOOK", "OPENING", "INTRO")
_CLOSING_WORDS = ("SYNTHESIS", "TAKEAWAY", "CLOSING", "CONCLUSION")


def align_section_ids(
    timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map a section id the LLM invented or copied from an example ("S1_HOOK")
    onto this format: opening and closing words by meaning, others by number."""
    ids = [s["section_id"] for s in narrative_sections]
    if not ids:
        return timeline
    by_number: dict[str, str] = {}
    for sid in ids:
        by_number.setdefault(sid.split("_", 1)[0], sid)
    for item in timeline:
        sid = str(item.get("section_id") or "")
        if not sid or sid in ids or sid == "S_PERSPECTIVES":
            continue
        upper = sid.upper()
        if any(word in upper for word in _OPENING_WORDS):
            item["section_id"] = ids[0]
        elif any(word in upper for word in _CLOSING_WORDS):
            item["section_id"] = ids[-1]
        elif upper.split("_", 1)[0] in by_number:
            item["section_id"] = by_number[upper.split("_", 1)[0]]
    return timeline


def limit_speaker_clips(timeline: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Keep the first `cap` speaker clips; drop the rest."""
    kept = 0
    result = []
    for item in timeline:
        if _item_type(item) == "speaker_clip":
            kept += 1
            if kept > cap:
                continue
        result.append(item)
    return result


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
    """Narration first in every section, then its speaker clips. The first and
    last sections are narration only: their clips move to the nearest middle section."""
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

    # Clips never open or close the film.
    middle = ordered_sections[1:-1]
    if middle:
        def take_clips(sid: str) -> list[dict[str, Any]]:
            clips = [x for x in section_buckets[sid] if x.get("type") == "speaker_clip"]
            section_buckets[sid] = [x for x in section_buckets[sid] if x.get("type") != "speaker_clip"]
            return clips

        opening, closing = take_clips(ordered_sections[0]), take_clips(ordered_sections[-1])
        after_opening, before_closing = middle[0], middle[-1]
        section_buckets[after_opening] = (
            [{**x, "section_id": after_opening} for x in opening] + section_buckets[after_opening])
        section_buckets[before_closing] += [{**x, "section_id": before_closing} for x in closing]

    result: list[dict[str, Any]] = []
    # title_card always first
    title_cards = [x for x in card_items if x.get("type") == "title_card"]
    closing_cards = [x for x in card_items if x.get("type") == "closing_card"]
    result.extend(title_cards)

    for sid in ordered_sections:
        items = section_buckets.get(sid, [])
        if not items:
            continue
        result.extend(x for x in items if x.get("type") != "speaker_clip")
        result.extend(x for x in items if x.get("type") == "speaker_clip")

    result.extend(unassigned)
    result.extend(closing_cards)
    return result


def _fill_missing_sections(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    original_timeline: list[dict[str, Any]],
    clip_sections: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Ensure every narrative section has at least one item.

    `clip_sections` limits where a speaker-clip donor may go (None: anywhere).

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
        donor = _take_speaker_donor() if clip_sections is None or sid in clip_sections else None
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
) -> list:
    """Add visual_mode/visual_asset_type/visual_asset_prompt/visual_title/visual_keywords to narration_broll items."""
    # An intent's format carries its own plan per section.
    plans = {**GENERIC_VISUAL_ASSET_PLANS, **{s["section_id"]: s["visual"] for s in narrative_sections if s.get("visual")}}
    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not item.get("visual_mode"):
            sid = item.get("section_id") or ""
            # A repeated section on a long video (e.g. "S3_EVIDENCE_2") isn't
            # in the plan table — only the base ids are. Without stripping the
            # pass suffix, the lookup misses and falls to the raw-slug
            # fallback below ("S3 Evidence 2" as a literal card headline).
            plan = plans.get(sid) or plans.get(_base_section_id(sid))
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
    shelf: list | None = None,
    speaker_target: int = SPEAKER_CLIP_TARGET,
    clips_required: bool = True,
) -> tuple[str, str]:
    """Build a structured LLM repair prompt describing failing gates and repair
    rules. Returns (system, user) — user_prompt and the current timeline
    summary (both untrusted: user-submitted / transcript-derived) are wrapped
    via wrap_untrusted() rather than concatenated raw into the instructions."""
    failed_gates = [k for k, v in quality_gates.items() if isinstance(v, bool) and not v]
    offending = quality_gates.get("offending_items", {})
    shelf = shelf or []
    n_speakers = count_speaker_clips(current_timeline)
    speaker_rule = (f"The timeline has {n_speakers} speaker clip(s) and needs at least {speaker_target}. "
                    "Add speaker clips from AVAILABLE CLIPS.")
    first_id = narrative_sections[0]["section_id"] if narrative_sections else "S1"
    middle_id = narrative_sections[1]["section_id"] if len(narrative_sections) > 2 else first_id
    clip_use = ("" if clips_required else
                f"   Speaker clips are optional for this request: use at most {OPTIONAL_CLIP_CAP}.\n")

    n_sections = max(1, len(narrative_sections))
    per_section_s = max(8, (target_duration - 10) // n_sections)
    target_words = max(40, int(per_section_s * 2.5))

    gate_descriptions = {
        "all_sections_covered": "Some narrative sections are missing from the timeline.",
        "all_speaker_labels_present": "Some speaker clips have no speaker label.",
        "all_speaker_quotes_complete": "Some speaker clips have incomplete quotes — demote them to narration_broll.",
        "no_instruction_leakage_in_voiceover": "Some voiceover texts start with instruction phrases instead of viewer-facing narration.",
        "all_voiceovers_topic_specific": "Some voiceover texts are generic boilerplate, not specific to the documentary topic.",
        "final_timeline_approved": speaker_rule,
        "has_speaker_clips": speaker_rule,
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
                # Full id: the repair must copy it exactly or the clip is discarded.
                "fragment_id": item.get("fragment_id") or "",
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

    shelf_json = json.dumps(
        [
            {
                "fragment_id": c.get("fragment_id"),
                "decision": c.get("decision"),
                "section_id": c.get("section_id"),
                "speaker": c.get("speaker", ""),
                "quote_text": (c.get("quote_text") or c.get("text") or "")[:400],
            }
            for c in shelf
        ],
        indent=2,
    )

    user = (
        f"The validated timeline (iteration {iteration}) failed these quality gates:\n"
        f"{failure_report}\n\n"
        f"ORIGINAL USER PROMPT: {wrap_untrusted(user_prompt)}\n"
        f"TARGET DURATION: {target_duration} seconds\n\n"
        f"REQUIRED NARRATIVE SECTIONS (all must be present):\n{sections_json}\n\n"
        f"CURRENT TIMELINE:\n{wrap_untrusted(current_tl_summary)}\n\n"
        "AVAILABLE CLIPS (approved, not yet used; decision=speaker_clip means real speaker audio):\n"
        f"{wrap_untrusted(shelf_json)}\n\n"
        "REPAIR RULES:\n"
        "1. Demote a speaker_clip to narration_broll only if quote_complete=false. Never demote a\n"
        "   speaker_clip whose quote_complete=true, and never remove one.\n"
        "2. Ensure all narrative sections are present.\n"
        "3. Every narration_broll voiceover_text must be viewer-facing narration — never start with\n"
        "   'Explain', 'Mention', 'Show how', 'End with', 'Start with', 'Open with', 'Describe', 'Close with', etc.\n"
        f"4. Every narration_broll voiceover_text must be ~{target_words} words ({target_words//15}-{target_words//15+2} sentences).\n"
        "5. Every narration_broll must include a non-empty visual_asset_prompt string.\n"
        "6. Keep the documentary order from the REQUIRED NARRATIVE SECTIONS.\n"
        "7. The final video must meaningfully answer the original prompt.\n"
        "8. PRESERVE existing voiceover_text if it is long and correct — only rewrite the offending items.\n"
        "9. CRITICAL: every \"fragment_id\" value must be copied EXACTLY, character-for-character,\n"
        "   from CURRENT TIMELINE or AVAILABLE CLIPS above. Never invent, abbreviate,\n"
        "   or make up a fragment_id (e.g. \"hook_001\") — fabricated ones will be discarded. If an\n"
        "   item has no real clip behind it, OMIT the \"fragment_id\" field entirely.\n"
        "10. The first and last sections are narration only. Speaker clips go in the middle sections,\n"
        "   each after that section's narration.\n" + clip_use +
        "11. Speaker clips must be on the AI4IA agenda:\n" + agenda_rule() + "\n\n"
        "Return a repaired final_timeline as a JSON array. Include only the array (no extra keys).\n"
        "Example structure:\n"
        '[\n'
        f'  {{"type": "narration_broll", "section_id": "{middle_id}", "fragment_id": "...", '
        '"voiceover_text": "evidence-driven narration about the topic...", "visual_asset_prompt": "...", '
        '"visual_title": "Context: Key Concepts", "visual_asset_type": "diagram"},\n'
        f'  {{"type": "speaker_clip", "section_id": "{middle_id}", "fragment_id": "...", "speaker": "...", '
        '"render_start": 0.0, "render_end": 0.0, "quote_text": "...", "quote_complete": true}\n'
        "]\n"
    )
    return _REPAIR_SYSTEM, user


def _llm_repair_timeline(
    current_timeline: list,
    quality_gates: dict,
    narrative_sections: list,
    user_prompt: str,
    iteration: int,
    target_duration: int = 120,
    shelf: list | None = None,
    speaker_target: int = SPEAKER_CLIP_TARGET,
    clips_required: bool = True,
) -> "list | None":
    """Ask LLM to repair a failed timeline. Returns new timeline list or None on failure."""
    system_text, user_text = _build_repair_prompt(
        current_timeline, quality_gates, narrative_sections, user_prompt, iteration, target_duration,
        shelf=shelf, speaker_target=speaker_target, clips_required=clips_required,
    )
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
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
    speaker_target: int = SPEAKER_CLIP_TARGET,
    clips_required: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Return (warnings_list, editorial_quality_gates_dict). A request that does
    not need clips passes the speaker-count gates with none."""
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

    gate_has_speakers = len(speaker_clips) > 0 or not clips_required
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
    enough_clips = len(speaker_clips) >= max(1, speaker_target) if clips_required else True
    gate_approved = gate_sections and gate_speakers and enough_clips

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


def frame_duration_fill_as_perspectives_section(
    enriched_timeline: list[dict[str, Any]],
    closing_section_id: str | None = None,
) -> int:
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
    if closing_section_id:
        _move_before_closing_section(enriched_timeline, [divider, *fill_added], closing_section_id)
    return len(fill_added)


def _move_before_closing_section(
    timeline: list[dict[str, Any]],
    block: list[dict[str, Any]],
    closing_section_id: str,
) -> None:
    """Move `block` in front of the closing section, so the film still ends on narration."""
    rest = [item for item in timeline if not any(item is b for b in block)]
    at = next((i for i, item in enumerate(rest) if item.get("section_id") == closing_section_id), None)
    if at is None:
        return
    timeline[:] = rest[:at] + block + rest[at:]


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

    user_text = f"""DOCUMENTARY TOPIC: {wrap_untrusted(user_prompt)}

CLIPS:
{wrap_untrusted(items_json)}

Return ONLY a JSON array, one entry per clip:
[{{"fragment_id": "...", "approved": true}}]
"""
    try:
        raw = post_with_retries({
            "model": STRONG_LLM_MODEL,
            "messages": [
                {"role": "system", "content": _TOPUP_VET_SYSTEM},
                {"role": "user", "content": user_text},
            ],
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
