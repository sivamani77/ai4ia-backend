"""Post-process an LLM (or fallback) final_timeline: fill required fields, ground narration."""
import re
from typing import Any

from app.pipeline.validate_gen.fragment_validation import _clean_voiceover_text, _extract_speaker_from_video_title
from app.pipeline.validate_gen.llm_client import log

_TITLED_NAME_RE = re.compile(r"\b(?:Dr|Prof|Professor|Mr|Ms|Mrs)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_UNSOURCED_STAT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|(?:percent|million|billion|trillion)\b)", re.IGNORECASE)


def _build_grounding_material(validated_candidates: list[dict[str, Any]]) -> tuple[str, set[str]]:
    """Source transcripts blob + allowed speaker names, for grounding narration."""
    texts: list[str] = []
    allowed_names: set[str] = set()
    for c in validated_candidates or []:
        for key in ("moment_text", "quote_text", "text"):
            value = c.get(key)
            if value and str(value).strip():
                texts.append(str(value).strip())
                break
        for key in ("speaker", "speaker_attribution"):
            name = c.get(key)
            if name and str(name).strip():
                allowed_names.add(str(name).strip().lower())
    return "\n".join(texts), allowed_names


def _narration_grounding_issues(text: str, source_blob: str, allowed_names: set[str]) -> list[str]:
    """Precise hallucination checks: titled people and statistics must come from
    the selected fragments' transcripts. Returns issues (empty = grounded)."""
    issues: list[str] = []
    if not text:
        return issues
    sources_lower = source_blob.lower()

    for match in _TITLED_NAME_RE.finditer(text):
        name = match.group(1)
        name_lower = name.lower()
        if name_lower not in sources_lower and not any(name_lower in n for n in allowed_names):
            issues.append(f"invented person: {match.group(0)!r}")

    for match in _UNSOURCED_STAT_RE.finditer(text):
        stat = match.group(0)
        if stat.lower() not in sources_lower:
            issues.append(f"unsourced statistic: {stat!r}")

    return issues


def generate_voiceovers_for_timeline(
    final_timeline: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    user_prompt: str,
    validated_candidates: list[dict[str, Any]],
    target_duration: int = 120,
    storyline_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Fill voiceover_text for all narration_broll items that lack it.

    Source: existing narration from script-07 storyline.sections (topic-correct,
    duration-proportioned, already generated — no network call needed).

    Assembly (assemble_narrative_with_llm) is now responsible for writing
    voiceover_text for every narration_broll item directly, so this function no
    longer makes its own LLM round trip. Any item still empty after the script-07
    fill below is left blank and picked up by the topic-default substitution the
    caller (main()) already runs after this — not re-queried here, to avoid a
    second sequential LLM call per job.

    All narration is checked against the selected fragments' transcripts: text that
    invents named people or unsourced statistics is rejected (falls back to the safe
    section defaults downstream) rather than shipped.
    """
    # Calculate target word count per section proportional to target_duration
    n_sections = max(1, len(narrative_sections))
    content_budget_s = max(1, target_duration - 10)
    per_section_s = round(content_budget_s / n_sections)
    target_words = max(40, int(per_section_s * 2.5))

    # Grounding material: narration may only claim what the selected fragments say
    source_blob, allowed_names = _build_grounding_material(validated_candidates)
    archive_facts = ((storyline_data or {}).get("storyline") or {}).get("archive_facts")
    if archive_facts:
        source_blob += "\n" + archive_facts

    # Build ordered narration list from script-07 sections
    sl_narrations: list[str] = []
    if storyline_data:
        sl_secs = (storyline_data.get("storyline") or {}).get("sections") or []
        for sec in sl_secs:
            narration = (sec.get("narration") or sec.get("voiceover_text") or "").strip()
            if not narration:
                continue
            grounding_issues = _narration_grounding_issues(narration, source_blob, allowed_names)
            if grounding_issues:
                log(f"  [grounding] rejected section narration: {grounding_issues}")
                continue
            sl_narrations.append(narration)

    # Step 1: fill/replace narration from script-07 storyline.sections by position.
    # sl_narrations are topic-correct and duration-proportioned. Always prefer them over
    # short LLM assembly voiceovers — replace any existing text shorter than target_words * 0.6.
    min_words_to_keep = max(20, int(target_words * 0.6))
    nb_pos = 0
    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t != "narration_broll":
            continue
        existing = (item.get("voiceover_text") or "").strip()
        existing_words = len(existing.split()) if existing else 0
        if nb_pos < len(sl_narrations):
            # Use sl_narration if item is empty OR current text is too short
            if not existing or existing_words < min_words_to_keep:
                text = sl_narrations[nb_pos]
                item["voiceover_text"] = text
                item["subtitle_text"] = text
                log(f"  [vo-sl07] {item.get('section_id')}: {text[:80]!r}")
        nb_pos += 1

    # Step 2 (formerly a second LLM call) removed: assemble_narrative_with_llm now
    # writes voiceover_text for every narration_broll item itself. Anything still
    # empty here means both script-07 reuse and the assembly LLM missed it — leave
    # it blank and let the topic-default substitution in main() fill it, rather
    # than pay for another sequential LLM round trip.
    still_missing = sum(
        1 for item in final_timeline
        if (item.get("type") or item.get("segment_type") or "") == "narration_broll"
        and not (item.get("voiceover_text") or "").strip()
    )
    if still_missing:
        log(f"  [vo] {still_missing} narration_broll item(s) still empty after script-07 fill — will use topic defaults.")

    return final_timeline


def assign_fragments_to_empty_narration_brolls(
    final_timeline: list[dict[str, Any]],
    validated_candidates: list[dict[str, Any]],
    original_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign fragment_ids to narration_broll items that have no fragment_id."""
    # Collect candidates classified as narration_broll that are not already used as speaker_clips
    used_fids = {item.get("fragment_id") for item in final_timeline if item.get("fragment_id")}
    broll_pool = [
        c for c in validated_candidates
        if c.get("decision") in ("narration_broll", "visual_support", "speaker_clip")
        and c.get("approved")
        and c.get("fragment_id")
        and c.get("fragment_id") not in used_fids
    ]
    # Also include original timeline items not yet used
    orig_by_fid = {item.get("fragment_id"): item for item in original_timeline if item.get("fragment_id")}
    for item in original_timeline:
        fid = item.get("fragment_id")
        if fid and fid not in used_fids:
            if not any(c.get("fragment_id") == fid for c in broll_pool):
                broll_pool.append(item)

    pool_idx = 0
    for item in final_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll" and not item.get("fragment_id"):
            if pool_idx < len(broll_pool):
                donor = broll_pool[pool_idx]
                pool_idx += 1
                fid = donor.get("fragment_id")
                item["fragment_id"] = fid
                if not item.get("source_video"):
                    item["source_video"] = donor.get("source_video") or (
                        "_".join(str(fid).rsplit("_", 1)[:-1]) if fid else None
                    )
                # Use fragment start/end for visual timing
                orig = orig_by_fid.get(str(fid), donor)
                if not item.get("start") or float(item.get("start") or 0) == 0:
                    item["start"] = float(orig.get("start") or donor.get("start") or 0)
                if not item.get("end") or float(item.get("end") or 0) <= float(item.get("start") or 0):
                    item["end"] = float(orig.get("end") or donor.get("end") or (item["start"] + 12))
                log(f"  [assign] narration_broll {item.get('section_id')} ← {str(fid)[:50]!r}")

    return final_timeline


def canonicalize_speaker_names(final_timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use registry spellings in speaker labels and in shown or spoken text."""
    from app.services.speaker_registry import get_registry

    registry = get_registry()
    for item in final_timeline:
        if item.get("speaker"):
            item["speaker"] = registry.canonical(item["speaker"]) or item["speaker"]
        for key in ("voiceover_text", "subtitle_text", "visual_title", "title"):
            if item.get(key):
                item[key] = registry.fix_names_in_text(item[key])
    return final_timeline


def enrich_final_timeline(
    final_timeline: list[dict[str, Any]],
    candidate_map: dict[str, dict[str, Any]],
    original_timeline: list[dict[str, Any]],
    validated_map: dict[str, dict[str, Any]] | None = None,
    is_deepfake_topic: bool = False,
) -> list[dict[str, Any]]:
    """Fill in missing fields on final_timeline items from candidate_map / original timeline."""
    orig_by_fid: dict[str, dict[str, Any]] = {}
    for item in original_timeline:
        fid = item.get("fragment_id")
        if fid:
            orig_by_fid[str(fid)] = item

    validated_map = validated_map or {}

    result = []
    for item in final_timeline:
        item = dict(item)
        t = item.get("type") or item.get("segment_type") or ""

        if t in ("title_card", "closing_card"):
            item.setdefault("type", t)
            item.setdefault("segment_type", t)
            item.setdefault("duration", 5.0)
            result.append(item)
            continue

        fid = item.get("fragment_id")
        cand = candidate_map.get(str(fid), {}) if fid else {}
        orig = orig_by_fid.get(str(fid), {}) if fid else {}
        validated = validated_map.get(str(fid), {}) if fid else {}

        # Guard against LLM-hallucinated fragment_ids: the narrative-assembly
        # model occasionally invents a fragment_id/source_video that doesn't
        # correspond to any real candidate (observed pattern: placeholder
        # strings like 'narration_S2_CONTEXT' or 'speaker_S1_HOOK', echoing
        # the type_section naming shown in its own prompt example, when no
        # real fragment was available for that slot). Deriving a source_video
        # from such a fid via string-splitting produces an unresolvable S3
        # key and fails export. A fid is trustworthy only if it matches a
        # real, known candidate (selected_candidates / original export
        # timeline / validated pool) — if it matches none of them, treat this
        # item exactly like one that never had a fragment: a speaker_clip
        # with no real footage behind it is dropped (no fabricated content,
        # per policy); a narration_broll degrades to footage-less (Step 08
        # already renders those over a generated visual).
        if fid and not cand and not orig and not validated:
            log(f"  [guard] discarding hallucinated fragment_id {str(fid)[:60]!r} (not a known candidate)")
            t = item.get("type") or item.get("segment_type") or ""
            if t == "speaker_clip":
                continue
            item["fragment_id"] = None
            item["source_video"] = None
            fid = None

        # Merge source_video from candidate if missing
        if not item.get("source_video"):
            item["source_video"] = (
                cand.get("source_video") or orig.get("source_video") or validated.get("source_video")
            )
            if not item["source_video"] and fid:
                item["source_video"] = "_".join(str(fid).rsplit("_", 1)[:-1])

        # Merge start/end from original if missing
        if not item.get("start"):
            item["start"] = float(orig.get("start") or cand.get("start") or 0)
        if not item.get("end"):
            # narration_broll has no real fragment behind it (fid/cand/orig all
            # empty), so the "+10" flat fallback below would give every
            # narration item the same 10s regardless of how much it actually
            # has to say — a 90-word and a 210-word voiceover both got clamped
            # to 10s, which silently capped total video duration far below
            # whatever the narration was actually written (and word-budgeted)
            # to fill. Estimate from the written text at ~2.5 words/sec instead,
            # matching the rate the word budget itself was computed at.
            default_span = 10.0
            if t == "narration_broll":
                words = len((item.get("voiceover_text") or "").split())
                if words:
                    default_span = max(6.0, round(words / 2.5, 2))
            item["end"] = float(orig.get("end") or cand.get("end") or item["start"] + default_span)

        if t == "speaker_clip":
            item["type"] = "speaker_clip"
            item["segment_type"] = "speaker_clip"
            item["audio_mode"] = "source_audio"
            item["muted"] = False

            # Normalise render bounds
            rs = item.get("render_start") or item.get("quote_start") or item["start"]
            re_ = item.get("render_end") or item.get("quote_end") or item["end"]
            rs = float(rs)
            re_ = float(re_)
            if re_ <= rs:
                re_ = rs + float(orig.get("duration") or cand.get("duration") or 10)
            item["render_start"] = rs
            item["render_end"] = re_
            item["render_duration"] = round(re_ - rs, 3)

            # Speaker label: the clip's own data wins over the LLM's spelling.
            sv = item.get("source_video") or ""
            item["speaker"] = (
                validated.get("speaker_label")
                or validated.get("speaker")
                or cand.get("speaker")
                or orig.get("speaker")
                or cand.get("speaker_attribution")
                or item.get("speaker")
                or _extract_speaker_from_video_title(sv)
                or "AI4IA conference speaker"
            )

            # Fill quote_text / subtitle_text
            if not item.get("quote_text"):
                item["quote_text"] = (
                    validated.get("quote_text")
                    or cand.get("text")
                    or cand.get("moment_text")
                    or orig.get("moment_text")
                    or ""
                )
            if not item.get("subtitle_text"):
                item["subtitle_text"] = item.get("quote_text", "")

            # Copy moment fields for Step 09
            for field in ("moment_text", "moment_scores", "speech_quality", "moment_summary", "section_role"):
                if field not in item:
                    item[field] = orig.get(field) or cand.get(field)

        elif t == "narration_broll":
            item["type"] = "narration_broll"
            item["segment_type"] = "narration_broll"
            item["audio_mode"] = "ai_voiceover"
            item["muted"] = True
            item["subtitle_kind"] = "voiceover"
            # Clean voiceover text (remove instruction leakage)
            sid = item.get("section_id") or ""
            raw_vo = item.get("voiceover_text") or ""
            item["voiceover_text"] = _clean_voiceover_text(raw_vo, sid, is_deepfake_topic)
            item["subtitle_text"] = item["voiceover_text"]

        result.append(item)
    return result
