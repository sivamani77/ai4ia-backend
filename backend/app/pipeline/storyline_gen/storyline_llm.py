import json
import re
from typing import Any

from app.pipeline.timeline_budget import timeline_segment_count_for_duration
from app.pipeline.storyline_gen.llm_client import (
    STORYLINE_LLM_BACKEND,
    STORYLINE_VLLM_MODEL,
    STRONG_LLM_MODEL,
    call_strong_llm,
    log,
)
from app.pipeline.storyline_gen.selection import (
    build_candidate_pack,
    build_export_timeline_from_candidates,
)


def generate_llm_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    candidate_pack = build_candidate_pack(candidates)

    _num_sections_hint = timeline_segment_count_for_duration(duration)
    _per_section_s_hint = round(max(1, duration - 10) / _num_sections_hint)
    _words_per_section_hint = max(30, round(_per_section_s_hint * 2.5))

    system_instruction = f"""
You are a senior documentary producer and AI research editor.

Create a production-ready storyline blueprint for a documentary-style video.
The source archive contains conference footage, speaker clips, panel discussions, and event visuals.
The final video should feel like a National Geographic / premium explainer documentary:
- narrator-led
- evidence-backed
- visually planned
- speaker soundbites used selectively
- generated or internet B-roll used where conference footage is insufficient

=== NARRATION LENGTH — CRITICAL REQUIREMENT ===
The video is {duration} seconds long with {_num_sections_hint} sections.
Each section narration MUST be at least {_words_per_section_hint} words long.
At 2.5 words/second TTS pace, {_words_per_section_hint} words = {_per_section_s_hint} seconds of voiceover per section.
NEVER write a narration shorter than {_words_per_section_hint} words. One or two sentences is NOT acceptable.
Write full, substantive paragraphs. Each narration should be a complete passage of {_words_per_section_hint}+ words.
===============================================

Important rules:
1. Do not invent speaker names unless they are visible in source_video text.
2. Use only fragment_id values from the provided candidate fragments.
3. Do not assign a permanent storyline_role to fragments.
4. Decide how each fragment is used only for this specific user prompt.
5. Separate real archive clips from generated/internet assets.
6. Keep narration cinematic but factual.
7. Return valid JSON only. No markdown. No backticks.
8. Respect evidence_gaps and must_not_claim. Never upgrade supporting/context clips into direct claims.
"""

    video_mode = director_plan.get("video_mode", "")
    if video_mode == "chronological_relay":
        system_instruction = system_instruction.rstrip() + """

TEMPORAL ARC RULES (chronological_relay mode):
- Each section corresponds to a distinct time period. The year_slot for each section is provided in the director_plan.
- Every section narration MUST open with a temporal anchor sentence: "In [year]..." or "By [year], ..." or "As early as [year]...".
- Do NOT invent years — use only the year_slot value provided for that section.
- The narration must reflect the intellectual/historical progression: context → problem → risk → solution → synthesis.
- Voice-over bridges between sections must reference the time gap: "Three years later...", "By the following decade...", etc.
- If year_slot is null for a section, omit the temporal anchor and write a thematic bridge instead.
"""
        relay_plan = dict(director_plan)
        relay_structure = relay_plan.get("recommended_structure") or []
        relay_plan["recommended_structure"] = relay_structure
        director_plan = relay_plan

    # Compute word budget so the LLM writes narration proportional to the target duration.
    # Cards take ~10s; remaining content split equally across sections. ~2.5 words/sec for TTS.
    _num_sections = timeline_segment_count_for_duration(duration)
    _content_budget_s = max(1, duration - 10)
    _per_section_s = round(_content_budget_s / _num_sections)
    _words_per_section = max(30, round(_per_section_s * 2.5))

    user_instruction = {
        "user_prompt": user_prompt,
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "narration_guidance": (
            f"The video is {duration} seconds long with {_num_sections} sections. "
            f"Each section has roughly {_per_section_s} seconds of content. "
            f"Write each section's narration/voiceover as approximately {_words_per_section} words "
            f"(at ~2.5 words/second TTS pace). "
            f"IMPORTANT: Do NOT write short narrations. You MUST write at least {_words_per_section} words per section. "
            f"Fill every section with substantive, specific, evidence-grounded content that the audience can follow."
        ),
        "style": style,
        "director_plan": director_plan,
        "evidence_package": evidence_package,
        "available_candidate_fragments": candidate_pack,
        "required_output_schema": {
            "title": "string",
            "logline": "string",
            "style": "string",
            "target_duration_seconds": "number",
            "estimated_duration_seconds": "number",
            "director_plan": "copy the provided director plan and refine planning_notes only if useful",
            "narrative_arc": [
                "opening_hook",
                "context",
                "problem_or_tension",
                "human_perspective",
                "expert_evidence",
                "solution_direction",
                "final_synthesis"
            ],
            "sections": [
                {
                    "section_id": "S1",
                    "section_type": "opening_hook | context | problem_or_tension | human_perspective | expert_evidence | solution_direction | final_synthesis",
                    "target_duration_seconds": "number",
                    "director_role": "opening_context | exact_evidence_clip | related_theme_clip | related_governance_clip | synthesis_narration | closing_takeaway | evidence_gap_narration | available_evidence_clip",
                    "evidence_role": "exact | related | synthesis | low_evidence",
                    "evidence_confidence": "high | medium | low",
                    "fallback_allowed": "boolean",
                    "planning_note": "string",
                    "narration": "voice-over narration for this section",
                    "archive_visual_fragments": [
                        {
                            "fragment_id": "must come from candidates",
                            "usage": "background_visual | evidence_visual | context_visual",
                            "start": "number",
                            "end": "number",
                            "reason": "why this visual supports the narration"
                        }
                    ],
                    "speaker_soundbites": [
                        {
                            "fragment_id": "must come from candidates",
                            "start": "number",
                            "end": "number",
                            "quote_summary": "short summary of what the speaker says",
                            "reason": "why this is useful"
                        }
                    ],
                    "generated_assets_needed": [
                        {
                            "asset_type": "title_card | transition_card | map_card | data_card | abstract_ai_visual",
                            "description": "what to generate",
                            "on_screen_text": "short text if needed"
                        }
                    ],
                    "internet_assets_needed": [
                        {
                            "asset_type": "b_roll | image | map | ambient_visual",
                            "description": "what to source",
                            "search_keywords": ["keyword1", "keyword2"]
                        }
                    ]
                }
            ],
            "voiceover_script": [
                {
                    "section_id": "S1",
                    "text": "narrator line",
                    "tone": "calm documentary narrator"
                }
            ],
            "selected_speaker_soundbites": [
                {
                    "fragment_id": "must come from candidates",
                    "source_video": "string",
                    "start": "number",
                    "end": "number",
                    "quote_summary": "string",
                    "use_in_section": "section_id"
                }
            ],
            "asset_plan": {
                "real_archive_assets": ["fragment_id list"],
                "generated_assets": ["descriptions"],
                "internet_assets": ["descriptions"],
                "voiceover_required": True,
                "subtitles_required": True,
                "background_music_required": True
            },
            "export_timeline": [
                {
                    "order": "number",
                    "timeline_type": "voiceover | speaker_clip | archive_visual | generated_card | internet_broll",
                    "section_id": "string",
                    "fragment_id": "optional string",
                    "duration_seconds": "number",
                    "description": "string"
                }
            ],
            "editor_notes": [
                "string"
            ]
        }
    }

    prompt = system_instruction.strip() + "\n\nINPUT:\n" + json.dumps(user_instruction, ensure_ascii=False, indent=2)

    log(f"Calling storyline model: {STRONG_LLM_MODEL if STORYLINE_LLM_BACKEND != 'vllm' else STORYLINE_VLLM_MODEL} (backend={STORYLINE_LLM_BACKEND})")
    response_text = call_strong_llm(prompt, timeout=600, temperature=0.4, top_p=0.9, max_tokens=24000)

    parsed = extract_json_object(response_text)

    if parsed is None:
        raise RuntimeError("Could not parse JSON from strong LLM response")

    # Strip empty fragment_id values from export_timeline (LLMs often emit "" for narration/card entries)
    for item in parsed.get("export_timeline") or []:
        if isinstance(item, dict) and str(item.get("fragment_id") or "").strip() == "":
            item.pop("fragment_id", None)

    return parsed


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown fences/backticks if model adds them.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip("`").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start:end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def repair_storyline_json(raw_text: str) -> dict[str, Any] | None:
    repair_prompt = f"""
Repair the following text into valid JSON only.
Do not add markdown or explanation.

TEXT:
{raw_text}
"""

    prompt = (
        "You are a JSON repair assistant. Output only valid JSON with no markdown or explanation.\n\n"
        + raw_text
    )
    response_text = call_strong_llm(prompt, timeout=300, temperature=0.1)
    return extract_json_object(response_text)


def expand_short_narrations(
    storyline: dict[str, Any],
    target_duration_seconds: int,
) -> dict[str, Any]:
    """Expand any section narrations that are shorter than the word budget."""
    sections = storyline.get("sections") or []
    if not sections:
        return storyline

    _n = timeline_segment_count_for_duration(target_duration_seconds)
    _per_s = round(max(1, target_duration_seconds - 10) / _n)
    min_words = max(30, round(_per_s * 2.5))

    expanded = 0

    for sec in sections:
        narr = sec.get("narration", "")
        word_count = len(narr.split())
        if word_count >= min_words:
            continue

        context = {
            "section_type": sec.get("section_type", ""),
            "planning_note": sec.get("planning_note", ""),
            "keywords": sec.get("evidence_terms", [])[:6],
        }

        expand_prompt = (
            f"Expand the following documentary narration to at least {min_words} words. "
            f"Keep the same meaning and tone. Write only the expanded narration text — no labels, no quotes, no explanation.\n\n"
            f"Context: {json.dumps(context)}\n\n"
            f"Original narration:\n{narr}\n\n"
            f"Expanded narration ({min_words}+ words):"
        )

        full_prompt = (
            "You are a documentary narration writer. Write only the expanded narration "
            "text with no labels, quotes, or explanation.\n\n" + expand_prompt
        )
        try:
            expanded_text = call_strong_llm(full_prompt, timeout=120, temperature=0.3)
            if expanded_text and len(expanded_text.split()) >= min_words:
                sec["narration"] = expanded_text
                sec["viewer_narration"] = expanded_text
                expanded += 1
        except Exception:
            pass  # Keep original if expansion fails

    if expanded:
        log(f"Expanded {expanded}/{len(sections)} short narrations to ≥{min_words} words.")

    return storyline


def validate_fragment_references(storyline: dict[str, Any], valid_fragment_ids: set[str]) -> list[str]:
    issues = []

    def check_value(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "fragment_id" and nested is not None and str(nested).strip() != "":
                    if str(nested) not in valid_fragment_ids:
                        issues.append(f"Invalid fragment_id at {path}.{key}: {nested}")
                else:
                    check_value(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, nested in enumerate(value):
                check_value(nested, f"{path}[{idx}]")

    check_value(storyline, "storyline")

    for idx, item in enumerate(storyline.get("export_timeline", []) or []):
        if "fragment_id" in item:
            value = str(item.get("fragment_id") or "").strip()
            if not value:
                issues.append(f"Empty fragment_id at storyline.export_timeline[{idx}]")
            elif value not in valid_fragment_ids:
                issues.append(f"Invalid export_timeline fragment_id at storyline.export_timeline[{idx}]: {value}")

    return issues


def validate_speaker_selection(selected: list[dict[str, Any]], intent: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    requested_speaker = intent.get("requested_speaker")
    if not requested_speaker:
        return selected, []

    filtered = [item for item in selected if item.get("speaker_match") is True]
    issues: list[str] = []
    if len(filtered) != len(selected):
        issues.append("Removed non-matching speaker candidates during final validation.")
    return filtered, issues


def enrich_storyline_with_director_plan(
    storyline: dict[str, Any],
    director_plan: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(storyline)
    enriched["director_plan"] = director_plan

    mode = director_plan.get("video_mode", "narration_led_explainer")
    default_section_roles = {
        "speaker_led_documentary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "synthesis_narration", "closing_takeaway"],
        "speaker_led_theme_summary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "closing_takeaway"],
        "low_evidence_speaker_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "closing_takeaway"],
        "no_evidence": ["opening_context", "evidence_gap_narration", "closing_takeaway"],
        "relay_of_ideas": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_governance_clip", "closing_takeaway"],
        "theme_summary_explainer": ["opening_context", "related_theme_clip", "related_theme_clip", "exact_evidence_clip", "related_governance_clip", "synthesis_narration", "closing_takeaway"],
        "narration_led_explainer": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_supporting_clip", "closing_takeaway"],
        "low_evidence_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "related_context", "related_context", "synthesis_narration", "closing_takeaway"],
    }
    role_sequence = default_section_roles.get(mode, default_section_roles["narration_led_explainer"])

    sections = []
    for idx, section in enumerate(enriched.get("sections", [])):
        section_copy = dict(section)
        has_speaker = bool(section_copy.get("speaker_soundbites"))
        has_visual = bool(section_copy.get("archive_visual_fragments"))

        section_copy.setdefault("director_role", role_sequence[idx] if idx < len(role_sequence) else "related_theme_clip")
        section_copy.setdefault("evidence_role", "exact" if has_speaker else ("related" if has_visual else "low_evidence"))
        section_copy.setdefault(
            "evidence_confidence",
            "high" if has_speaker else ("medium" if has_visual else "low"),
        )
        section_copy.setdefault("fallback_allowed", section_copy.get("evidence_role") != "exact")
        section_copy.setdefault(
            "planning_note",
            "Used as related supporting evidence because exact prompt clips were limited." if section_copy.get("evidence_role") != "exact" else "Anchored by an exact or near-exact evidence clip.",
        )
        if mode == "chronological_relay":
            for frag_list_key in ("speaker_soundbites", "archive_visual_fragments"):
                frags = section_copy.get(frag_list_key)
                if frags:
                    section_copy[frag_list_key] = sorted(
                        frags,
                        key=lambda f: (f.get("video_year") is None, f.get("video_year") or 0),
                    )
            all_years = [
                f.get("video_year")
                for key in ("speaker_soundbites", "archive_visual_fragments")
                for f in (section_copy.get(key) or [])
                if f.get("video_year") is not None
            ]
            if all_years:
                sorted_years = sorted(all_years)
                mid = len(sorted_years) // 2
                section_copy["year_slot"] = sorted_years[mid]
            else:
                section_copy["year_slot"] = None

        sections.append(section_copy)

    enriched["sections"] = sections
    enriched["planning_summary"] = {
        "video_mode": director_plan.get("video_mode"),
        "evidence_strength": director_plan.get("evidence_strength"),
        "exact_clip_count": director_plan.get("exact_clip_count"),
        "related_clip_count": director_plan.get("related_clip_count"),
        "strong_related_count": director_plan.get("strong_related_count"),
        "usable_speaker_clip_count": director_plan.get("usable_speaker_clip_count"),
        "rejected_clip_count": director_plan.get("rejected_clip_count"),
        "fallback_risk": director_plan.get("fallback_risk"),
        "planning_notes": director_plan.get("planning_notes", []),
    }
    return enriched


def ensure_storyline_export_timeline(
    storyline: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
) -> dict[str, Any]:
    enriched = dict(storyline)
    timeline = list(enriched.get("export_timeline") or [])
    if not timeline and selected_candidates:
        timeline = build_export_timeline_from_candidates(
            selected_candidates=selected_candidates,
            prompt_profile=prompt_profile,
            target_duration_seconds=target_duration_seconds,
            title=enriched.get("title"),
        )
    enriched["export_timeline"] = timeline
    return enriched
