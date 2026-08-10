import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.paths import PROJECT_ROOT
from app.pipeline.timeline_budget import (
    build_duration_budget,
    clamp_timeline_windows_to_source,
    fill_timeline_to_target,
    timeline_segment_count_for_duration,
)
from app.pipeline.storyline_gen.llm_client import (
    EMBED_MODEL,
    FAST_LLM_MODEL,
    STORYLINE_LLM_BACKEND,
    STORYLINE_VLLM_MODEL,
    STRONG_LLM_MODEL,
    log,
)
from app.pipeline.storyline_gen.moment_repair import _window_intro_penalty
from app.pipeline.storyline_gen.narrative import (
    CINEMATIC_HOOKS,
    _make_cinematic_title,
    build_documentary_arc,
    build_evidence_grounded_narration,
    detect_generic_fallback_narration,
    validate_narration_prompt_alignment,
)
from app.pipeline.storyline_gen import retrieval
from app.pipeline.storyline_gen.selection import build_candidate_pack
from app.pipeline.storyline_gen.text_sanitize import validate_viewer_facing_text
from app.pipeline.storyline_gen.text_utils import (
    SIDE_TOPIC_TERMS,
    parse_list_field,
    truncate_text,
)


STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"

STORYLINES_DIR.mkdir(parents=True, exist_ok=True)


STRICT_NO_GENERIC_FALLBACK = True


def validate_storyline_before_save(
    storyline: dict,
    prompt_profile: dict,
    selected_candidates: list[dict],
    video_mode: str,
    evidence_strength: str,
) -> list[str]:
    """Strict pre-save validation. Returns list of validation issues."""
    issues = []
    if not STRICT_NO_GENERIC_FALLBACK:
        return issues

    sections = storyline.get("sections") or []
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    all_narration = " ".join(s.get("narration", "") for s in sections)

    generic_issues = detect_generic_fallback_narration(all_narration, prompt_profile)
    if generic_issues and selected_candidates:
        issues.extend(generic_issues)

    if selected_candidates and all_narration.strip():
        alignment_issues = validate_narration_prompt_alignment(all_narration, prompt_profile, selected_candidates)
        issues.extend(alignment_issues)

    pipeline_issues = validate_viewer_facing_text(storyline)
    issues.extend(pipeline_issues)

    side_topic_lower = [t.lower() for t in SIDE_TOPIC_TERMS]
    for section in sections:
        sid = section.get("section_id", "?")
        narration = (section.get("narration") or "").lower()
        for term in side_topic_lower:
            if term in narration:
                issues.append(f"side_topic_leakage_detected: '{term}' in section {sid} narration")
                break

    allowed_no_evidence = {"title_card", "closing_card", "evidence_gap_disclosure", "synthesis_narration", "final_synthesis"}
    allowed_claim_types_no_evidence = {"synthesis", "evidence_gap"}
    for section in sections:
        section_type = section.get("section_type", "")
        section_id = section.get("section_id", "?")
        claim_type = section.get("claim_type", "")
        if section_type in allowed_no_evidence:
            continue
        if claim_type in allowed_claim_types_no_evidence:
            continue
        has_frags = bool(
            section.get("evidence_fragment_ids")
            or section.get("archive_visual_fragments")
            or section.get("speaker_soundbites")
        )
        if not has_frags and selected_candidates:
            issues.append(f"section_missing_evidence_links: {section_id} ({section_type})")

    if len(selected_candidates) == 0 and video_mode not in (
        "low_evidence_brief", "low_evidence_speaker_brief", "no_evidence", "low_evidence_disclosure"
    ):
        issues.append(f"low_evidence_video_mode_mismatch: no selected_candidates but video_mode={video_mode}")

    # Phase 4O: moment-level validation on export_timeline
    export_timeline = storyline.get("export_timeline") or []
    for titem in export_timeline:
        if titem.get("type") in ("title_card", "closing_card"):
            continue
        fid = titem.get("fragment_id")
        seg_type = titem.get("type", "")
        if not fid:
            continue

        # speaker_clip must have moment_text
        if seg_type == "speaker_clip" and not titem.get("moment_text"):
            issues.append(f"selected_clip_missing_moment_text: {fid}")

        # speaker_clip must not have intro language
        moment_text = (titem.get("moment_text") or "").lower()
        if seg_type == "speaker_clip" and moment_text:
            intro_pen = _window_intro_penalty(moment_text)
            if intro_pen >= 0.35:
                issues.append(f"speaker_clip_has_intro_language: {fid}")

        # speaker_clip must not have low speech ratio
        speech_q = titem.get("speech_quality") or {}
        if seg_type == "speaker_clip" and speech_q:
            if not speech_q.get("is_usable_speech", True):
                issues.append(f"speaker_clip_has_low_speech_ratio: {fid}")

        # archive_visual should be muted
        if seg_type == "archive_visual" and titem.get("muted") is False:
            issues.append(f"archive_visual_not_muted: {fid}")

        # subtitle_text must not use full fragment text for speaker clips
        subtitle = titem.get("subtitle_text", "")
        if seg_type == "speaker_clip" and subtitle:
            # Warn if subtitle is > 3× the moment_text length (likely full-fragment text)
            moment = titem.get("moment_text", "")
            if moment and len(subtitle) > len(moment) * 3:
                issues.append(f"speaker_subtitle_uses_full_fragment_text: {fid}")

        # moment_selection_reason should be present
        if not titem.get("moment_selection_reason"):
            issues.append(f"moment_selection_missing_reason: {fid}")

    return issues


def build_fallback_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
    evidence_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_speaker = prompt_intent.get("requested_speaker")
    topic_terms = prompt_intent.get("topic_terms") or []
    evidence_gaps = (evidence_package or {}).get("evidence_gaps") or []
    must_not_claim = (evidence_package or {}).get("must_not_claim") or []
    topic_label = " ".join(topic_terms[:3]).title() if topic_terms else (requested_speaker or "AI")

    mode_to_roles = {
        "speaker_led_documentary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "synthesis_narration", "closing_takeaway"],
        "speaker_led_theme_summary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "closing_takeaway"],
        "low_evidence_speaker_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "closing_takeaway"],
        "no_evidence": ["opening_context", "evidence_gap_narration", "closing_takeaway"],
        "relay_of_ideas": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_governance_clip", "closing_takeaway"],
        "theme_summary_explainer": ["opening_context", "related_theme_clip", "related_theme_clip", "exact_evidence_clip", "related_governance_clip", "synthesis_narration", "closing_takeaway"],
        "narration_led_explainer": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_supporting_clip", "closing_takeaway"],
        "low_evidence_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "related_context", "related_context", "synthesis_narration", "closing_takeaway"],
    }
    _base_roles = mode_to_roles.get(director_plan.get("video_mode"), mode_to_roles["narration_led_explainer"])
    # Extend role_sequence to cover any number of sections by cycling the middle roles
    _middle = _base_roles[1:-1] if len(_base_roles) > 2 else _base_roles
    _needed = timeline_segment_count_for_duration(duration) + 4  # generous headroom
    role_sequence = [_base_roles[0]] + (_middle * ((_needed // max(1, len(_middle))) + 1))[:_needed] + [_base_roles[-1]]

    # No candidates: low_evidence_disclosure only
    if not candidates:
        if requested_speaker:
            disclosure_text = (
                f"The AI4IA archive does not contain direct clips from {requested_speaker}. "
                f"No documentary video was generated because using unrelated clips would be misleading."
            )
        elif "crypto" in set(topic_terms):
            disclosure_text = (
                f"The AI4IA archive does not contain direct evidence for AI and cryptocurrency. "
                f"Related clips discuss cryptography, cybersecurity, or broader AI governance, "
                f"which should not be presented as cryptocurrency evidence."
            )
        else:
            gap_detail = " ".join(evidence_gaps[:2]) if evidence_gaps else ""
            disclosure_text = (
                f"The archive does not contain enough direct evidence for this prompt ({topic_label}). "
                f"No final documentary video was generated because using unrelated clips would be misleading. "
                + gap_detail
            )
            if must_not_claim:
                disclosure_text += " " + must_not_claim[0]

        return {
            "title": f"Evidence Gap: {topic_label}",
            "logline": disclosure_text,
            "style": style,
            "target_duration_seconds": duration,
            "estimated_duration_seconds": 0,
            "generation_mode": "low_evidence_disclosure",
            "fallback_reason": "no_selected_candidates",
            "director_plan": director_plan,
            "narrative_arc": ["evidence_gap_disclosure"],
            "sections": [
                {
                    "section_id": "S1",
                    "section_type": "evidence_gap_disclosure",
                    "target_duration_seconds": 15,
                    "director_role": "evidence_gap_narration",
                    "evidence_role": "evidence_gap",
                    "evidence_confidence": "none",
                    "fallback_allowed": True,
                    "planning_note": "No archive evidence found; disclosure only.",
                    "narration": disclosure_text,
                    "evidence_fragment_ids": [],
                    "source_videos": [],
                    "evidence_terms": [],
                    "claim_type": "evidence_gap",
                    "archive_visual_fragments": [],
                    "speaker_soundbites": [],
                    "generated_assets_needed": [],
                    "internet_assets_needed": [],
                }
            ],
            "voiceover_script": [{"section_id": "S1", "text": disclosure_text, "tone": "calm informational narrator"}],
            "selected_speaker_soundbites": [],
            "asset_plan": {
                "real_archive_assets": [],
                "generated_assets": [],
                "internet_assets": [],
                "voiceover_required": True,
                "subtitles_required": False,
                "background_music_required": False,
            },
            "export_timeline": [],
            "editor_notes": [
                "low_evidence_disclosure: no archive evidence found for this prompt.",
                "should_export_video: false",
            ],
            "planning_summary": {
                "video_mode": director_plan.get("video_mode"),
                "evidence_strength": director_plan.get("evidence_strength"),
                "exact_clip_count": 0,
                "related_clip_count": 0,
                "fallback_risk": "high",
                "planning_notes": director_plan.get("planning_notes", []),
            },
            "should_export_video": False,
        }

    # Candidates exist: build evidence-grounded narration
    soundbites_cap = max(8, timeline_segment_count_for_duration(duration))
    soundbites = [
        item for item in candidates
        if "speaker_soundbite" in parse_list_field(item.get("clip_usage_candidates"))
    ][:soundbites_cap]
    visuals = candidates[:max(18, timeline_segment_count_for_duration(duration) * 2)]

    grounded = build_evidence_grounded_narration(
        prompt_profile=prompt_intent,
        selected_candidates=candidates,
        evidence_package=evidence_package or {},
        target_duration_seconds=duration,
    )
    grounded_sections = grounded.get("sections") or []

    section_types_sequence = [
        "opening_hook", "context", "problem_or_tension",
        "human_perspective", "expert_evidence", "solution_direction", "final_synthesis",
        # extended roles for longer videos (repeats cycle gracefully)
        "context", "human_perspective", "expert_evidence",
        "problem_or_tension", "solution_direction", "context",
        "expert_evidence", "human_perspective", "solution_direction",
        "context", "expert_evidence", "final_synthesis",
    ]
    built_sections = []

    for idx, gs in enumerate(grounded_sections):
        section_id = gs.get("section_id", f"S{idx + 1}")
        section_type = gs.get("section_type") or (section_types_sequence[idx] if idx < len(section_types_sequence) else "final_synthesis")
        narration = gs.get("narration", "")
        evidence_fragment_ids = gs.get("evidence_fragment_ids") or []
        source_videos = gs.get("source_videos") or []
        evidence_terms = gs.get("evidence_terms") or []
        claim_type = gs.get("claim_type", "direct_evidence")

        section_visuals = [v for v in visuals if v.get("fragment_id") in evidence_fragment_ids] or visuals[idx * 2: idx * 2 + 2]
        section_soundbites = soundbites[idx: idx + 1]
        section_role = role_sequence[idx] if idx < len(role_sequence) else "related_theme_clip"

        if claim_type == "evidence_gap":
            evidence_role = "evidence_gap"
            evidence_confidence = "none"
        elif section_soundbites:
            evidence_role = "exact"
            evidence_confidence = "high"
        elif section_visuals:
            evidence_role = "related"
            evidence_confidence = "medium"
        else:
            evidence_role = "low_evidence"
            evidence_confidence = "low"

        built_sections.append({
            "section_id": section_id,
            "section_type": section_type,
            "target_duration_seconds": round(duration / max(1, len(grounded_sections))),
            "director_role": section_role,
            "evidence_role": evidence_role,
            "evidence_confidence": evidence_confidence,
            "fallback_allowed": evidence_role != "exact",
            "planning_note": (
                f"Evidence-grounded narration from {len(evidence_fragment_ids)} fragment(s)."
                if evidence_fragment_ids else "Evidence-grounded narration; disclosure section."
            ),
            "narration": narration,
            "evidence_fragment_ids": evidence_fragment_ids,
            "source_videos": source_videos,
            "evidence_terms": evidence_terms,
            "claim_type": claim_type,
            "archive_visual_fragments": [
                {
                    "fragment_id": item["fragment_id"],
                    "usage": "evidence_visual",
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "reason": item.get("summary", "Evidence-grounded archive visual."),
                }
                for item in section_visuals
            ],
            "speaker_soundbites": [
                {
                    "fragment_id": item["fragment_id"],
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "quote_summary": item.get("summary"),
                    "reason": "Evidence-matched speaker soundbite.",
                }
                for item in section_soundbites
            ],
            "generated_assets_needed": [],
            "internet_assets_needed": [],
        })

    doc_arc = build_documentary_arc(prompt_intent, candidates, duration)

    if requested_speaker:
        title = f"{requested_speaker}: Archive Evidence"
        logline = doc_arc.get("narrative_thesis") or f"A documentary tracing {requested_speaker}'s contributions to AI4IA."
    elif topic_label:
        topic_term_0 = (topic_terms[:1] or [None])[0]
        hook = CINEMATIC_HOOKS.get(topic_term_0 or "", "")
        raw_prompt = prompt_intent.get("raw_prompt") or prompt_intent.get("query") or ""
        title = _make_cinematic_title(topic_terms, topic_label, raw_prompt=raw_prompt)
        logline = hook or doc_arc.get("narrative_thesis") or f"A documentary on {title}."
    else:
        title = "Artificial Intelligence and the Human Question"
        logline = doc_arc.get("narrative_thesis") or "A documentary on artificial intelligence and society."

    return {
        "title": title,
        "logline": logline,
        "closing_meaning": doc_arc.get("closing_meaning", ""),
        "central_question": doc_arc.get("central_question", ""),
        "style": style,
        "target_duration_seconds": duration,
        "estimated_duration_seconds": duration,
        "generation_mode": grounded.get("generation_mode", "evidence_grounded_fallback"),
        "fallback_reason": grounded.get("fallback_reason", "llm_unavailable_or_failed"),
        "director_plan": director_plan,
        "narrative_arc": [s["section_type"] for s in built_sections],
        "sections": built_sections,
        "voiceover_script": [
            {"section_id": s["section_id"], "text": s["narration"], "tone": "observational documentary narrator"}
            for s in built_sections if s.get("narration")
        ],
        "selected_speaker_soundbites": [
            {
                "fragment_id": item["fragment_id"],
                "source_video": item.get("source_video"),
                "start": item.get("start"),
                "end": item.get("end"),
                "quote_summary": item.get("summary"),
                "use_in_section": f"S{idx + 1}",
            }
            for idx, item in enumerate(soundbites[:7])
        ],
        "asset_plan": {
            "real_archive_assets": [item["fragment_id"] for item in visuals[:18]],
            "generated_assets": [],
            "internet_assets": [],
            "voiceover_required": True,
            "subtitles_required": True,
            "background_music_required": True,
        },
        "export_timeline": [],
        "editor_notes": [
            f"evidence_grounded_fallback: LLM narration not available; documentary arc built from {len(candidates)} selected recordings.",
            f"generation_mode: {grounded.get('generation_mode')}",
        ],
        "planning_summary": {
            "video_mode": director_plan.get("video_mode"),
            "evidence_strength": director_plan.get("evidence_strength"),
            "exact_clip_count": director_plan.get("exact_clip_count"),
            "related_clip_count": director_plan.get("related_clip_count"),
            "fallback_risk": director_plan.get("fallback_risk"),
            "planning_notes": director_plan.get("planning_notes", []),
        },
        "should_export_video": len(candidates) > 0,
    }


def save_storyline(
    storyline: dict[str, Any],
    user_prompt: str,
    prompt_intent: dict[str, Any],
    duration: int,
    style: str,
    retrieved: list[dict[str, Any]],
    buckets: dict[str, list[dict[str, Any]]],
    director_plan: dict[str, Any],
    evidence_metrics: dict[str, Any],
    validation_issues: list[str],
    evidence_package: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
    candidate_bucket_summary: dict[str, Any] | None = None,
    corpus_diagnostics: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # job_id-scoped filename when available (always true when launched by the
    # backend pipeline) — without it, two jobs' storyline files can collide on
    # this same-second timestamp, and the caller's "newest file in the
    # directory" lookup (backend/app/pipeline/base.py::_newest_file_after)
    # would then pick up the WRONG job's storyline under concurrency > 1.
    # Falls back to the old timestamp-only name for manual/no-job-id CLI runs.
    filename = f"storyline_{job_id}_{timestamp}.json" if job_id else f"storyline_{timestamp}.json"
    output_path = STORYLINES_DIR / filename

    selected = buckets.get("selected_candidates", [])
    supporting = buckets.get("supporting_candidates", [])
    context = buckets.get("context_candidates", [])
    rejected = buckets.get("rejected_candidates", [])

    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "user_prompt": user_prompt,
        "prompt_profile": prompt_intent,
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "style": style,
        "corpus_version": retrieval._ACTIVE_CORPUS_VERSION,
        "embeddings_dir": str(retrieval._ACTIVE_EMBEDDINGS_DIR),
        "corpus_diagnostics": corpus_diagnostics or retrieval._CORPUS_DIAGNOSTICS,
        "generation_mode": storyline.get("generation_mode", "unknown"),
        "llm_parse_error": storyline.get("llm_parse_error"),
        "fallback_reason": storyline.get("fallback_reason"),
        "should_export_video": storyline.get("should_export_video", True),
        "models": {
            "embedding_model": EMBED_MODEL,
            "strong_llm_model": STORYLINE_VLLM_MODEL if STORYLINE_LLM_BACKEND == "vllm" else STRONG_LLM_MODEL,
            "fast_llm_model": FAST_LLM_MODEL,
        },
        "source_files": {
            "embedding_matrix": str(retrieval._ACTIVE_MATRIX_PATH),
            "embedding_records": str(retrieval._ACTIVE_JSONL_PATH),
        },
        "retrieval_summary": {
            "retrieved_count": len(retrieved),
            "selected_candidate_count": len(selected),
            "supporting_candidate_count": len(supporting),
            "context_candidate_count": len(context),
            "rejected_candidate_count": len(rejected),
            "top_semantic_score": retrieved[0]["semantic_score"] if retrieved else None,
        },
        "retrieval_diagnostics": retrieval_diagnostics,
        "candidate_bucket_summary": candidate_bucket_summary or {
            "selected": len(selected),
            "supporting": len(supporting),
            "context": len(context),
            "rejected": len(rejected),
        },
        "candidate_contract": {
            "selected_candidates": "final-video-safe evidence only",
            "supporting_candidates": "narration support only — not for speaker soundbites",
            "context_candidates": "background/context only — not for final video",
            "rejected_candidates": "not for final video",
        },
        "planning_summary": {
            "video_mode": director_plan.get("video_mode"),
            "evidence_strength": director_plan.get("evidence_strength"),
            "exact_clip_count": evidence_metrics.get("exact_clip_count"),
            "related_clip_count": evidence_metrics.get("related_clip_count"),
            "strong_related_count": evidence_metrics.get("strong_related_count"),
            "usable_speaker_clip_count": evidence_metrics.get("usable_speaker_clip_count"),
            "usable_speaker_visual_count": evidence_metrics.get("usable_speaker_visual_count", 0),
            "supporting_candidate_count": evidence_metrics.get("supporting_candidate_count"),
            "context_candidate_count": evidence_metrics.get("contextual_count"),
            "rejected_clip_count": evidence_metrics.get("rejected_clip_count"),
            "fallback_risk": director_plan.get("fallback_risk"),
            "planning_notes": director_plan.get("planning_notes", []),
        },
        "validation_issues": validation_issues,
        "timeline_repair": storyline.get("timeline_repair", {"required": False}),
        "evidence_package": evidence_package,
        "export_timeline": storyline.get("export_timeline", []),
        "selected_candidates": build_candidate_pack(selected),
        "supporting_candidates": build_candidate_pack(supporting[:10]),
        "context_candidates": build_candidate_pack(context[:5]),
        "rejected_candidates": build_candidate_pack(rejected[:5]),
        "debug_retrieved_candidates": build_candidate_pack(retrieved[:25]),
        "storyline": storyline,
    }
    # Duration floor: clamp planned windows to real footage, then fill with more
    # ranked real content until the plan reaches the target (floor semantics).
    export_tl = storyline.get("export_timeline") or []
    fill_report: dict[str, Any] | None = None
    if export_tl and duration:
        clamp_notes = clamp_timeline_windows_to_source(export_tl)
        for note in clamp_notes:
            log(f"  [clamp] {note}")
        fill_report = fill_timeline_to_target(
            export_tl,
            {"selected": selected, "supporting": supporting, "context": context},
            duration,
        )
        storyline["export_timeline"] = export_tl
        payload["export_timeline"] = export_tl
        payload["window_clamp_notes"] = clamp_notes
        payload["duration_fill_report"] = fill_report
        log(
            f"Duration fill: {fill_report['status']} — "
            f"{fill_report['total_before_seconds']:.1f}s → {fill_report['total_after_seconds']:.1f}s "
            f"(floor {fill_report['floor_seconds']:.1f}s, +{len(fill_report['segments_added'])} segments)"
        )

    payload["duration_budget"] = build_duration_budget(
        storyline.get("export_timeline") or [],
        duration,
        stage="07_generate_storyline",
    )

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Content-exhaustion is NOT a failure: when the genuinely relevant speaker
    # pool can't fill the requested duration, we ship the honest best-effort
    # length with a warning rather than padding with filler or failing the job.
    if fill_report and fill_report["status"] == "content_exhausted":
        log(
            f"WARNING: relevant content exhausted — {fill_report['total_after_seconds']:.1f}s of "
            f"genuinely relevant material for a {duration}s target "
            f"(shortfall {fill_report['shortfall_seconds']:.1f}s). "
            f"Delivering the honest best-effort duration; no filler added."
        )

    return output_path


def print_summary(storyline: dict[str, Any], output_path: Path) -> None:
    log("")
    log("Storyline generated")
    log(f"Output: {output_path}")

    title = storyline.get("title", "Untitled")
    log(f"Title: {title}")
    director_plan = storyline.get("director_plan", {})
    if director_plan:
        log(f"Video mode: {director_plan.get('video_mode')} | evidence: {director_plan.get('evidence_strength')} | exact={director_plan.get('exact_clip_count')} related={director_plan.get('related_clip_count')}")

    sections = storyline.get("sections", [])
    log(f"Sections: {len(sections)}")

    for section in sections:
        section_id = section.get("section_id")
        section_type = section.get("section_type")
        duration = section.get("target_duration_seconds")
        narration = truncate_text(section.get("narration", ""), 120)
        log(f"- {section_id} | {section_type} | {duration}s | {narration}")
