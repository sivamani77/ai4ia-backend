import argparse
import json
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.storyline_gen.candidate_scoring import annotate_candidate_relevance  # noqa: E402
from app.pipeline.storyline_gen.director import (  # noqa: E402
    bucket_candidates,
    build_director_plan,
    compute_evidence_metrics,
    get_selected_candidate_cap,
)
from app.pipeline.storyline_gen.llm_client import log, warm_up_ollama  # noqa: E402
from app.pipeline.storyline_gen.moment_repair import (  # noqa: E402
    repair_or_rebuild_timeline_with_moments,
    timeline_requires_moment_repair,
)
from app.pipeline.storyline_gen.narrative import build_documentary_arc  # noqa: E402
from app.pipeline.storyline_gen.prompt_intent import (  # noqa: E402
    detect_prompt_intent,
    load_known_speakers,
)
from app.pipeline.storyline_gen.retrieval import (  # noqa: E402
    DEFAULT_TOP_K,
    EMBEDDINGS_DIR,
    activate_corpus,
    compute_retrieval_diagnostics,
    get_filtered_records_or_load,
    retrieve_candidates,
)
from app.pipeline.storyline_gen.selection import (  # noqa: E402
    _assign_relay_sections_by_year,
    build_evidence_gap_notes,
    build_storyline_evidence_package,
    select_balanced_candidates,
    select_documentary_candidates,
)
from app.pipeline.storyline_gen.storyline_llm import (  # noqa: E402
    enrich_storyline_with_director_plan,
    expand_short_narrations,
    generate_llm_storyline,
    validate_fragment_references,
    validate_speaker_selection,
)
from app.pipeline.storyline_gen.text_sanitize import sanitize_viewer_facing_storyline_text  # noqa: E402
from app.pipeline.storyline_gen.validation_save import (  # noqa: E402
    build_fallback_storyline,
    print_summary,
    save_storyline,
    validate_storyline_before_save,
)

DEFAULT_FINAL_CANDIDATES = 35




def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentary storyline from AI4IA fragments.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt for the documentary storyline."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=180,
        help="Target video duration in seconds."
    )
    parser.add_argument(
        "--style",
        default="National Geographic-style documentary explainer",
        help="Video style."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of fragments to retrieve before balancing."
    )
    parser.add_argument(
        "--final-candidates",
        type=int,
        default=DEFAULT_FINAL_CANDIDATES,
        help="Number of fragments sent to the storyline model."
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip LLM storyline generation and use deterministic fallback."
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(EMBEDDINGS_DIR),
        help="Directory containing embedding matrix and JSONL. Default: data/embeddings/"
    )
    parser.add_argument(
        "--corpus-version",
        default="original_v1",
        help="Filter to a specific source_transcript_version (e.g. cleaned_v2, original_v1). Default: original_v1"
    )
    parser.add_argument(
        "--min-transcript-quality",
        type=float,
        default=0.0,
        help="Exclude fragments with transcript_quality_score below this threshold. Default: 0.0"
    )
    parser.add_argument(
        "--exclude-manual-review",
        action="store_true",
        help="Exclude fragments with audio_quality_status == 'manual_review'."
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Backend job id, when launched by the pipeline. Used to scope the "
             "output storyline filename so concurrent jobs can't pick up each "
             "other's file (see save_storyline())."
    )

    args = parser.parse_args()

    log("Starting Step 07: Generate Documentary Storyline")
    log(f"Prompt: {args.prompt}")
    log(f"Duration: {args.duration}")
    log(f"Style: {args.style}")
    log(f"Top K: {args.top_k}")
    log(f"Final candidates: {args.final_candidates}")

    corpus_diagnostics = activate_corpus(
        embeddings_dir=args.embeddings_dir,
        corpus_version=args.corpus_version,
        min_transcript_quality=args.min_transcript_quality,
        exclude_manual_review=args.exclude_manual_review,
    )

    # Warm up in the background — it overlaps with the CPU/embedding work below
    # (corpus scoring, candidate retrieval) instead of blocking ahead of it, so
    # the ~minutes-long model cold-load isn't wasted serial time on the critical path.
    _warm_up_thread = threading.Thread(target=warm_up_ollama, daemon=True)
    _warm_up_thread.start()

    known_speakers = load_known_speakers()
    log(f"Loaded {len(known_speakers)} known speakers.")
    all_records = get_filtered_records_or_load()
    prompt_intent = detect_prompt_intent(args.prompt, known_speakers, all_records, target_duration_seconds=args.duration)
    log(f"Prompt intent: {json.dumps(prompt_intent, ensure_ascii=False)}")

    raw_retrieved = retrieve_candidates(args.prompt, prompt_intent, top_k=args.top_k)
    retrieved = annotate_candidate_relevance(raw_retrieved, args.prompt, prompt_intent, known_speakers)

    buckets = bucket_candidates(retrieved, prompt_intent)
    selected = buckets["selected_candidates"]
    supporting = buckets["supporting_candidates"]
    context = buckets["context_candidates"]
    rejected_cands = buckets["rejected_candidates"]

    log(f"Retrieved: {len(retrieved)} | Selected: {len(selected)} | Supporting: {len(supporting)} | Context: {len(context)} | Rejected: {len(rejected_cands)}")

    selected = select_balanced_candidates(selected, limit=min(args.final_candidates, len(selected))) if selected else []
    buckets["selected_candidates"] = selected

    selected, selection_issues = validate_speaker_selection(selected, prompt_intent)
    buckets["selected_candidates"] = selected

    # Preliminary metrics to determine cap
    pre_metrics = compute_evidence_metrics(selected, supporting, context, rejected_cands)
    pre_plan = build_director_plan(pre_metrics, intent=prompt_intent)
    video_mode = pre_plan["video_mode"]
    evidence_strength = pre_plan["evidence_strength"]

    # Speaker-first, duration-aware selection. Reserves slots for real speaker
    # voices before the general relevance cap so they are never crowded out by
    # marginally-higher-scoring narration-only clips.
    cap = get_selected_candidate_cap(args.duration, video_mode, evidence_strength)
    pre_cap_count = len(selected)
    selected, speaker_selection_diag = select_documentary_candidates(
        selected,
        target_duration_seconds=args.duration,
        cap=cap,
        requested_speaker=prompt_intent.get("requested_speaker"),
    )
    if evidence_strength == "low":
        cap_reason = "evidence_strength=low"
    elif video_mode in ("low_evidence_brief", "low_evidence_speaker_brief", "no_evidence"):
        cap_reason = f"video_mode={video_mode}"
    else:
        cap_reason = f"target_duration_seconds={args.duration}"
    buckets["selected_candidates"] = selected
    log(
        f"Speaker-first selection: {speaker_selection_diag['speaker_eligible_in_pool']} eligible in pool → "
        f"{speaker_selection_diag['speaker_slots_reserved']} reserved "
        f"(target {speaker_selection_diag['speaker_target']}, "
        f"{speaker_selection_diag['speakers_per_section']}/section × ~{speaker_selection_diag['estimated_sections']} sections), "
        f"{speaker_selection_diag['fill_slots_used']} fill slots"
    )

    candidate_bucket_summary = {
        "selected": len(selected),
        "supporting": len(supporting),
        "context": len(context),
        "rejected": len(rejected_cands),
        "selected_cap": cap,
        "pre_cap_count": pre_cap_count,
        "cap_reason": cap_reason,
        "speaker_selection": speaker_selection_diag,
    }
    log(f"Cap applied: {pre_cap_count} → {len(selected)} (cap={cap}, reason={cap_reason})")

    # Recompute metrics on capped selected
    evidence_metrics = compute_evidence_metrics(selected, supporting, context, rejected_cands)
    director_plan = build_director_plan(evidence_metrics, intent=prompt_intent)
    retrieval_diagnostics = compute_retrieval_diagnostics(prompt_intent, retrieved, buckets, args.top_k)

    # Build evidence-gap notes and append to planning notes
    prompt_terms = set(prompt_intent.get("topic_terms") or [])
    gap_notes = build_evidence_gap_notes(prompt_intent, prompt_terms, evidence_metrics)
    existing_notes = director_plan.get("planning_notes", [])
    merged_notes = existing_notes[:]
    for note in gap_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    director_plan["planning_notes"] = merged_notes
    evidence_package = build_storyline_evidence_package(prompt_intent, buckets)

    valid_fragment_ids = {item.get("fragment_id") for item in selected}

    # For chronological_relay: pre-assign real fragments (grouped by year) to each
    # recommended section BEFORE the LLM call so the LLM receives concrete year_slot
    # values and actual clips to narrate around. Without this the LLM receives
    # year_slot=null for every section and writes narrations with no temporal anchors.
    if video_mode == "chronological_relay":
        relay_structure = director_plan.get("recommended_structure") or []
        if relay_structure:
            director_plan["recommended_structure"] = _assign_relay_sections_by_year(
                relay_structure, selected
            )
            log(f"chronological_relay: pre-assigned year slots to {len(relay_structure)} sections from {len(set(c.get('video_year') for c in selected if c.get('video_year') is not None))} distinct years")

    log(f"Selected candidates after cap: {len(selected)}")
    log(f"Evidence: exact={evidence_metrics['exact_clip_count']} strong={evidence_metrics['strong_related_count']} supporting={evidence_metrics.get('supporting_candidate_count', 0)} usable_speaker={evidence_metrics['usable_speaker_clip_count']}")

    storyline = None
    generation_mode = "llm"
    llm_parse_error = None
    fallback_reason = None

    if not args.fallback_only:
        if _warm_up_thread.is_alive():
            log("Waiting for Ollama warm-up to finish before the storyline call...")
        _warm_up_thread.join(timeout=590)

        try:
            storyline = generate_llm_storyline(
                user_prompt=args.prompt,
                duration=args.duration,
                candidates=selected,
                style=args.style,
                director_plan=director_plan,
                prompt_intent=prompt_intent,
                evidence_package=evidence_package,
            )
            storyline["generation_mode"] = "llm"
        except RuntimeError as exc:
            error_msg = str(exc)
            if "Could not parse JSON" in error_msg or "parse" in error_msg.lower():
                log(f"LLM JSON parse failed: {exc}")
                log("Falling back to evidence-grounded fallback (json parse failure).")
                generation_mode = "llm_json_parse_failed"
                llm_parse_error = error_msg
                fallback_reason = "llm_json_parse_failed_used_evidence_grounded_fallback"
            else:
                log(f"Strong LLM storyline failed: {exc}")
                generation_mode = "llm_failed"
                llm_parse_error = error_msg
                fallback_reason = "llm_request_failed"
        except Exception as exc:
            log(f"Strong LLM storyline failed: {exc}")
            generation_mode = "llm_failed"
            llm_parse_error = str(exc)
            fallback_reason = "llm_request_failed"

    if storyline is None:
        log(f"Using evidence-grounded fallback (mode={generation_mode}, reason={fallback_reason}).")
        storyline = build_fallback_storyline(
            user_prompt=args.prompt,
            duration=args.duration,
            candidates=selected,
            style=args.style,
            director_plan=director_plan,
            prompt_intent=prompt_intent,
            evidence_package=evidence_package,
        )
        if generation_mode in ("llm_json_parse_failed", "llm_failed"):
            storyline["generation_mode"] = generation_mode
        if llm_parse_error:
            storyline["llm_parse_error"] = llm_parse_error
        if fallback_reason:
            storyline["fallback_reason"] = fallback_reason

    # Phase 4O.3: If LLM returned no sections (or malformed sections), rebuild from
    # build_documentary_arc so that Step 08 can generate voiceover narration.
    if not storyline.get("sections") and selected:
        log("Sections missing from LLM output — rebuilding from build_documentary_arc.")
        arc = build_documentary_arc(
            prompt_profile=prompt_intent,
            selected_candidates=selected,
            target_duration_seconds=args.duration,
        )
        rebuilt_sections = arc.get("sections") or []
        # Stamp mandatory section fields
        for idx, sec in enumerate(rebuilt_sections, start=1):
            sec.setdefault("section_id", f"S{idx}")
            role = sec.get("role", "")
            sec.setdefault("section_type", role or f"section_{idx}")
            sec.setdefault("narration", sec.get("viewer_narration", ""))
            sec.setdefault("viewer_narration", sec.get("narration", ""))
        storyline["sections"] = rebuilt_sections
        storyline.setdefault("central_question", arc.get("central_question", ""))
        storyline.setdefault("narrative_thesis", arc.get("narrative_thesis", ""))
        storyline.setdefault("closing_meaning", arc.get("closing_meaning", ""))
        log(f"Rebuilt {len(rebuilt_sections)} sections from documentary arc.")

    # Backfill evidence_fragment_ids, claim_type, evidence_terms into sections that lack them
    for section in (storyline.get("sections") or []):
        if not section.get("evidence_fragment_ids"):
            frag_ids = []
            for frag in (section.get("archive_visual_fragments") or []):
                fid = frag.get("fragment_id")
                if fid:
                    frag_ids.append(fid)
            for sb in (section.get("speaker_soundbites") or []):
                fid = sb.get("fragment_id")
                if fid:
                    frag_ids.append(fid)
            section["evidence_fragment_ids"] = list(dict.fromkeys(frag_ids))
        if not section.get("claim_type"):
            has_evidence = bool(section.get("evidence_fragment_ids") or section.get("archive_visual_fragments") or section.get("speaker_soundbites"))
            section_type_val = section.get("section_type", "")
            if section_type_val == "evidence_gap_disclosure":
                section["claim_type"] = "evidence_gap"
            elif has_evidence:
                section["claim_type"] = "direct_evidence"
            else:
                section["claim_type"] = "synthesis"
        if not section.get("evidence_terms"):
            section_frags = [c for c in selected if c.get("fragment_id") in (section.get("evidence_fragment_ids") or [])]
            ev_terms = list(dict.fromkeys(
                t for c in section_frags
                for t in (c.get("direct_evidence_terms") or []) + (c.get("expanded_evidence_terms") or [])
            ))[:6]
            section["evidence_terms"] = ev_terms

    storyline = enrich_storyline_with_director_plan(storyline, director_plan)

    # Expand narrations that are shorter than the word budget
    storyline = expand_short_narrations(storyline, args.duration)

    # Sanitize all viewer-facing text before validation and save
    storyline, sanitize_notes = sanitize_viewer_facing_storyline_text(storyline, prompt_intent)
    if sanitize_notes:
        unrepairable = [n for n in sanitize_notes if n.startswith("viewer_facing_text_unrepairable")]
        repaired = [n for n in sanitize_notes if n.startswith("repaired:")]
        if repaired:
            log(f"Sanitized {len(repaired)} viewer-facing text field(s) of internal pipeline language.")
        if unrepairable:
            log(f"Warning: {len(unrepairable)} viewer-facing field(s) could not be fully repaired.")
            for note in unrepairable[:5]:
                log(f"  {note}")

    # Phase 4O: rebuild or repair the timeline so every fragment item has moment fields.
    # This replaces raw LLM timeline items unconditionally.
    storyline = repair_or_rebuild_timeline_with_moments(
        storyline=storyline,
        selected_candidates=selected,
        prompt_profile=prompt_intent,
        target_duration_seconds=args.duration,
        video_mode=director_plan.get("video_mode"),
    )

    # Strict pre-save validation (runs after sanitization + timeline repair)
    strict_issues = validate_storyline_before_save(
        storyline=storyline,
        prompt_profile=prompt_intent,
        selected_candidates=selected,
        video_mode=director_plan.get("video_mode", ""),
        evidence_strength=director_plan.get("evidence_strength", ""),
    )

    # Hard check: if any fragment item still lacks moment fields after repair, flag it
    final_timeline = storyline.get("export_timeline") or []
    if selected and timeline_requires_moment_repair(final_timeline):
        strict_issues.append("timeline_moment_repair_failed")

    if selected and not final_timeline:
        strict_issues.append("selected_candidates_present_but_export_timeline_empty")

    validation_issues = selection_issues + validate_fragment_references(storyline, valid_fragment_ids) + strict_issues

    if validation_issues:
        log("Validation issues found:")
        for issue in validation_issues[:20]:
            log(f"- {issue}")

    output_path = save_storyline(
        storyline=storyline,
        user_prompt=args.prompt,
        prompt_intent=prompt_intent,
        duration=args.duration,
        style=args.style,
        retrieved=retrieved,
        buckets=buckets,
        director_plan=director_plan,
        evidence_metrics=evidence_metrics,
        validation_issues=validation_issues,
        evidence_package=evidence_package,
        retrieval_diagnostics=retrieval_diagnostics,
        candidate_bucket_summary=candidate_bucket_summary,
        corpus_diagnostics=corpus_diagnostics,
        job_id=args.job_id,
    )

    print_summary(storyline, output_path)


if __name__ == "__main__":
    main()
