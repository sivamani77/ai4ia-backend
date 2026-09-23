"""
Step 07b — LLM Editorial Validation Layer

Runs after Step 07 (generate_storyline) and before Step 08 (generate_assets).

For each candidate fragment:
  1. Load full transcript for source video.
  2. Extract 45s surrounding context window.
  3. Ask LLM to validate the fragment against the documentary narrative.
  4. Collect approved candidates with expanded quote windows.

Then run a single full-narrative assembly call to build the final timeline.

Output: validated storyline JSON with `final_timeline` and `validated_narrative` keys.
Step 08 uses `final_timeline` directly when present.

Editorial logic lives in app.pipeline.validate_gen — this script is just the
CLI entry point (argument parsing + top-level orchestration).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    DURATION_FILL_MARGIN,
    _candidate_is_speaker_eligible,
    build_duration_budget,
    fill_timeline_to_target,
    timeline_total_seconds,
)
from app.pipeline.validate_gen.fallback_timeline import build_fallback_final_timeline  # noqa: E402
from app.pipeline.validate_gen.fragment_validation import (  # noqa: E402
    _base_section_id,
    _clean_voiceover_text,
    fallback_validation,
    validate_candidate_with_llm_async,
)
from app.pipeline.validate_gen.llm_client import (  # noqa: E402
    STORYLINE_LLM_BACKEND,
    STORYLINE_VLLM_MODEL,
    STRONG_LLM_MODEL,
    log,
)
from app.pipeline.validate_gen.narrative_assembly import assemble_narrative_with_llm  # noqa: E402
from app.pipeline.validate_gen.narrative_templates import (  # noqa: E402
    DEEPFAKE_TOPIC_TERMS,
    DEEPFAKE_VOICEOVER_DEFAULTS,
    GENERIC_VOICEOVER_DEFAULTS,
)
from app.pipeline.validate_gen.quality_gates import (  # noqa: E402
    CRITICAL_GATE_NAMES,
    EXTRA_SPEAKER_CHECKS,
    MAX_VALIDATION_ITERATIONS,
    SPEAKER_CLIP_TARGET,
    _add_visual_asset_plan,
    _critical_gates_pass,
    _enforce_section_rhythm,
    _fill_missing_sections,
    _llm_repair_timeline,
    _llm_vet_topup_items,
    _validate_speaker_quotes,
    check_quality_gates,
    frame_duration_fill_as_perspectives_section,
    repair_can_help,
    repair_is_better,
    repair_shelf,
    speaker_clip_target,
    unchecked_speaker_candidates,
)
from app.pipeline.validate_gen.timeline_enrichment import (  # noqa: E402
    # _build_grounding_material / _narration_grounding_issues aren't called
    # from this script directly (only from generate_voiceovers_for_timeline
    # below) — imported anyway so they stay reachable as module attributes,
    # since tests load this file directly via importlib and call them on it.
    _build_grounding_material,
    _narration_grounding_issues,
    assign_fragments_to_empty_narration_brolls,
    canonicalize_speaker_names,
    enrich_final_timeline,
    generate_voiceovers_for_timeline,
)
from app.pipeline.validate_gen.topic_detection import detect_narrative_sections  # noqa: E402
from app.services.speaker_registry import get_registry  # noqa: E402
from app.pipeline.validate_gen.transcript import (  # noqa: E402
    extract_context_window,
    find_transcript_file,
    load_transcript_segments,
)

STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM editorial validation layer for AI4IA storylines.")
    parser.add_argument("--storyline", required=True, help="Path to input storyline JSON.")
    parser.add_argument(
        "--transcripts-dir",
        default=str(PROJECT_ROOT / "data" / "transcripts_cleaned_aligned"),
        help="Primary transcripts directory.",
    )
    parser.add_argument(
        "--fallback-transcripts-dir",
        default=str(PROJECT_ROOT / "data" / "transcripts"),
        help="Fallback transcripts directory.",
    )
    parser.add_argument("--output", required=True, help="Path to write validated storyline JSON.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls, use score-based fallback for all decisions.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_VALIDATION_ITERATIONS,
        help="Maximum LLM repair iterations for the narrative validation loop.",
    )
    args = parser.parse_args()

    storyline_path = Path(args.storyline).resolve()
    transcripts_dir = Path(args.transcripts_dir).resolve()
    fallback_dir = Path(args.fallback_transcripts_dir).resolve()
    output_path = Path(args.output).resolve()

    log("Starting Step 07b: LLM Editorial Validation")
    log(f"Input:  {storyline_path}")
    log(f"Output: {output_path}")

    # Load storyline
    data = json.loads(storyline_path.read_text(encoding="utf-8"))
    user_prompt = data.get("user_prompt") or ""
    target_duration = int(data.get("target_duration_seconds") or 120)
    prompt_profile = data.get("prompt_profile") or data.get("prompt_intent") or {}
    topic_terms = prompt_profile.get("topic_terms") or []

    export_timeline = data.get("export_timeline") or []
    selected_candidates = data.get("selected_candidates") or []

    # Build candidate lookup by fragment_id
    candidate_map: dict[str, dict] = {}
    for c in selected_candidates:
        fid = c.get("fragment_id")
        if fid:
            candidate_map[str(fid)] = c

    # Merge export_timeline items with candidate data
    fragment_items = [
        item for item in export_timeline
        if item.get("fragment_id") and item.get("type") not in ("title_card", "closing_card")
    ]

    # Detect narrative structure
    narrative_sections = detect_narrative_sections(user_prompt, topic_terms, target_duration)
    log(f"Narrative structure: {[s['section_id'] for s in narrative_sections]}")
    log(f"Fragment items to validate: {len(fragment_items)}")

    # Validate each fragment (concurrent — each LLM call is independent).
    # Transcript/context lookup is local disk I/O and stays synchronous; only
    # the LLM round trip is awaited, so the semaphore purely throttles Ollama load.
    def _prepare_fragment(i: int, item: dict) -> tuple[dict, list[dict]]:
        fid = str(item.get("fragment_id") or "")
        source_video = item.get("source_video") or ""
        start = float(item.get("start") or 0)
        end = float(item.get("end") or start + 10)

        log(f"\n[{i+1}] Preparing: {fid[:60]!r}")
        log(f"  source_video: {source_video[:60]!r}  window: {start:.1f}→{end:.1f}s")

        cand_data = candidate_map.get(fid, {})
        merged = {**cand_data, **item}
        merged["fragment_id"] = fid
        merged["source_video"] = source_video
        merged["start"] = start
        merged["end"] = end

        transcript_path = find_transcript_file(source_video, transcripts_dir, fallback_dir)
        context_window: list[dict] = []
        if transcript_path:
            log(f"  transcript: {transcript_path.name}")
            segs = load_transcript_segments(transcript_path)
            context_window = extract_context_window(segs, start, end)
            log(f"  context window: {len(context_window)} segments ({len(context_window)} around {start:.1f}–{end:.1f}s)")
        else:
            log(f"  transcript: NOT FOUND for {source_video[:50]!r}")
        return merged, context_window

    def _finalize_result(
        i: int,
        merged: dict,
        result: dict | None,
    ) -> tuple[int, dict, bool]:
        used_fallback = False
        if result is None:
            used_fallback = True
            log(f"  WARNING: LLM validation failed — using score-based fallback.")
            result = fallback_validation(merged)
            result["_fallback_reason"] = "llm_failed"
            log(f"  fallback decision: {result['decision']} | approved={result['approved']}")
        else:
            log(f"  LLM decision: {result.get('decision')} | approved={result.get('approved')} | section={result.get('section_id')} | confidence={result.get('confidence', 0):.2f}")
            if not result.get("approved"):
                log(f"  reason: {result.get('reason', '')}")

        result["source_video"] = merged["source_video"]
        result["start"] = merged["start"]
        result["end"] = merged["end"]
        result["text"] = merged.get("text") or merged.get("moment_text") or ""
        return i, result, used_fallback

    def _validate_items(items: list[dict]) -> list[tuple[dict, bool]]:
        """(result, used_fallback) for each item, in order."""
        raw_results: dict[int, tuple[dict, bool]] = {}
        if args.skip_llm:
            for i, item in enumerate(items):
                merged, _ = _prepare_fragment(i, item)
                result = fallback_validation(merged)
                log(f"  fallback decision: {result['decision']} | approved={result['approved']}")
                result["source_video"] = merged["source_video"]
                result["start"] = merged["start"]
                result["end"] = merged["end"]
                result["text"] = merged.get("text") or merged.get("moment_text") or ""
                raw_results[i] = (result, False)
            return [raw_results[i] for i in sorted(raw_results)]

        # Concurrency cap: local Ollama can serialize past this — 6 chosen as a
        # doubling of the prior thread-based cap of 3; tune down if Ollama logs
        # queuing/timeouts under load.
        max_concurrency = min(6, max(1, len(items)))

        async def _validate_all_async() -> None:
            semaphore = asyncio.Semaphore(max_concurrency)
            async with httpx.AsyncClient() as client:
                async def _run_one(i: int, item: dict) -> None:
                    merged, context_window = _prepare_fragment(i, item)
                    result = await validate_candidate_with_llm_async(
                        candidate=merged,
                        context_window=context_window,
                        narrative_sections=narrative_sections,
                        user_prompt=user_prompt,
                        target_duration=target_duration,
                        client=client,
                        semaphore=semaphore,
                    )
                    idx, final_result, used_fallback = _finalize_result(i, merged, result)
                    raw_results[idx] = (final_result, used_fallback)

                await asyncio.gather(*(
                    _run_one(i, item) for i, item in enumerate(items)
                ))

        asyncio.run(_validate_all_async())
        return [raw_results[i] for i in sorted(raw_results)]

    results = _validate_items(fragment_items)

    # Too few speaker clips approved: check more of stage 07's shortlist.
    approved_speakers = sum(1 for r, _ in results if r.get("approved") and r.get("decision") == "speaker_clip")
    if approved_speakers < SPEAKER_CLIP_TARGET:
        extras = unchecked_speaker_candidates(
            selected_candidates + (data.get("supporting_candidates") or []),
            checked_fids={str(item.get("fragment_id")) for item in fragment_items},
            limit=EXTRA_SPEAKER_CHECKS,
        )
        if extras:
            log(f"\nOnly {approved_speakers} speaker clip(s) approved — checking {len(extras)} more from the shortlist.")
            for c in extras:
                candidate_map.setdefault(str(c["fragment_id"]), c)
            results += _validate_items(extras)

    validated_candidates: list[dict] = [r for r, _ in results]
    llm_fallback_count = sum(1 for _, used_fallback in results if used_fallback)

    approved_count = sum(1 for c in validated_candidates if c.get("approved"))
    rejected_count = len(validated_candidates) - approved_count
    log(f"\nValidation complete: {approved_count} approved, {rejected_count} rejected out of {len(validated_candidates)}")
    if llm_fallback_count:
        log(f"WARNING: {llm_fallback_count}/{len(validated_candidates)} fragments used score-based fallback due to LLM failures — results may be lower quality.")
    speaker_target = speaker_clip_target(validated_candidates)
    log(f"Speaker clip target: {speaker_target}")

    # Full narrative assembly
    log("\nRunning full narrative assembly...")
    assembled: dict | None = None

    if not args.skip_llm:
        assembled = assemble_narrative_with_llm(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            target_duration=target_duration,
            storyline_data=data,
        )

    if assembled is None:
        log("  LLM assembly unavailable — using score-based fallback.")
        assembled = build_fallback_final_timeline(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            storyline_data=data,
            target_duration=target_duration,
        )

    # Detect topic for defaults — only match actual deepfake terms in the prompt,
    # never match topic_terms (they come from the prompt and would always match)
    is_deepfake_topic = any(
        term.lower() in user_prompt.lower()
        for term in DEEPFAKE_TOPIC_TERMS
    )

    # Build validated_map keyed by fragment_id for speaker resolution
    validated_map: dict[str, dict] = {}
    for vc in validated_candidates:
        fid = vc.get("fragment_id")
        if fid:
            validated_map[str(fid)] = vc

    # Enrich final_timeline with required fields
    raw_timeline = assembled.get("final_timeline") or []
    if not raw_timeline:
        log("  WARNING: assembled final_timeline is empty — using fallback.")
        assembled = build_fallback_final_timeline(
            validated_candidates=validated_candidates,
            narrative_sections=narrative_sections,
            storyline_data=data,
            target_duration=target_duration,
        )
        raw_timeline = assembled.get("final_timeline") or []

    enriched_timeline = enrich_final_timeline(
        final_timeline=raw_timeline,
        candidate_map=candidate_map,
        original_timeline=export_timeline,
        validated_map=validated_map,
        is_deepfake_topic=is_deepfake_topic,
    )

    # Assign fragments to narration_broll items that have no fragment_id
    log("\nAssigning fragments to empty narration_broll items...")
    enriched_timeline = assign_fragments_to_empty_narration_brolls(
        final_timeline=enriched_timeline,
        validated_candidates=validated_candidates,
        original_timeline=export_timeline,
    )

    # Generate voiceover text for narration_broll items with empty voiceover_text
    if not args.skip_llm:
        log("\nGenerating voiceover texts for narration_broll items...")
        enriched_timeline = generate_voiceovers_for_timeline(
            final_timeline=enriched_timeline,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            validated_candidates=validated_candidates,
            target_duration=target_duration,
            storyline_data=data,
        )
    else:
        defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        for item in enriched_timeline:
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll" and not (item.get("voiceover_text") or "").strip():
                sid = item.get("section_id") or ""
                item["voiceover_text"] = defaults.get(_base_section_id(sid), "")
                item["subtitle_text"] = item["voiceover_text"]

    # Ensure all narrative sections are represented in the timeline
    enriched_timeline = _fill_missing_sections(
        final_timeline=enriched_timeline,
        narrative_sections=narrative_sections,
        validated_candidates=validated_candidates,
        original_timeline=export_timeline,
    )

    # Validate speaker quote completeness
    enriched_timeline = _validate_speaker_quotes(enriched_timeline)

    # Clean remaining instruction leakage from voiceover texts
    defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
    for item in enriched_timeline:
        t = item.get("type") or item.get("segment_type") or ""
        if t == "narration_broll":
            sid = item.get("section_id") or ""
            raw_vo = item.get("voiceover_text") or ""
            cleaned = _clean_voiceover_text(raw_vo, sid, is_deepfake_topic)
            if not cleaned:
                cleaned = defaults.get(_base_section_id(sid), "")
            item["voiceover_text"] = cleaned
            item["subtitle_text"] = cleaned

    # Enforce section rhythm: S1 → speaker, all other sections → narration then speaker
    enriched_timeline = _enforce_section_rhythm(enriched_timeline, narrative_sections)

    # Add visual asset plan to all narration_broll items
    enriched_timeline = _add_visual_asset_plan(enriched_timeline, narrative_sections, is_deepfake_topic)

    assembled["final_timeline"] = enriched_timeline

    # Iterative quality-gate repair loop
    max_iters = args.max_iterations
    warnings, quality_gates = check_quality_gates(enriched_timeline, narrative_sections, speaker_target)
    repair_iter = 0
    for repair_iter in range(1, max_iters + 1):
        if _critical_gates_pass(quality_gates):
            log(f"\nAll critical gates pass (iteration {repair_iter}).")
            break
        if args.skip_llm:
            log(f"\nLLM skipped — not repairing critical gate failures.")
            break
        if repair_iter >= max_iters:
            log(f"\nMax repair iterations ({max_iters}) reached. Proceeding with current timeline.")
            break
        failed = [k for k, v in quality_gates.items() if isinstance(v, bool) and not v and k in CRITICAL_GATE_NAMES]
        log(f"\nCritical gate failures on iteration {repair_iter}: {failed}")
        shelf = repair_shelf(enriched_timeline, validated_candidates)
        if not repair_can_help(quality_gates, shelf):
            log("  [repair] Failures need speaker clips that were not approved — keeping current timeline.")
            break
        before = (enriched_timeline, quality_gates, warnings)
        log("Sending repair prompt to LLM...")
        repaired_tl = _llm_repair_timeline(
            current_timeline=enriched_timeline,
            quality_gates=quality_gates,
            narrative_sections=narrative_sections,
            user_prompt=user_prompt,
            iteration=repair_iter,
            target_duration=target_duration,
            shelf=shelf,
            speaker_target=speaker_target,
        )
        if not repaired_tl:
            log("  [repair] LLM returned nothing — keeping current timeline.")
            break
        log(f"  [repair] LLM returned {len(repaired_tl)} items — re-enriching...")
        enriched_timeline = enrich_final_timeline(
            final_timeline=repaired_tl,
            candidate_map=candidate_map,
            original_timeline=export_timeline,
            validated_map=validated_map,
            is_deepfake_topic=is_deepfake_topic,
        )
        enriched_timeline = assign_fragments_to_empty_narration_brolls(
            enriched_timeline, validated_candidates, export_timeline
        )
        if not args.skip_llm:
            enriched_timeline = generate_voiceovers_for_timeline(
                enriched_timeline, narrative_sections, user_prompt, validated_candidates,
                target_duration=target_duration, storyline_data=data,
            )
        enriched_timeline = _fill_missing_sections(
            enriched_timeline, narrative_sections, validated_candidates, export_timeline
        )
        enriched_timeline = _validate_speaker_quotes(enriched_timeline)
        _defaults = DEEPFAKE_VOICEOVER_DEFAULTS if is_deepfake_topic else GENERIC_VOICEOVER_DEFAULTS
        for _item in enriched_timeline:
            _t = _item.get("type") or _item.get("segment_type") or ""
            if _t == "narration_broll":
                _sid = _item.get("section_id") or ""
                _raw_vo = _item.get("voiceover_text") or ""
                _cleaned = _clean_voiceover_text(_raw_vo, _sid, is_deepfake_topic)
                if not _cleaned:
                    _cleaned = _defaults.get(_base_section_id(_sid), "")
                _item["voiceover_text"] = _cleaned
                _item["subtitle_text"] = _cleaned
        enriched_timeline = _enforce_section_rhythm(enriched_timeline, narrative_sections)
        enriched_timeline = _add_visual_asset_plan(enriched_timeline, narrative_sections, is_deepfake_topic)
        warnings, quality_gates = check_quality_gates(enriched_timeline, narrative_sections, speaker_target)
        if not repair_is_better(before[:2], (enriched_timeline, quality_gates)):
            log("  [repair] Rejected: it would lose speaker clips or pass fewer gates — keeping the previous timeline.")
            enriched_timeline, quality_gates, warnings = before
            break
    else:
        repair_iter = max_iters

    if warnings:
        log("\nQuality gate results:")
        for w in warnings:
            log(f"  {w}")
    else:
        log("\nQuality gates: all passed.")
    assembled.setdefault("warnings", [])
    assembled["warnings"].extend(warnings)

    # ------------------------------------------------------------------
    # Duration floor: validation may have trimmed or rejected clips — repay
    # the shortfall with the next ranked real candidates before locking.
    # ------------------------------------------------------------------
    duration_fill_report: dict | None = None
    speaker_clip_count = sum(1 for item in enriched_timeline if (item.get("type") or item.get("segment_type")) == "speaker_clip")
    if target_duration or speaker_clip_count == 0:
        floor = float(target_duration or 0) * DURATION_FILL_MARGIN
        current_total = timeline_total_seconds(enriched_timeline)
        if current_total < floor or speaker_clip_count == 0:
            used_fids = {str(i.get("fragment_id")) for i in enriched_timeline if i.get("fragment_id")}
            approved_pool = [
                c for c in (validated_candidates or [])
                if c.get("approved") and str(c.get("fragment_id") or "") not in used_fids
            ]
            # Widen the pool beyond the LLM-approved/supporting/context sets: also
            # pull speaker-eligible fragments that Step 07 rejected purely on
            # relevance rank/cap grounds (not disqualified). "All eligible
            # speakers" for this topic means the full known pool, not just the
            # subset that survived earlier trimming.
            rejected_speaker_pool = [
                c for c in (data.get("rejected_candidates") or [])
                if _candidate_is_speaker_eligible(c)
                and str(c.get("fragment_id") or "") not in used_fids
            ]
            extended_pool = (
                (data.get("supporting_candidates") or [])
                + (data.get("context_candidates") or [])
                + rejected_speaker_pool
            )
            log(
                f"\nDuration top-up: validated timeline is {current_total:.1f}s "
                f"(floor {floor:.1f}s) — pulling additional ranked candidates."
            )
            for _topup_round in range(3):
                duration_fill_report = fill_timeline_to_target(
                    enriched_timeline,
                    {"selected": approved_pool, "supporting": extended_pool},
                    target_duration,
                )
                unvetted = [
                    i for i in enriched_timeline
                    if i.get("added_by_duration_fill") and i.get("needs_validation")
                ]
                if args.skip_llm or not unvetted:
                    break
                rejected_fids = _llm_vet_topup_items(unvetted, user_prompt)
                if not rejected_fids:
                    break
                log(f"  [topup] LLM rejected {len(rejected_fids)} off-topic top-up item(s) — refilling.")
                enriched_timeline[:] = [
                    i for i in enriched_timeline
                    if str(i.get("fragment_id") or "") not in rejected_fids
                ]
                extended_pool = [
                    c for c in extended_pool
                    if str(c.get("fragment_id") or "") not in rejected_fids
                ]
            for item in enriched_timeline:
                if item.get("added_by_duration_fill"):
                    item["needs_validation"] = False
            if duration_fill_report:
                log(
                    f"Duration top-up: {duration_fill_report['status']} — "
                    f"{duration_fill_report['total_before_seconds']:.1f}s → "
                    f"{duration_fill_report['total_after_seconds']:.1f}s"
                )

            n_framed = frame_duration_fill_as_perspectives_section(enriched_timeline)
            if n_framed:
                log(
                    f"Duration top-up: framed {n_framed} added speaker clip(s) as an "
                    f"'Additional Perspectives' closing section."
                )
            assembled["final_timeline"] = enriched_timeline

    # Re-verify quality gates after duration top-up / rescue net
    warnings, quality_gates = check_quality_gates(enriched_timeline, narrative_sections, speaker_target)

    # Registry spellings everywhere, whatever the prompt or the LLM wrote.
    enriched_timeline = canonicalize_speaker_names(enriched_timeline)
    for key in ("title", "logline"):
        if assembled.get(key):
            assembled[key] = get_registry().fix_names_in_text(assembled[key])

    # Lock every validated clip window: downstream stages (08/09) render these
    # verbatim and may only extend narration for TTS, never shrink anything.
    for item in enriched_timeline:
        if not item.get("fragment_id"):
            continue
        item["window_locked"] = True
        seg_t = item.get("type") or item.get("segment_type") or ""
        if seg_t == "speaker_clip" and not item.get("render_start"):
            if item.get("start") is not None and item.get("end") is not None:
                item["render_start"] = float(item["start"])
                item["render_end"] = float(item["end"])
                item["render_duration"] = round(item["render_end"] - item["render_start"], 3)

    validated_total_seconds = timeline_total_seconds(enriched_timeline)

    # Print timeline summary
    log("\n=== Validated Final Timeline ===")
    sc_count = quality_gates["speaker_clip_count"]
    nb_count = quality_gates["narration_broll_count"]
    log(f"  total items:     {len(enriched_timeline)}")
    log(f"  speaker_clips:   {sc_count}")
    log(f"  narration_broll: {nb_count}")
    log(f"  quality_score:   {assembled.get('editorial_quality_score', 0):.2f}")
    log(f"  gates: sections={quality_gates['all_sections_covered']}  "
        f"speakers={quality_gates['all_speaker_labels_present']}  "
        f"quotes={quality_gates['all_speaker_quotes_complete']}  "
        f"voiceover_clean={quality_gates['no_instruction_leakage_in_voiceover']}  "
        f"approved={quality_gates['final_timeline_approved']}")

    for i, item in enumerate(enriched_timeline, 1):
        t = item.get("type") or "?"
        sid = item.get("section_id") or ""
        fid = (item.get("fragment_id") or "")[:50]
        speaker = item.get("speaker") or ""
        rs = item.get("render_start") or item.get("start") or ""
        re_ = item.get("render_end") or item.get("end") or ""
        qc = item.get("quote_complete")
        log(f"  [{i}] {t:<18} {sid:<14} {fid}")
        if t == "speaker_clip":
            log(f"       speaker={speaker!r}  {rs}→{re_}  quote_complete={qc}")
            text = (item.get("quote_text") or item.get("subtitle_text") or "")[:120]
            log(f"       quote: {text!r}")
        elif t == "narration_broll":
            vo = (item.get("voiceover_text") or "")[:120]
            log(f"       vo: {vo!r}")

    # Write output — copy original data, add validated fields
    output_data = dict(data)
    output_data["llm_validated"] = True
    output_data["validation_run_at"] = datetime.utcnow().isoformat()
    _effective_strong_model = STORYLINE_VLLM_MODEL if STORYLINE_LLM_BACKEND == "vllm" else STRONG_LLM_MODEL
    output_data["validated_with"] = _effective_strong_model if not args.skip_llm else "fallback"
    output_data["repair_iterations"] = repair_iter
    output_data["final_timeline"] = enriched_timeline
    output_data["editorial_quality_gates"] = quality_gates
    output_data["validated_narrative"] = {
        "title": assembled.get("title"),
        "logline": assembled.get("logline"),
        "narrative_arc": assembled.get("narrative_arc") or [],
        "final_timeline": enriched_timeline,
        "rejected_candidates": assembled.get("rejected_candidates") or [],
        "editorial_quality_score": assembled.get("editorial_quality_score", 0),
        "editorial_quality_gates": quality_gates,
        "warnings": assembled.get("warnings") or [],
    }
    # Also merge into storyline sub-object so Step 08 can find it either way
    output_data["storyline"]["final_timeline"] = enriched_timeline
    output_data["storyline"]["validated_narrative"] = output_data["validated_narrative"]
    output_data["duration_budget"] = build_duration_budget(
        enriched_timeline,
        target_duration,
        stage="07b_validate_storyline",
    )
    output_data["validated_total_seconds"] = validated_total_seconds
    if duration_fill_report:
        output_data["duration_fill_report"] = duration_fill_report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nValidated storyline written: {output_path}")

    # Below-floor after an honest top-up means the relevant speaker pool is
    # exhausted — ship the best-effort duration with a warning, never fail the
    # job or pad with filler. (Integrity shortfalls — dropped/missing content —
    # are still hard-failed downstream in stage 09; this is purely the
    # "not enough relevant material exists" case.)
    if target_duration and validated_total_seconds < float(target_duration):
        log(
            f"WARNING: validated timeline is {validated_total_seconds:.1f}s vs {target_duration}s "
            f"requested (shortfall {float(target_duration) - validated_total_seconds:.1f}s). "
            f"Relevant content exhausted — delivering honest best-effort duration, no filler."
        )

    log("Step 07b complete.")


if __name__ == "__main__":
    main()
