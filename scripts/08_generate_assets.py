"""
Step 08 — Generate Refined Cinematic Assets

Runs after Step 07b (LLM editorial validation) and before Step 09 (export video).

Turns a validated storyline into the final asset set: rendered title/section/
closing/year-stamp cards, narration visual assets, TTS voice-over audio, a
voice-over script text file, and the export-timeline/asset-manifest JSON
consumed by scripts/09_export_video.py.

Asset generation logic lives in app.pipeline.assets_gen — this script is just
the CLI entry point (argument parsing + top-level orchestration).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    build_duration_budget,
    clamp_timeline_windows_to_source,
)
from app.pipeline.assets_gen.utils import log, slugify  # noqa: E402
from app.pipeline.assets_gen.storyline_io import (  # noqa: E402
    FRAGMENTS_CLASSIFIED_DIR,
    SPEAKERS_METADATA_DIR,
    STORYLINES_DIR,
    _is_validated_storyline_payload,
    derive_source_video_from_fragment_id,
    enrich_clip_candidate,
    enrich_fragment_refs,
    latest_storyline_path,
    load_classified_fragment_lookup,
    load_speaker_lookup,
    load_storyline,
)
from app.pipeline.assets_gen.text_copy import (  # noqa: E402
    DISALLOWED_VOICEOVER_PHRASES,
    DOCUMENTARY_CHAPTER_TITLES,
    GENERIC_CARD_TITLES,
    INTERNAL_PIPELINE_TERMS_08,
    SECTION_HEADLINES,
    detect_generic_card_title,
    detect_generic_voiceover,
    detect_internal_pipeline_language_08,
    infer_section_copy,
    make_display_title,
    repair_internal_pipeline_language_08,
    sanitize_visible_text,
)
from app.pipeline.assets_gen.cards import (  # noqa: E402
    ACCENT_COLOR,
    BACKGROUND_COLOR,
    CARD_HEIGHT,
    CARD_WIDTH,
    CINEMATIC_BACKGROUND_ASSET,
    DARK_OVERLAY,
    GOLD_ACCENT,
    MUTED_TEXT_COLOR,
    PANEL_COLOR,
    TEXT_COLOR,
    VISIBLE_BRAND_LABEL,
    VISIBLE_SERIES_LABEL,
    _render_cinematic_background,
    add_cinematic_frame,
    create_cinematic_background,
    create_closing_card,
    create_narration_visual_asset,
    create_section_card,
    create_title_card,
    create_year_stamp_card,
    draw_wrapped_text,
    find_font,
    generate_visual_assets_for_timeline,
)
from app.pipeline.assets_gen.voiceover import (  # noqa: E402
    build_voiceover_script,
    generate_tts,
    get_media_duration,
    reconcile_narration_duration,
    voiceover_text_for_tts,
    write_voiceover_script_txt,
)
from app.pipeline.assets_gen.export_timeline import (  # noqa: E402
    CLIP_EXPAND_SECONDS,
    INTRO_ADMIN_TERMS_08,
    MOMENT_FIELDS_FROM_STORYLINE,
    POST_ROLL_SECONDS,
    PRE_ROLL_SECONDS,
    QUOTE_COMPLETE_THRESHOLD,
    _detect_intro_admin_in_moment,
    _speaker_clip_quality_score,
    _text_ends_complete,
    _text_starts_mid_sentence,
    build_export_timeline_from_candidates,
    build_hybrid_export_timeline,
    collect_archive_fragments,
    collect_generated_assets,
    collect_internet_assets,
    compute_quote_render_bounds,
    dedupe_manifest_fragments,
    enrich_timeline_with_moment_fields,
)
from app.pipeline.assets_gen.manifest_validation import (  # noqa: E402
    STRICT_NO_GENERIC_FALLBACK,
    preflight_audit_manifest,
    validate_asset_manifest_before_save,
)

GENERATED_ASSETS_DIR = Path(os.environ.get("AI4IA_LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "data" / "generated_assets")))

DEFAULT_VOICE = "en-GB-RyanNeural"
DEFAULT_RATE = "-6%"
DEFAULT_VOLUME = "+0%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined cinematic assets from latest AI4IA storyline.")
    parser.add_argument(
        "--storyline",
        default=None,
        help="Path to storyline JSON. Defaults to latest file in data/storylines."
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="edge-tts voice name."
    )
    parser.add_argument(
        "--rate",
        default=DEFAULT_RATE,
        help="TTS rate, for example -6%% or +0%%."
    )
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME,
        help="TTS volume, for example +0%%."
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Generate cards and manifest only, skip voice-over audio."
    )

    args = parser.parse_args()

    storyline_path = Path(args.storyline) if args.storyline else latest_storyline_path()
    storyline_path = storyline_path.resolve()

    log("Starting Step 08: Generate Refined Cinematic Assets")
    log(f"Storyline: {storyline_path}")

    payload = load_storyline(storyline_path)
    story = payload["storyline"]
    story["selected_candidates"] = payload.get("selected_candidates") or []
    prompt_profile = payload.get("prompt_profile") or payload.get("prompt_intent") or {}

    # Phase 4V: if the storyline was validated by 07b, use final_timeline directly.
    # final_timeline lives at top level of the JSON (and also inside story after 07b).
    _validated_final_timeline = (
        payload.get("final_timeline")
        or story.get("final_timeline")
        or (payload.get("validated_narrative") or {}).get("final_timeline")
        or (story.get("validated_narrative") or {}).get("final_timeline")
    )
    _is_validated_storyline = bool(_validated_final_timeline) and payload.get("llm_validated", False)

    if _is_validated_storyline:
        log("Validated storyline detected — using final_timeline from 07b.")
        story["export_timeline"] = _validated_final_timeline
    elif not (story.get("export_timeline") or []) and payload.get("selected_candidates"):
        story["export_timeline"] = build_export_timeline_from_candidates(
            selected_candidates=story["selected_candidates"],
            prompt_profile=prompt_profile,
            target_duration_seconds=int(story.get("target_duration_seconds") or payload.get("target_duration_seconds") or 60),
            title=story.get("title"),
        )

    # Phase 4V: for validated storylines, override title/logline from validated_narrative.
    _validated_narrative = payload.get("validated_narrative") or story.get("validated_narrative") or {}
    if _is_validated_storyline and _validated_narrative.get("title"):
        story["title"] = _validated_narrative["title"]
    if _is_validated_storyline and _validated_narrative.get("logline"):
        story["logline"] = _validated_narrative["logline"]

    title = make_display_title(story.get("title", "AI4IA Documentary"))
    logline = sanitize_visible_text(story.get("logline", ""))
    internal_style = story.get("style", payload.get("style", ""))

    storyline_id = storyline_path.stem
    asset_root = GENERATED_ASSETS_DIR / storyline_id
    cards_dir = asset_root / "cards"
    audio_dir = asset_root / "audio"
    section_audio_dir = audio_dir / "sections"

    asset_root.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    section_audio_dir.mkdir(parents=True, exist_ok=True)

    if _is_validated_storyline:
        # Phase 4V: extract voiceover script from narration_broll items in final_timeline.
        voiceover_script = []
        for idx, item in enumerate(_validated_final_timeline or []):
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll":
                vo_text = item.get("voiceover_text") or item.get("subtitle_text") or ""
                sid = item.get("section_id") or f"S{idx}"
                if vo_text:
                    voiceover_script.append({
                        "section_id": sid,
                        "section_type": t,
                        "text": vo_text,
                        "_timeline_index": idx,
                    })
        log(f"Validated voiceover script: {len(voiceover_script)} narration_broll items")
    else:
        voiceover_script = build_voiceover_script(story)
    full_voiceover_text = voiceover_text_for_tts(voiceover_script)

    voiceover_script_path = asset_root / "voiceover_script.txt"
    voiceover_text_path = asset_root / "voiceover_full_text.txt"

    write_voiceover_script_txt(
        output_path=voiceover_script_path,
        title=title,
        voiceover_script=voiceover_script,
    )

    voiceover_text_path.write_text(full_voiceover_text, encoding="utf-8")

    log(f"Voice-over script written: {voiceover_script_path}")
    log(f"Voice-over full text written: {voiceover_text_path}")

    title_card_path = cards_dir / "title_card.png"
    create_title_card(
        output_path=title_card_path,
        title=title,
        logline=logline,
    )
    log(f"Title card generated: {title_card_path}")

    closing_card_path = cards_dir / "closing_card.png"
    _topic_terms_close = list(prompt_profile.get("topic_terms") or [])
    _requested_speaker_close = prompt_profile.get("requested_speaker")
    _primary_topic_close = _topic_terms_close[0] if _topic_terms_close else None
    _topic_label_close = " ".join(_topic_terms_close[:2]).title() if _topic_terms_close else None

    _cinematic_closings = {
        "healthcare": ("The Human Measure of Medicine", "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."),
        "health": ("The Human Measure of Medicine", "AI in healthcare will be judged not by speed, but by whether it makes care safer and fairer."),
        "women": ("The Measure of Progress", "Progress means women gain real agency, real voice, and real power."),
        "empowerment": ("The Measure of Progress", "Progress means women gain real agency, real voice, and real power."),
        "governance": ("The Shape of Accountability", "AI governance will be measured by who it protects and who it answers to."),
        "youth": ("What They Inherit", "The next generation will decide whether to accept or change the systems they are given."),
        "education": ("The Promise of Learning", "The future of learning is defined by whether every student can access knowledge that is true, useful, and theirs."),
        "crypto": ("The Digital Promise", "A more open financial system will only arrive through transparency, safety, and rules that protect people."),
        "accessibility": ("The Invisible Divide", "Access to information is not a given — it must be built, maintained, and defended."),
    }

    _storyline_closing = story.get("closing_meaning", "")

    if _requested_speaker_close:
        closing_headline = f"{_requested_speaker_close}: The Contribution"
        closing_subtitle = _storyline_closing or f"A documentary on {_requested_speaker_close}'s contributions to AI4IA."
    elif _primary_topic_close and _primary_topic_close in _cinematic_closings:
        closing_headline, closing_subtitle = _cinematic_closings[_primary_topic_close]
        if _storyline_closing:
            closing_subtitle = _storyline_closing
    elif _topic_label_close:
        closing_headline = f"AI and {_topic_label_close}"
        closing_subtitle = _storyline_closing or f"A documentary on AI and {_topic_label_close}."
    else:
        closing_headline = make_display_title(title)
        closing_subtitle = _storyline_closing or "A documentary from the AI4IA conference archive."
    create_closing_card(
        output_path=closing_card_path,
        title=title,
        closing_headline=closing_headline,
        closing_subtitle=closing_subtitle,
    )
    log(f"Closing card generated: {closing_card_path}")

    section_cards = []

    for section in story.get("sections", []):
        section_id = section.get("section_id", "S")
        section_type = section.get("section_type", "section")

        copy = infer_section_copy(section, prompt_profile=prompt_profile)

        section_card_path = cards_dir / f"{section_id}_{slugify(section_type)}.png"

        create_section_card(
            output_path=section_card_path,
            section_id=section_id,
            section_type=section_type,
            headline=copy["headline"],
            subtitle=copy["subtitle"],
            eyebrow=copy["eyebrow"],
        )

        section_cards.append(
            {
                "section_id": section_id,
                "section_type": section_type,
                "path": str(section_card_path),
                "eyebrow": copy["eyebrow"],
                "headline": copy["headline"],
                "subtitle": copy["subtitle"],
            }
        )

        log(f"Section card generated: {section_card_path}")

    year_stamp_cards = []
    _seen_year_stamp_years: set[int] = set()
    for item in story.get("export_timeline") or []:
        if item.get("type") == "year_stamp":
            year_val = item.get("year")
            if year_val is not None and year_val not in _seen_year_stamp_years:
                _seen_year_stamp_years.add(year_val)
                year_stamp_path = cards_dir / f"year_stamp_{year_val}.png"
                try:
                    create_year_stamp_card(year=year_val, output_path=year_stamp_path)
                    year_stamp_cards.append({"year": year_val, "path": str(year_stamp_path)})
                    log(f"Year stamp card generated: {year_stamp_path}")
                except Exception as exc:
                    log(f"WARNING: failed to render year_stamp card for {year_val}: {exc}")

    tts_status = "skipped"
    section_voiceovers = []

    if not args.skip_tts:
        if not full_voiceover_text:
            log("No voice-over text found. Skipping TTS.")
            tts_status = "no_text"
        else:
            section_jobs = []
            for item in voiceover_script:
                section_id = item.get("section_id")
                section_type = item.get("section_type", "section")
                text = item.get("text", "").strip()
                if not section_id or not text:
                    continue
                section_audio_path = section_audio_dir / f"{section_id}_{slugify(section_type)}_voiceover.mp3"
                section_jobs.append((section_id, section_type, text, section_audio_path))

            log(f"Generating {len(section_jobs)} section voice-over(s) concurrently with edge-tts voice: {args.voice}")

            async def _tts_with_log(label: str, text: str, output_path: Path) -> None:
                t0 = time.time()
                await generate_tts(
                    text=text,
                    output_path=output_path,
                    voice=args.voice,
                    rate=args.rate,
                    volume=args.volume,
                )
                log(f"  [tts] {label} done in {time.time() - t0:.1f}s")

            async def _generate_all() -> None:
                await asyncio.gather(
                    *(
                        _tts_with_log(section_id, text, section_audio_path)
                        for section_id, _, text, section_audio_path in section_jobs
                    )
                )

            _t0 = time.time()
            asyncio.run(_generate_all())
            log(f"All TTS calls finished in {time.time() - _t0:.1f}s wall-clock.")

            for section_id, section_type, text, section_audio_path in section_jobs:
                section_voiceovers.append(
                    {
                        "section_id": section_id,
                        "section_type": section_type,
                        "text": text,
                        "audio_path": str(section_audio_path),
                        "provider": "edge-tts",
                        "voice": args.voice,
                        "rate": args.rate,
                        "volume": args.volume,
                    }
                )

            tts_status = "generated"
            log(f"Section voice-overs generated: {len(section_voiceovers)}")

    fragment_lookup = load_classified_fragment_lookup()
    metadata_stats = {
        "enriched_fragment_refs": 0,
        "missing_fragment_refs": set(),
    }
    enriched_story = enrich_fragment_refs(story, fragment_lookup, metadata_stats)
    enriched_story["selected_candidates"] = story.get("selected_candidates") or []

    if not (enriched_story.get("export_timeline") or []) and payload.get("selected_candidates"):
        enriched_story["export_timeline"] = build_export_timeline_from_candidates(
            selected_candidates=enriched_story["selected_candidates"],
            prompt_profile=payload.get("prompt_profile") or payload.get("prompt_intent") or {},
            target_duration_seconds=int(enriched_story.get("target_duration_seconds") or payload.get("target_duration_seconds") or 60),
            title=enriched_story.get("title"),
        )

    if _is_validated_storyline:
        # Phase 4V: validated storyline — final_timeline already has correct types.
        # Attach audio_path from section_voiceovers to narration_broll items (by order).
        nb_idx = 0
        final_tl = enriched_story.get("export_timeline") or []
        for item in final_tl:
            t = item.get("type") or item.get("segment_type") or ""
            if t == "narration_broll" and nb_idx < len(section_voiceovers):
                sv = section_voiceovers[nb_idx]
                item["audio_path"] = sv.get("audio_path", "")
                if not item.get("subtitle_text"):
                    item["subtitle_text"] = sv.get("text", "")
                if not item.get("voiceover_text"):
                    item["voiceover_text"] = sv.get("text", "")
                item["subtitle_kind"] = "voiceover"
                reconcile_narration_duration(item, item["audio_path"])
                nb_idx += 1
            elif t == "speaker_clip" and not item.get("render_duration"):
                # Locked validated window: render exactly what validation approved.
                s_v, e_v = item.get("render_start", item.get("start")), item.get("render_end", item.get("end"))
                try:
                    if s_v is not None and e_v is not None and float(e_v) > float(s_v):
                        item["render_duration"] = round(float(e_v) - float(s_v), 3)
                except (TypeError, ValueError):
                    pass
        enriched_story["export_timeline"] = final_tl
        log(f"Phase 4V: {nb_idx} narration_broll items received audio_path from section_voiceovers.")

        # Phase 4V+: generate topic-specific PIL visual assets for narration_broll items
        visuals_dir = asset_root / "visuals"
        final_tl = generate_visual_assets_for_timeline(final_tl, visuals_dir)
        enriched_story["export_timeline"] = final_tl
        _gen_vis_count = sum(1 for x in final_tl if x.get("render_visual_path"))
        log(f"Phase 4V+: {_gen_vis_count} narration_broll visual assets generated in {visuals_dir}")
    else:
        # Phase 4O: enrich export_timeline with moment fields from the storyline
        # (without overwriting fields already present)
        storyline_timeline = payload.get("export_timeline") or enriched_story.get("export_timeline") or []
        if enriched_story.get("export_timeline"):
            enriched_story["export_timeline"] = enrich_timeline_with_moment_fields(
                export_timeline=enriched_story["export_timeline"],
                storyline_timeline=storyline_timeline,
            )

        # Phase 4S: build hybrid timeline — select best fragments as speaker_clip/source_audio,
        # convert the rest to narration_broll/ai_voiceover using available section voiceovers.
        if enriched_story.get("export_timeline"):
            vo_paths = [sv.get("audio_path", "") for sv in section_voiceovers if sv.get("audio_path")]
            # Build audio_path → voiceover text map for narration_broll subtitle coverage
            vo_path_to_text: dict[str, str] = {
                sv["audio_path"]: sv.get("text", "")
                for sv in section_voiceovers
                if sv.get("audio_path") and sv.get("text")
            }
            enriched_story["export_timeline"] = build_hybrid_export_timeline(
                export_timeline=enriched_story["export_timeline"],
                selected_candidates=enriched_story.get("selected_candidates") or payload.get("selected_candidates") or [],
                section_voiceover_paths=vo_paths,
                fragment_lookup=fragment_lookup,
                vo_path_to_text=vo_path_to_text,
            )

    archive_fragments = collect_archive_fragments(enriched_story)
    generated_assets = collect_generated_assets(enriched_story)
    internet_assets = collect_internet_assets(enriched_story)

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "storyline_path": str(storyline_path),
        "storyline_id": storyline_id,
        "asset_root": str(asset_root),
        "title": title,
        "logline": logline,
        "internal_style_direction": internal_style,
        "visible_brand_label": VISIBLE_BRAND_LABEL,
        "visible_series_label": VISIBLE_SERIES_LABEL,
        "target_duration_seconds": story.get("target_duration_seconds"),
        "estimated_duration_seconds": story.get("estimated_duration_seconds"),
        "prompt_intent": prompt_profile,
        "storyline": enriched_story,
        "director_plan": enriched_story.get("director_plan"),
        "planning_summary": enriched_story.get("planning_summary"),
        "visual_rules": {
            "do_not_show_prompt": True,
            "do_not_show_internal_style": True,
            "do_not_show_long_narration_on_cards": True,
            "cards_use_short_cinematic_copy": True,
            "section_voiceover_only": True,
        },
        "voice": {
            "provider": "edge-tts",
            "voice": args.voice,
            "rate": args.rate,
            "volume": args.volume,
            "status": tts_status,
            "script_path": str(voiceover_script_path),
            "full_text_path": str(voiceover_text_path),
            "audio_path": None,
            "section_voiceovers": section_voiceovers,
            "important_note": "Use section_voiceovers for final export."
        },
        "cards": {
            "title_card": str(title_card_path),
            "closing_card": str(closing_card_path),
            "section_cards": section_cards,
            "year_stamp_cards": year_stamp_cards,
        },
        "archive_fragments": archive_fragments,
        "generated_assets_requested_by_storyline": generated_assets,
        "internet_assets_requested_by_storyline": internet_assets,
        "metadata_enrichment": {
            "enabled": True,
            "fragment_lookup_size": len(fragment_lookup),
            "enriched_fragment_refs": metadata_stats["enriched_fragment_refs"],
            "missing_fragment_refs": sorted(metadata_stats["missing_fragment_refs"]),
            "enriched_fields": [
                "source_video",
                "source_video_file",
                "video_id",
                "video_title",
                "speaker",
                "speaker_attribution",
                "text",
                "summary",
                "keywords",
                "primary_theme",
                "primary_chapter",
                "theme_tags",
                "chapter_tags",
                "clip_usage_candidates",
                "quote_strength",
                "clarity_score",
                "narration_support_score",
            ],
        },
        "next_step": {
            "script": "scripts/09_export_video.py",
            "purpose": "Create refined cinematic export with zoom, fades, section voice-over, speaker audio, and compact subtitles."
        }
    }

    # Visual quality gates (validated storylines only)
    if _is_validated_storyline:
        _tl_for_vqg = enriched_story.get("export_timeline") or []
        _nb_items = [x for x in _tl_for_vqg if (x.get("type") or x.get("segment_type")) == "narration_broll"]
        _gen_vis = [x for x in _nb_items if x.get("render_visual_path")]
        _muted_arch = [x for x in _nb_items if not x.get("render_visual_path") and not x.get("visual_path")]
        manifest["visual_strategy"] = (
            "speaker_clip = source video + source audio. "
            "narration_broll = generated topic visual + AI voiceover."
        )
        manifest["visual_quality_gates"] = {
            "all_narration_broll_have_generated_visual": len(_gen_vis) == len(_nb_items) and len(_nb_items) > 0,
            "generated_visual_segments": len(_gen_vis),
            "muted_archive_narration_segments": len(_muted_arch),
            "total_narration_broll": len(_nb_items),
        }

    # Write hybrid timeline to all downstream-compatible keys
    hybrid_tl = enriched_story.get("export_timeline") or []
    manifest["export_timeline"] = hybrid_tl
    manifest["timeline"] = hybrid_tl

    # Re-clamp windows to real footage AFTER build_hybrid_export_timeline's post-roll
    # padding + quote expansion. Step 07 already clamped the raw start/end, but that ran
    # before this padding pushed render_end outward and before any 07b/08 fragment
    # reshuffle — so re-run it here, the last point windows change before Step 09 renders.
    # Mutates hybrid_tl (same list object as manifest["timeline"]/["export_timeline"]).
    step08_clamp_notes = clamp_timeline_windows_to_source(hybrid_tl)
    for _note in step08_clamp_notes:
        log(f"  [clamp-08] {_note}")
    manifest["window_clamp_notes_step08"] = step08_clamp_notes

    manifest["duration_budget"] = build_duration_budget(
        hybrid_tl,
        manifest.get("target_duration_seconds"),
        stage="08_generate_assets",
    )

    # Pre-flight audit: every timeline item must be renderable before Step 09 runs.
    # NOTE: this guards RENDERABILITY (missing media, unusable segments), not
    # duration. A below-target plan is content exhaustion — warn and ship the
    # honest best-effort duration; never a fatal preflight failure. (Integrity
    # shortfalls, i.e. rendered < planned, are still hard-failed in Step 09.)
    preflight_failures = preflight_audit_manifest(manifest, skip_tts=args.skip_tts)
    manifest["preflight_failures"] = preflight_failures

    _budget = manifest["duration_budget"]
    if _budget.get("meets_floor") is False:
        log(
            f"WARNING: planned timeline {_budget['planned_total_seconds']:.1f}s is below target "
            f"{_budget['target_seconds']:.1f}s (shortfall {_budget['shortfall_seconds']:.1f}s) — "
            f"relevant content exhausted, shipping honest best-effort duration, no filler."
        )

    # Strict asset manifest validation before saving
    manifest_validation_issues = validate_asset_manifest_before_save(
        manifest=manifest,
        prompt_profile=prompt_profile,
        selected_candidates=story.get("selected_candidates") or [],
    )
    if manifest_validation_issues:
        log("Asset manifest validation issues:")
        for issue in manifest_validation_issues:
            log(f"  - {issue}")
    existing_manifest_issues = manifest.get("validation_issues") or []
    manifest["validation_issues"] = existing_manifest_issues + manifest_validation_issues

    manifest_path = asset_root / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest_manifest_path = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"
    latest_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Hybrid timeline diagnostics
    sc_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "speaker_clip")
    nb_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "narration_broll")
    av_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) == "archive_visual")
    card_tl_count = sum(1 for x in hybrid_tl if (x.get("type") or x.get("segment_type")) in ("title_card", "closing_card"))
    log("")
    log("Final hybrid timeline summary:")
    log(f"  total items:                {len(hybrid_tl)}")
    log(f"  speaker_clip/source_audio:  {sc_count}")
    log(f"  narration_broll/ai_voiceover: {nb_count}")
    log(f"  archive_visual/silent:      {av_count}")
    log(f"  cards:                      {card_tl_count}")

    log("")
    log("Refined asset generation complete")
    log(f"Asset root: {asset_root}")
    log(f"Manifest: {manifest_path}")
    log(f"Latest manifest: {latest_manifest_path}")
    log(f"Voice-over status: {tts_status}")
    log(f"Section voice-overs: {len(section_voiceovers)}")
    log(f"Archive fragments listed: {len(archive_fragments)}")
    log(f"Metadata enrichment: {metadata_stats['enriched_fragment_refs']} refs enriched, {len(metadata_stats['missing_fragment_refs'])} missing")
    log(f"Section cards generated: {len(section_cards)}")
    log("Visible card text is now sanitized. Internal style/prompt text will not appear on cards.")

    if preflight_failures:
        log("")
        log(f"PRE-FLIGHT AUDIT FAILED — {len(preflight_failures)} issue(s), manifest is NOT render-safe:")
        for failure in preflight_failures:
            log(f"  ✗ {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()
