"""
Step 09 — Export Refined Final Video

Runs after Step 08 (asset generation). Takes the asset manifest, refines/
selects the export timeline, transcribes/aligns narration for burned-in
subtitles, renders each segment (cards, archive clips, narration/b-roll) via
ffmpeg, concatenates into the final video, and optionally writes an HLS
stream and subtitle files.

Export logic lives in app.pipeline.export_gen — this script is just the CLI
entry point (argument parsing + top-level orchestration).
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import build_duration_budget  # noqa: E402
from app.pipeline.media_probe import media_duration_or_raise as get_media_duration  # noqa: E402
from app.pipeline.export_gen.ffmpeg_utils import log, run_cmd, run_cmd_capture  # noqa: E402
from app.pipeline.export_gen.manifest_io import load_manifest, find_source_video, slugify  # noqa: E402
from app.pipeline.export_gen.clip_render import concat_segments  # noqa: E402
from app.pipeline.export_gen.timeline_selection import (  # noqa: E402
    build_refined_timeline,
    ensure_requested_speaker,
    get_canonical_export_timeline,
)
from app.pipeline.export_gen.subtitles import build_subtitle_artifacts, get_segment_subtitle_text, write_srt, write_vtt  # noqa: E402
from app.pipeline.export_gen.segment_pipeline import (  # noqa: E402
    ExportIntegrityError,
    HLSStreamWriter,
    _build_canonical_rendering_timeline,
    _render_timeline_items,
)

GENERATED_ASSETS_DIR = Path(os.environ.get("AI4IA_LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "data" / "generated_assets")))
EXPORTS_DIR = Path(os.environ.get("AI4IA_EXPORTS_DIR", str(PROJECT_ROOT / "data" / "exports")))
TEMP_DIR = PROJECT_ROOT / "temp" / "video_export"

LATEST_MANIFEST_PATH = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
Path("/tmp/ai4ia_pipeline/video_export").mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("/tmp/ai4ia_pipeline/video_export")


def export_video(manifest_path: Path, output_name: str | None, burn_subs: bool, hls_dir: Path | None = None) -> Path:
    _export_t0 = time.time()
    manifest = load_manifest(manifest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_slug = slugify(manifest.get("title", "ai4ia_documentary"), 50)

    export_id = f"{timestamp}_{title_slug}"

    # work_dir holds per-segment render output (000_....mp4, 001_....mp4, ...).
    # Two exports whose export_id collides (same title, same wall-clock second —
    # real under concurrent/retried jobs) must never share a work_dir, or their
    # segment files interleave and corrupt concat_segments' output. export_id
    # alone is kept human-readable for the output filename fallback below;
    # work_dir gets an extra uniqueness suffix and a not-already-populated check.
    work_dir = TEMP_DIR / f"{export_id}_{uuid.uuid4().hex[:8]}"
    if work_dir.exists() and any(work_dir.iterdir()):
        raise RuntimeError(f"Work dir {work_dir} already exists and is non-empty — refusing to reuse it.")
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"Work dir: {work_dir}")

    # Phase 4O.3: prefer canonical export_timeline over rebuilt section-based timeline
    canonical_tl = get_canonical_export_timeline(manifest)
    fragment_items_in_canonical = [
        item for item in (canonical_tl or [])
        if item.get("fragment_id") or item.get("type") not in ("title_card", "closing_card")
    ]

    if canonical_tl and fragment_items_in_canonical:
        # Determine which key was used
        sl_tl = (manifest.get("storyline") or {}).get("export_timeline")
        tl_key = "storyline.export_timeline" if (isinstance(sl_tl, list) and sl_tl) else "export_timeline"
        log(f"Using canonical timeline from: {tl_key} ({len(canonical_tl)} items, {len(fragment_items_in_canonical)} fragment items).")
        _manifest_video_mode = (
            (manifest.get("director_plan") or {}).get("video_mode")
            or (manifest.get("planning_summary") or {}).get("video_mode")
            or ((manifest.get("storyline") or {}).get("director_plan") or {}).get("video_mode")
        )
        timeline, quality_checks = _build_canonical_rendering_timeline(manifest, video_mode=_manifest_video_mode)
        quality_checks["speaker_visual_validation"] = ensure_requested_speaker(manifest, timeline)
    else:
        if canonical_tl:
            log("WARNING: canonical_timeline_exists_but_not_used — canonical timeline has no fragment items. Falling back to build_refined_timeline.")
            quality_checks_extra = {"canonical_timeline_exists_but_not_used": True}
        else:
            log("No canonical export_timeline found. Using build_refined_timeline (legacy).")
            quality_checks_extra = {}
        timeline, quality_checks = build_refined_timeline(manifest)
        quality_checks.update(quality_checks_extra if canonical_tl else {})

    # Pre-render validation summary
    archive_visual_items = [t for t in timeline if t.get("segment_type") == "archive_visual"]
    speaker_clip_items = [t for t in timeline if t.get("segment_type") == "speaker_clip"]
    narration_broll_items = [t for t in timeline if t.get("segment_type") == "narration_broll"]
    card_items_tl = [t for t in timeline if t.get("type") == "card"]
    log(f"Rendering canonical timeline:")
    log(f"  total         : {len(timeline)}")
    log(f"  cards         : {len(card_items_tl)}")
    log(f"  speaker_clips : {len(speaker_clip_items)}")
    log(f"  archive_visuals:{len(archive_visual_items)}")
    log(f"  narration_broll:{len(narration_broll_items)}")

    # Compute per-segment minimum duration from the target duration so each narration_broll
    # segment fills its proportional share of the target video length.
    _manifest_target_dur = (
        manifest.get("target_duration_seconds")
        or (manifest.get("storyline") or {}).get("target_duration_seconds")
        or 0
    )
    _content_item_count = max(1, len(narration_broll_items) + len(speaker_clip_items) + len(archive_visual_items))
    _target_sec_per_segment = (
        float(_manifest_target_dur) / _content_item_count
        if _manifest_target_dur else 0.0
    )
    if _manifest_target_dur:
        log(f"  target_dur={_manifest_target_dur}s → min_sec_per_segment={_target_sec_per_segment:.1f}s ({_content_item_count} content items)")

    # Pre-render integrity pass: resolve every segment's media BEFORE spending any
    # ffmpeg time, so a broken manifest fails in seconds with the full list of
    # problems instead of failing (or silently shrinking) mid-render.
    _integrity_failures: list[str] = []
    for _pf_idx, _pf_item in enumerate(timeline, start=1):
        _pf_type = _pf_item.get("segment_type") or _pf_item.get("type") or ""
        _pf_label = f"segment {_pf_idx} ({_pf_type}, fragment={str(_pf_item.get('fragment_id') or '')[:40]!r})"

        if _pf_item.get("type") == "card":
            _pf_card = _pf_item.get("path", "")
            if (not _pf_card or not Path(_pf_card).exists()) and _pf_type != "year_stamp":
                _integrity_failures.append(f"{_pf_label}: card image missing at {_pf_card!r}")
            continue

        _pf_visual = (
            _pf_item.get("render_visual_path") or _pf_item.get("visual_path") or _pf_item.get("asset_path")
        )
        _pf_visual_ok = bool(_pf_visual) and Path(_pf_visual).exists()
        _pf_source = find_source_video(_pf_item.get("source_video"), _pf_item.get("fragment_id"))
        if not _pf_source and not _pf_visual_ok:
            _integrity_failures.append(
                f"{_pf_label}: source video not found (source_video={_pf_item.get('source_video')!r}) "
                f"and no generated visual"
            )

        _pf_audio = _pf_item.get("audio_path")
        if _pf_audio and not Path(_pf_audio).exists():
            _integrity_failures.append(f"{_pf_label}: voiceover audio missing at {_pf_audio!r}")

    if _integrity_failures:
        raise ExportIntegrityError(
            f"Pre-render integrity check failed with {len(_integrity_failures)} issue(s):\n  - "
            + "\n  - ".join(_integrity_failures)
        )
    log(f"Pre-render integrity check passed: all {len(timeline)} segments have resolvable media. (elapsed so far: {time.time() - _export_t0:.1f}s)")

    hls_writer = HLSStreamWriter(hls_dir) if hls_dir is not None else None
    if hls_writer is not None:
        log(f"  [hls] streaming enabled — writing playlist/segments to {hls_dir}")
    try:
        segment_paths, segment_infos, current_time = _render_timeline_items(
            timeline, work_dir, _target_sec_per_segment, hls_writer=hls_writer,
        )
    except Exception:
        if hls_writer is not None:
            hls_writer.finalize(success=False)
        raise
    log(f"  [phase-timing] segment rendering phase done (elapsed so far: {time.time() - _export_t0:.1f}s)")

    # Post-render validation
    rendered_archive_visuals = sum(1 for s in segment_infos if s.get("segment_type") == "archive_visual")
    rendered_speaker = sum(1 for s in segment_infos if s.get("segment_type") == "speaker_clip")
    rendered_narration = sum(1 for s in segment_infos if s.get("segment_type") == "narration_broll")
    source_audio_segments = sum(
        1 for s in segment_infos
        if s["timeline_item"].get("audio_mode") == "source_audio"
        or s.get("segment_type") == "speaker_clip"
    )
    ai_voiceover_segments = sum(
        1 for s in segment_infos
        if s["timeline_item"].get("audio_mode") == "ai_voiceover"
        or s.get("segment_type") == "narration_broll"
    )
    has_any_audio = (
        rendered_speaker > 0
        or rendered_narration > 0
        or any(s["timeline_item"].get("type") == "card" and s["timeline_item"].get("audio_path") for s in segment_infos)
    )

    # Check for AI voiceover attached to speaker_clip (hard error)
    ai_over_speaker = [
        s for s in segment_infos
        if s.get("segment_type") == "speaker_clip"
        and s["timeline_item"].get("audio_mode") == "ai_voiceover"
    ]

    log(f"Post-render audio summary:")
    log(f"  speaker_clips (source_audio): {rendered_speaker}")
    log(f"  narration_broll (ai_voiceover): {rendered_narration}")
    log(f"  archive_visual (silent): {rendered_archive_visuals}")
    log(f"  source_audio_segments: {source_audio_segments}")
    log(f"  ai_voiceover_segments: {ai_voiceover_segments}")

    validation_warnings: list[str] = []
    if ai_over_speaker:
        msg = f"ai_voiceover_over_speaker_clip: {len(ai_over_speaker)} speaker_clip(s) incorrectly have ai_voiceover"
        log(f"ERROR: {msg}")
        validation_warnings.append(msg)
    if rendered_speaker == 0:
        msg = "no_speaker_clips_in_final_video: all clips are muted/narration"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)
    if source_audio_segments == 0:
        msg = "source_audio_segments_zero: no clips use original source audio"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)
    if ai_voiceover_segments > 0 and rendered_speaker == 0:
        msg = "only_ai_voiceover_detected: final video has only AI narration, no real speaker audio"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)
    if archive_visual_items and rendered_archive_visuals == 0 and rendered_narration == 0:
        msg = f"archive_visuals_not_rendered: {len(archive_visual_items)} archive_visual items but 0 rendered"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)
    if len(timeline) > len(card_items_tl) and len(segment_paths) <= len(card_items_tl):
        msg = "export_rendered_only_cards: timeline has non-card items but only cards were rendered"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)
    if not has_any_audio:
        msg = "final_video_has_no_speech_audio: no speaker clips, narration b-roll, or card voiceovers rendered"
        log(f"WARNING: {msg}")
        validation_warnings.append(msg)

    quality_checks["post_render_validation"] = validation_warnings
    quality_checks["source_audio_segments"] = source_audio_segments
    quality_checks["ai_voiceover_segments"] = ai_voiceover_segments
    quality_checks["rendered_speaker_clips"] = rendered_speaker
    quality_checks["speaker_clip_count"] = rendered_speaker

    # Fix 9: surface any render-time footage bandaids (source video ended before the
    # requested window). These should be caught by the upstream clamp in Steps 07/08;
    # if any appear here, an unindexed source slipped through — worth investigating,
    # not silently shipping.
    footage_bandaids = [s["footage_bandaid"] for s in segment_infos if s.get("footage_bandaid")]
    quality_checks["footage_bandaids"] = footage_bandaids
    if footage_bandaids:
        log(f"WARNING: {len(footage_bandaids)} segment(s) needed a render-time footage bandaid:")
        for _b in footage_bandaids:
            log(f"  - {_b}")

    if not segment_paths:
        if hls_writer is not None:
            hls_writer.finalize(success=False)
        raise RuntimeError("No video segments created.")

    if output_name:
        p = Path(output_name)
        # If output_name already contains a directory component, use it as-is
        # (relative to cwd or absolute). Otherwise place it under EXPORTS_DIR.
        output_path = p if p.parent != Path(".") else EXPORTS_DIR / p
    else:
        output_path = EXPORTS_DIR / f"{export_id}.mp4"

    srt_output_path = EXPORTS_DIR / f"{output_path.stem}.srt"
    vtt_output_path = EXPORTS_DIR / f"{output_path.stem}.vtt"

    subtitle_entries, subtitle_meta = build_subtitle_artifacts(segment_infos)
    write_srt(subtitle_entries, srt_output_path)
    write_vtt(subtitle_entries, vtt_output_path)

    concat_segments(
        segment_paths,
        output_path,
        subtitle_path=srt_output_path if burn_subs else None,
    )
    log(f"  [phase-timing] concat phase done (elapsed so far: {time.time() - _export_t0:.1f}s)")

    # Fix 6: A/V ending balance — harmonize audio and video stream durations cleanly
    #
    # This MUST compare the video STREAM against the audio STREAM. It previously
    # used get_media_duration (format=duration) as the video side, but the format
    # duration is max(all streams) — which is normally the audio stream — so the
    # check was comparing audio against itself and reported delta≈0 for every file
    # ever produced. It could not detect A/V desync by construction, which is how
    # a 41 ms offset shipped with the guard reporting "within tolerance".
    #
    # The original reason for using format=duration still stands, though: a
    # concat-produced MP4 can report stream=duration as "N/A", and treating that
    # as 0.0 used to trigger the trim branch and chop the whole video down. So the
    # video stream is probed directly, and anything unusable falls back to the
    # format duration — which degrades to the old always-zero behaviour rather
    # than to a destructive trim.
    def _probe_stream_duration(stream: str) -> float:
        try:
            raw = run_cmd_capture([
                "ffprobe", "-v", "error", "-select_streams", stream,
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output_path),
            ]).strip()
            return float(raw)
        except (ValueError, TypeError):
            return 0.0

    try:
        _format_dur = get_media_duration(output_path)
        _vid_dur_check = _probe_stream_duration("v:0") or _format_dur
        _aud_dur_check = _probe_stream_duration("a:0") or _format_dur
        _av_delta = _aud_dur_check - _vid_dur_check
        log(f"  A/V balance probe: video={_vid_dur_check:.2f}s audio={_aud_dur_check:.2f}s delta={_av_delta:.2f}s")
        if abs(_av_delta) > 0.15:
            if _av_delta > 0:
                # Audio stream longer than video — trim trailing audio to video duration
                log(f"  A/V balance: audio ({_aud_dur_check:.2f}s) exceeds video ({_vid_dur_check:.2f}s) by {_av_delta:.2f}s — trimming trailing audio")
                _trimmed = output_path.parent / (output_path.stem + "_avtrim.mp4")
                run_cmd([
                    "ffmpeg", "-y", "-i", str(output_path),
                    "-t", f"{_vid_dur_check:.3f}",
                    "-c", "copy",
                    str(_trimmed),
                ])
                _trimmed.replace(output_path)
                log(f"  A/V balance: audio stream trimmed to match video duration ({_vid_dur_check:.2f}s)")
            else:
                # Video stream longer than audio — pad audio stream with silence
                _abs_delta = abs(_av_delta)
                log(f"  A/V balance: video ({_vid_dur_check:.2f}s) exceeds audio ({_aud_dur_check:.2f}s) by {_abs_delta:.2f}s — padding audio stream")
                _padded = output_path.parent / (output_path.stem + "_avpad.mp4")
                run_cmd([
                    "ffmpeg", "-y", "-i", str(output_path),
                    "-af", f"apad=pad_dur={_abs_delta + 0.15:.3f}",
                    "-c:v", "copy",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-shortest",
                    str(_padded),
                ])
                _padded.replace(output_path)
                log(f"  A/V balance: audio stream padded — final duration ≈ {_vid_dur_check:.2f}s")
        else:
            log(f"  A/V balance: streams within tolerance ({abs(_av_delta):.2f}s) — no correction needed")

        # Record the measured offset so a regression is visible in QA instead of
        # only in a log line. Audio LATE past ~125 ms (or EARLY past ~45 ms) is
        # where a viewer starts seeing it against a speaker's lips (ITU-R BT.1359).
        quality_checks["av_sync_offset_ms"] = round(_av_delta * 1000, 1)
        quality_checks["av_sync_within_perceptual_limit"] = -0.045 <= _av_delta <= 0.125
    except Exception as _av_exc:
        log(f"  A/V balance check failed (non-fatal): {_av_exc}")
        quality_checks["av_sync_offset_ms"] = None
        quality_checks["av_sync_within_perceptual_limit"] = None

    if hls_writer is not None:
        hls_writer.finalize(success=True)

    rhythm = {
        "enabled": True,
        "narration_broll_segments": len(narration_broll_items),
        "static_card_segments": len(card_items_tl),
        "speaker_lower_thirds": sum(1 for item in timeline if item.get("lower_third")),
        "fallback_to_card_segments": sum(1 for item in timeline if item.get("segment_type") == "fallback_narration"),
    }

    # Build export manifest with per-segment rendering status
    rendered_segments_manifest = []
    for s in segment_infos:
        tl_item = s["timeline_item"]
        _seg_t = s.get("segment_type") or tl_item.get("segment_type") or ""
        _is_card_seg = _seg_t in ("title_card", "closing_card", "year_stamp") or tl_item.get("type") == "card"
        _display_type = _seg_t if _seg_t in (
            "speaker_clip", "archive_visual", "narration_broll",
            "title_card", "closing_card", "year_stamp",
        ) else (tl_item.get("type") or _seg_t or "unknown")
        rendered_segments_manifest.append({
            "type": _display_type,
            "segment_type": _seg_t,
            "audio_mode": tl_item.get("audio_mode") or ("source_audio" if (_seg_t == "speaker_clip") else "silent"),
            "fragment_id": s.get("fragment_id") or tl_item.get("fragment_id"),
            "source_video": tl_item.get("source_video"),
            "speaker": tl_item.get("speaker") or tl_item.get("speaker_name") or tl_item.get("primary_speaker"),
            "start": tl_item.get("start"),
            "end": tl_item.get("end"),
            "duration": s.get("duration"),
            "muted": s.get("muted") if s.get("muted") is not None else tl_item.get("muted"),
            "subtitle_text": get_segment_subtitle_text(tl_item),
            "rendered": True,
            "video_year": None if _is_card_seg else tl_item.get("video_year", None),
        })

    export_manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "work_dir": str(work_dir),
        "timeline": timeline,
        "segments": segment_infos,
        "rendered_segments": rendered_segments_manifest,
        "burn_subtitles": burn_subs,
        "audio_strategy": "Canonical export_timeline is used directly. archive_visual = muted frames. speaker_clip = source audio. narration_broll = muted frames + external voiceover.",
        "visual_strategy": (
            "speaker_clip = source video + source audio. "
            "narration_broll = generated topic visual + AI voiceover (when visual_path present) "
            "or archive video fallback. "
            "Title/closing = PIL cards from manifest."
        ),
        "subtitle_style": "External SRT/VTT generated from final timeline with chunked full-text captions. Burn-in uses the generated SRT when enabled.",
        "subtitles": {
            "srt_path": str(srt_output_path),
            "vtt_path": str(vtt_output_path),
            "entry_count": len(subtitle_entries),
            "coverage": subtitle_meta["coverage"],
            "warnings": subtitle_meta["warnings"],
        },
        "rhythm": rhythm,
        "quality_checks": quality_checks,
        "duration_budget": build_duration_budget(
            timeline,
            _manifest_target_dur or None,
            stage="09_export_video",
        ),
        "validation_warnings": validation_warnings,
        "generated_visual_segments": sum(1 for s in segment_infos if s.get("visual_mode") == "generated_asset"),
        "muted_archive_narration_segments": sum(
            1 for s in segment_infos
            if s.get("segment_type") == "narration_broll" and s.get("visual_mode") != "generated_asset"
        ),
    }

    export_manifest_path = EXPORTS_DIR / f"{output_path.stem}_export_manifest.json"
    export_manifest_path.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Timeline metadata sidecar
    _tl_meta_path = EXPORTS_DIR / f"{output_path.stem}_timeline_metadata.json"
    _tl_cursor = 0.0
    _tl_events = []
    _card_types = {"title_card", "closing_card"}
    _video_mode = (
        (manifest.get("director_plan") or {}).get("video_mode")
        or (manifest.get("planning_summary") or {}).get("video_mode")
        or ((manifest.get("storyline") or {}).get("director_plan") or {}).get("video_mode")
    )
    for _si, _seg in enumerate(segment_infos):
        _ti = _seg["timeline_item"]
        _stype = _seg.get("segment_type") or _ti.get("segment_type") or _ti.get("type") or ""
        _dur = float(_seg.get("duration") or 0.0)
        _t_start = _tl_cursor
        _t_end = _tl_cursor + _dur
        _tl_cursor = _t_end
        if _stype in _card_types:
            continue
        _year = None
        if _stype == "year_stamp":
            _year = _ti.get("year")
        else:
            _year = _ti.get("video_year")
        _speaker_val = (
            _ti.get("speaker") or _ti.get("speaker_name") or _ti.get("primary_speaker") or _seg.get("speaker")
        )
        _speaker = _speaker_val if (_stype in ("speaker_clip", "speaker_visual") or _speaker_val) else None
        _tl_events.append({
            "segment_index": _si,
            "type": _stype,
            "year": _year,
            "speaker": _speaker,
            "topic": _ti.get("primary_chapter"),
            "section_role": _ti.get("section_role"),
            "clip_start": _ti.get("start"),
            "clip_end": _ti.get("end"),
            "timeline_start": round(_t_start, 3),
            "timeline_end": round(_t_end, 3),
        })
    _tl_meta = {
        "export_file": output_path.name,
        "generated_at": datetime.utcnow().isoformat(),
        "video_mode": _video_mode,
        "timeline_events": _tl_events,
    }
    _tl_meta_path.write_text(
        json.dumps(_tl_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Timeline metadata sidecar: {_tl_meta_path}")

    # Post-export ffprobe duration check
    fmt_dur: float = 0.0
    try:
        fmt_dur = get_media_duration(output_path)
        vid_dur_str = run_cmd_capture([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ]).strip() or "0"
        aud_dur_str = run_cmd_capture([
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ]).strip() or "0"
        vid_dur = float(vid_dur_str)
        aud_dur = float(aud_dur_str)
        dur_delta = abs(vid_dur - aud_dur)
        log("")
        log("Final duration check:")
        log(f"  format_duration:       {fmt_dur:.2f}s")
        log(f"  video_stream_duration: {vid_dur:.2f}s")
        log(f"  audio_stream_duration: {aud_dur:.2f}s")
        log(f"  duration_delta:        {dur_delta:.2f}s")
        if dur_delta > 0.2:
            log(f"WARNING: duration_delta ({dur_delta:.2f}s) exceeds 0.2s — possible A/V sync issue.")
        _target = float(_manifest_target_dur) if _manifest_target_dur else 0.0
        _planned = float((export_manifest.get("duration_budget") or {}).get("planned_total_seconds") or 0.0)
        export_manifest["duration_check"] = {
            "format_duration": fmt_dur,
            "video_stream_duration": vid_dur,
            "audio_stream_duration": aud_dur,
            "duration_delta": dur_delta,
            "target_duration_seconds": _target or None,
            "planned_total_seconds": _planned or None,
            "meets_target_floor": (fmt_dur >= _target) if _target else None,
        }
        # Integrity against DROPPED content is enforced structurally elsewhere:
        # _render_one_segment raises on any skip or zero-duration segment, and
        # the segment-count check (rendered vs planned items) is the reliable
        # drop signal. We deliberately do NOT hard-fail on rendered-vs-summed-
        # duration: on a long multi-segment video, fades, ffmpeg -shortest, and
        # chronological-relay bridge recomputation make the real total legitimately
        # a bit under the naive sum of declared item durations — that drift is not
        # a defect. Duration below target is only ever a content-exhaustion warning.
        if _target and fmt_dur < _target:
            log(
                f"WARNING: rendered {fmt_dur:.1f}s is below the {_target:.1f}s requested — "
                f"relevant content exhausted (planned {_planned:.1f}s). Shipping honest "
                f"best-effort duration; no filler added."
            )
        elif _planned and fmt_dur < _planned - max(5.0, _planned * 0.15):
            # Informational only: unusually large drift from the plan, but the
            # video still meets target. Worth noting, not worth failing.
            log(
                f"NOTE: rendered {fmt_dur:.1f}s is {_planned - fmt_dur:.1f}s under the "
                f"summed plan {_planned:.1f}s (fades/-shortest/relay-bridge drift); "
                f"target {_target:.1f}s still met."
            )
        export_manifest_path.write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except ExportIntegrityError:
        raise
    except Exception as exc:
        log(f"Post-export duration check failed: {exc}")

    # Post-render audio summary
    sc_rendered = quality_checks.get("rendered_speaker_clips", 0)
    log("")
    log("Post-render audio summary:")
    log(f"  speaker_clips (source_audio):  {sc_rendered}")
    log(f"  narration_broll (ai_voiceover): {quality_checks.get('narration_broll_count', 0)}")
    log(f"  source_audio_segments:         {source_audio_segments}")
    log(f"  ai_voiceover_segments:         {ai_voiceover_segments}")
    log(f"  final_duration_seconds:        {fmt_dur:.2f}s")

    log("")
    log("Export complete")
    log(f"Output video: {output_path}")
    log(f"Export manifest: {export_manifest_path}")
    log(f"  [phase-timing] total export_video() wall-clock: {time.time() - _export_t0:.1f}s")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export refined AI4IA documentary video from generated assets.")
    parser.add_argument(
        "--manifest",
        default=str(LATEST_MANIFEST_PATH),
        help="Path to asset_manifest.json. Defaults to latest asset manifest."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output MP4 filename, for example ai4ia_demo_refined.mp4"
    )
    parser.add_argument(
        "--burn-subtitles",
        action="store_true",
        help="Burn the generated subtitles into the final video."
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Backward-compatible alias: do not burn subtitles into the final video. External SRT/VTT files are still generated."
    )
    parser.add_argument(
        "--hls-dir",
        default=None,
        help="If given, progressively write an HLS playlist + .ts segments to this directory "
             "as each segment renders, so a client can start streaming before export finishes."
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()

    log("Starting Step 09: Export Refined Final Video")
    log(f"Manifest: {manifest_path}")

    output_path = export_video(
        manifest_path=manifest_path,
        output_name=args.output,
        burn_subs=args.burn_subtitles and not args.no_subtitles,
        hls_dir=Path(args.hls_dir).resolve() if args.hls_dir else None,
    )

    log(f"Done: {output_path}")


if __name__ == "__main__":
    main()
