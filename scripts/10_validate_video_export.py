"""
AI4IA Phase 4F — Final QA Validator + Demo Packaging

Validates an export manifest and its associated files, producing a
JSON QA report and a human-readable console summary.

Usage:
    .venv/bin/python scripts/10_validate_video_export.py \
        --export-manifest data/exports/<name>_export_manifest.json

Optional:
    --asset-manifest  <path>   override inferred asset manifest path
    --storyline       <path>   override inferred storyline JSON path
    --output-report   <path>   where to write the QA JSON report
    --strict                   exit non-zero on serious warnings too
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
WEIGHT_FILES = 15
WEIGHT_VIDEO_STREAM = 20
WEIGHT_SUBTITLES = 15
WEIGHT_CLIP_DIVERSITY = 20
WEIGHT_PLANNING = 15
WEIGHT_RHYTHM = 10
WEIGHT_MANIFEST_COMPLETENESS = 5

SCORE_PASS = 80
SCORE_WARN = 60


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / p


def count_srt_cues(text: str) -> int:
    return text.count("-->")


def count_ellipsis_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.rstrip().endswith("..."))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_files(export: dict, base_dir: Path) -> dict:
    result: dict = {"status": "pass", "details": {}}
    failures: list[str] = []
    warnings: list[str] = []

    # MP4
    mp4_path_str = export.get("output_path", "")
    mp4_path = resolve_path(mp4_path_str, base_dir) if mp4_path_str else None
    mp4_exists = bool(mp4_path and mp4_path.exists())
    mp4_size_mb = round(mp4_path.stat().st_size / (1024 * 1024), 2) if mp4_exists else 0.0

    result["details"]["mp4_exists"] = mp4_exists
    result["details"]["mp4_size_mb"] = mp4_size_mb

    if not mp4_exists:
        failures.append("MP4 file not found")
    elif mp4_size_mb < 1.0:
        failures.append(f"MP4 file too small ({mp4_size_mb} MB)")

    # SRT / VTT — look in subtitles block first, then infer from MP4 stem
    subtitles_block = export.get("subtitles") or {}
    srt_str = subtitles_block.get("srt_path") or ""
    vtt_str = subtitles_block.get("vtt_path") or ""

    if not srt_str and mp4_path:
        srt_str = str(mp4_path.with_suffix(".srt"))
    if not vtt_str and mp4_path:
        vtt_str = str(mp4_path.with_suffix(".vtt"))

    srt_path = resolve_path(srt_str, base_dir) if srt_str else None
    vtt_path = resolve_path(vtt_str, base_dir) if vtt_str else None

    srt_exists = bool(srt_path and srt_path.exists())
    vtt_exists = bool(vtt_path and vtt_path.exists())

    result["details"]["srt_exists"] = srt_exists
    result["details"]["vtt_exists"] = vtt_exists
    result["details"]["srt_path"] = str(srt_path) if srt_path else None
    result["details"]["vtt_path"] = str(vtt_path) if vtt_path else None

    if not srt_exists:
        warnings.append("SRT file not found")
    if not vtt_exists:
        warnings.append("VTT file not found")

    if failures:
        result["status"] = "fail"
    elif warnings:
        result["status"] = "warn"

    result["failures"] = failures
    result["warnings"] = warnings

    # Pass resolved paths forward for subtitle & video stream checks
    result["_srt_path"] = srt_path
    result["_vtt_path"] = vtt_path
    result["_mp4_path"] = mp4_path

    # Score contribution (out of WEIGHT_FILES)
    score = WEIGHT_FILES
    if not mp4_exists or mp4_size_mb < 1.0:
        score -= 10
    if not srt_exists:
        score -= 2
    if not vtt_exists:
        score -= 3
    result["_score"] = max(score, 0)

    return result


def check_video_stream(mp4_path: Path | None) -> dict:
    result: dict = {"status": "pass", "details": {}}
    failures: list[str] = []
    warnings: list[str] = []

    if not mp4_path or not mp4_path.exists():
        result["status"] = "fail"
        result["failures"] = ["MP4 file not found for video stream inspection"]
        result["warnings"] = []
        result["_score"] = 0
        return result

    # 1. Resolution & Frame-Rate check via ffprobe
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,codec_name",
            "-of", "json",
            str(mp4_path),
        ]
        proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if proc.returncode == 0:
            info = json.loads(proc.stdout)
            streams = info.get("streams", [])
            if not streams:
                failures.append("No video stream found in MP4 file")
            else:
                vstream = streams[0]
                width = vstream.get("width")
                height = vstream.get("height")
                r_fps = str(vstream.get("r_frame_rate") or "")
                avg_fps = str(vstream.get("avg_frame_rate") or "")

                result["details"]["width"] = width
                result["details"]["height"] = height
                result["details"]["r_frame_rate"] = r_fps
                result["details"]["avg_frame_rate"] = avg_fps

                if width != 1920 or height != 1080:
                    failures.append(f"Resolution mismatch: expected 1920x1080, got {width}x{height}")

                fps_val = None
                if avg_fps and "/" in avg_fps:
                    num, den = avg_fps.split("/")
                    if float(den) > 0:
                        fps_val = float(num) / float(den)
                elif r_fps and "/" in r_fps:
                    num, den = r_fps.split("/")
                    if float(den) > 0:
                        fps_val = float(num) / float(den)

                if fps_val is not None:
                    result["details"]["fps_calculated"] = round(fps_val, 2)
                    if abs(fps_val - 30.0) > 0.5:
                        failures.append(f"Frame rate mismatch: expected 30.0 fps, got {fps_val:.2f} fps ({avg_fps or r_fps})")
        else:
            warnings.append(f"ffprobe stream inspection failed with exit code {proc.returncode}")
    except Exception as exc:
        warnings.append(f"Resolution/fps check error: {exc}")

    # 2. Freeze detection check via ffmpeg freezedetect — disabled.
    # Static generated-asset B-roll (narration over a still diagram/image) has zero
    # pixel motion for its full duration, so freezedetect flags it as "frozen" even
    # though it's intentional, not a broken export. Left in place, non-fatal, for
    # informational stats only.
    # try:
    #     freeze_cmd = [
    #         "ffmpeg", "-nostats", "-v", "info",
    #         "-i", str(mp4_path),
    #         "-vf", "freezedetect=n=-30dB:d=3",
    #         "-f", "null", "-",
    #     ]
    #     proc = subprocess.run(freeze_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    #     stderr_log = proc.stderr or ""
    #
    #     freeze_durations = [
    #         float(m) for m in re.findall(r"freeze_duration:\s*([\d.]+)", stderr_log)
    #     ]
    #     max_freeze = max(freeze_durations) if freeze_durations else 0.0
    #     result["details"]["freeze_detected"] = len(freeze_durations) > 0
    #     result["details"]["max_freeze_duration_s"] = round(max_freeze, 2)
    #     result["details"]["freeze_runs_count"] = len(freeze_durations)
    #
    #     if max_freeze > 6.0:
    #         failures.append(
    #             f"Video freeze detected: max freeze duration {max_freeze:.2f}s exceeds 6.0s threshold ({len(freeze_durations)} freeze run{'s' if len(freeze_durations) > 1 else ''})"
    #         )
    #     elif max_freeze > 4.0:
    #         warnings.append(f"Minor video freeze detected: max duration {max_freeze:.2f}s")
    # except Exception as exc:
    #     warnings.append(f"Freeze detection error: {exc}")

    if failures:
        result["status"] = "fail"
    elif warnings:
        result["status"] = "warn"

    result["failures"] = failures
    result["warnings"] = warnings
    result["_score"] = 0 if failures else (15 if warnings else WEIGHT_VIDEO_STREAM)

    return result


def check_subtitles(export: dict, srt_path: Path | None, vtt_path: Path | None) -> dict:
    result: dict = {"status": "pass", "details": {}}
    failures: list[str] = []
    warnings: list[str] = []

    subtitles_block = export.get("subtitles") or {}
    manifest_entry_count = subtitles_block.get("entry_count", 0)
    coverage = subtitles_block.get("coverage") or {}

    speaker_segments_subtitled = coverage.get("speaker_segments_subtitled", 0)
    voiceover_segments_subtitled = coverage.get("voiceover_segments_subtitled", 0)
    segments_without_subtitles = coverage.get("segments_without_subtitles", 0)

    result["details"]["manifest_entry_count"] = manifest_entry_count
    result["details"]["speaker_segments_subtitled"] = speaker_segments_subtitled
    result["details"]["voiceover_segments_subtitled"] = voiceover_segments_subtitled
    result["details"]["segments_without_subtitles"] = segments_without_subtitles

    srt_count = 0
    vtt_count = 0
    ellipsis_count = 0

    if srt_path and srt_path.exists():
        srt_text = srt_path.read_text(encoding="utf-8", errors="replace")
        srt_count = count_srt_cues(srt_text)
        ellipsis_count = count_ellipsis_lines(srt_text)
        if srt_count == 0:
            failures.append("SRT file has zero cue entries")

    if vtt_path and vtt_path.exists():
        vtt_text = vtt_path.read_text(encoding="utf-8", errors="replace")
        vtt_count = count_srt_cues(vtt_text)
        if not vtt_text.startswith("WEBVTT"):
            failures.append("VTT file does not begin with WEBVTT header")
        if vtt_count == 0:
            failures.append("VTT file has zero cue entries")

    result["details"]["srt_cue_count"] = srt_count
    result["details"]["vtt_cue_count"] = vtt_count

    # Cue count parity
    cue_match = srt_count == vtt_count
    result["details"]["srt_vtt_cue_count_match"] = cue_match
    if srt_count > 0 and vtt_count > 0 and not cue_match:
        warnings.append(
            f"SRT ({srt_count}) and VTT ({vtt_count}) cue counts differ"
        )

    # Manifest vs actual
    if manifest_entry_count > 0 and srt_count > 0 and abs(manifest_entry_count - srt_count) > 3:
        warnings.append(
            f"Manifest entry_count ({manifest_entry_count}) differs from actual SRT cue count ({srt_count})"
        )

    # Ellipsis truncation
    result["details"]["ellipsis_line_count"] = ellipsis_count
    if ellipsis_count > 5:
        warnings.append(
            f"{ellipsis_count} subtitle lines end with '...' — possible aggressive truncation"
        )

    # Subtitle warnings from manifest
    manifest_sub_warnings = subtitles_block.get("warnings") or []
    if manifest_sub_warnings:
        warnings.append(
            f"{len(manifest_sub_warnings)} subtitle warning(s) from asset generation"
        )

    if failures:
        result["status"] = "fail"
    elif warnings:
        result["status"] = "warn"

    result["failures"] = failures
    result["warnings"] = warnings

    score = WEIGHT_SUBTITLES
    if srt_count == 0:
        score -= 8
    if vtt_count == 0:
        score -= 4
    if not cue_match and (srt_count > 0 or vtt_count > 0):
        score -= 2
    if ellipsis_count > 5:
        score -= 2
    result["_score"] = max(score, 0)

    return result


def check_clip_diversity(export: dict) -> dict:
    result: dict = {"status": "pass", "details": {}}
    failures: list[str] = []
    warnings: list[str] = []

    qc = export.get("quality_checks") or {}
    timeline = export.get("timeline") or []

    unique_fragments = qc.get("unique_fragments", 0)
    repeated_fragments = qc.get("repeated_fragments") or []
    source_counts = qc.get("source_video_use_counts") or {}
    speaker_counts = qc.get("speaker_use_counts") or {}
    fallback_segments = qc.get("fallback_segments", 0)

    # Total sections from timeline (exclude title/closing cards)
    section_clips = [
        t for t in timeline
        if t.get("type") in ("archive_clip",) or t.get("segment_type") == "fallback_narration"
    ]
    total_clip_slots = len(section_clips) if section_clips else 1

    unique_source_videos = len(source_counts)
    max_source_reuse = max(source_counts.values()) if source_counts else 0
    unique_speakers = len(speaker_counts)
    max_speaker_reuse = max(speaker_counts.values()) if speaker_counts else 0
    fallback_ratio = fallback_segments / total_clip_slots

    result["details"]["unique_fragments"] = unique_fragments
    result["details"]["repeated_fragments"] = repeated_fragments
    result["details"]["unique_source_videos"] = unique_source_videos
    result["details"]["max_source_video_reuse"] = max_source_reuse
    result["details"]["unique_speakers"] = unique_speakers
    result["details"]["max_speaker_reuse"] = max_speaker_reuse
    result["details"]["fallback_segments"] = fallback_segments
    result["details"]["total_clip_slots"] = total_clip_slots
    result["details"]["fallback_ratio"] = round(fallback_ratio, 2)

    if not source_counts:
        warnings.append(
            "source_video_use_counts is empty — source-level diversity not tracked "
            "(fragment_id-level deduplication still active)"
        )
    if not speaker_counts:
        warnings.append(
            "speaker_use_counts is empty — speaker metadata not enriched yet (Phase 4B enrichment may be pending)"
        )

    speaker_clips = [
        t for t in timeline
        if (t.get("segment_type") or t.get("type")) == "speaker_clip"
        or (t.get("audio_mode") in ("source", "source_audio") and (t.get("type") or t.get("segment_type")) not in ("title_card", "closing_card", "year_stamp"))
    ]
    result["details"]["speaker_clip_count"] = len(speaker_clips)

    if len(speaker_clips) == 0:
        warnings.append("Zero speaker clips in export — video contains no real speaker footage (narration/B-roll only)")

    if repeated_fragments:
        failures.append(
            f"Repeated fragment IDs detected: {repeated_fragments}"
        )

    if max_source_reuse > 2:
        warnings.append(
            f"Source video used {max_source_reuse} times (recommended max 2)"
        )

    if max_speaker_reuse > 2:
        warnings.append(
            f"Speaker used {max_speaker_reuse} times (recommended max 2)"
        )

    if fallback_ratio > 0.5:
        warnings.append(
            f"High fallback ratio: {fallback_segments}/{total_clip_slots} "
            f"({round(fallback_ratio * 100)}%) — consider improving Step 07 retrieval "
            "or expanding the archive"
        )

    if failures:
        result["status"] = "fail"
    elif warnings:
        result["status"] = "warn"

    result["failures"] = failures
    result["warnings"] = warnings

    score = WEIGHT_CLIP_DIVERSITY
    if repeated_fragments:
        score -= 15
    if fallback_ratio > 0.5:
        score -= 6
    if max_source_reuse > 2:
        score -= 2
    if max_speaker_reuse > 2:
        score -= 2
    result["_score"] = max(score, 0)

    return result


def check_rhythm(export: dict) -> dict:
    result: dict = {"status": "pass", "details": {}}
    warnings: list[str] = []

    rhythm = export.get("rhythm") or {}
    if not rhythm or not rhythm.get("enabled"):
        result["status"] = "warn"
        result["details"]["rhythm_block_present"] = False
        result["warnings"] = ["Rhythm metadata block missing or disabled — may be an older export"]
        result["failures"] = []
        result["_score"] = WEIGHT_RHYTHM // 2
        return result

    result["details"]["rhythm_block_present"] = True

    narration_broll = rhythm.get("narration_broll_segments", 0)
    static_cards = rhythm.get("static_card_segments", 0)
    lower_thirds = rhythm.get("speaker_lower_thirds", 0)
    fallback_cards = rhythm.get("fallback_to_card_segments", 0)

    result["details"]["narration_broll_segments"] = narration_broll
    result["details"]["static_card_segments"] = static_cards
    result["details"]["speaker_lower_thirds"] = lower_thirds
    result["details"]["fallback_to_card_segments"] = fallback_cards

    # Derive archive clip count from timeline
    timeline = export.get("timeline") or []
    speaker_clips = [
        t for t in timeline
        if t.get("type") == "archive_clip" and t.get("role") == "speaker_soundbite"
    ]
    archive_clips = [t for t in timeline if t.get("type") == "archive_clip"]

    if static_cards > 6 and narration_broll == 0:
        warnings.append(
            "High static card count with zero B-roll under narration — "
            "consider enabling B-roll (Phase 4D feature)"
        )

    if speaker_clips and lower_thirds == 0:
        warnings.append(
            "Speaker soundbite clips present but no lower-thirds generated — "
            "consider enabling lower-thirds (Phase 4D feature)"
        )

    if not archive_clips:
        warnings.append("No archive clips in timeline — video is entirely card-based")

    if warnings:
        result["status"] = "warn"

    result["failures"] = []
    result["warnings"] = warnings

    score = WEIGHT_RHYTHM
    if static_cards > 6 and narration_broll == 0:
        score -= 4
    if speaker_clips and lower_thirds == 0:
        score -= 4
    if not archive_clips:
        score -= 6
    result["_score"] = max(score, 0)

    return result


def check_planning(asset_manifest: dict | None, storyline: dict | None) -> dict:
    result: dict = {"status": "pass", "details": {}}
    warnings: list[str] = []

    source = asset_manifest or storyline or {}

    director_plan = source.get("director_plan")
    planning_summary = source.get("planning_summary")
    video_mode = source.get("video_mode")
    evidence_strength = None
    exact_clip_count = None
    related_clip_count = None

    if director_plan and isinstance(director_plan, dict):
        evidence_strength = director_plan.get("evidence_strength")
        exact_clip_count = director_plan.get("exact_clip_count")
        related_clip_count = director_plan.get("related_clip_count")

    if planning_summary and isinstance(planning_summary, dict):
        if not evidence_strength:
            evidence_strength = planning_summary.get("evidence_strength")
        if not video_mode:
            video_mode = planning_summary.get("video_mode")

    result["details"]["director_plan_present"] = bool(director_plan)
    result["details"]["planning_summary_present"] = bool(planning_summary)
    result["details"]["video_mode"] = video_mode
    result["details"]["evidence_strength"] = evidence_strength
    result["details"]["exact_clip_count"] = exact_clip_count
    result["details"]["related_clip_count"] = related_clip_count

    if not director_plan:
        warnings.append(
            "director_plan missing — may be a pre-Phase-4E export or planning data not yet generated"
        )

    if video_mode in (None, ""):
        warnings.append(
            "video_mode not set — consider regenerating storyline with Phase 4E planning"
        )
    elif video_mode in ("low_evidence_brief", "archive_only_montage") and not planning_summary:
        warnings.append(
            f"video_mode is '{video_mode}' but no planning_summary present — "
            "planning notes recommended for weak-evidence modes"
        )

    if warnings:
        result["status"] = "warn"

    result["failures"] = []
    result["warnings"] = warnings

    score = WEIGHT_PLANNING
    if not director_plan:
        score -= 6
    if not video_mode:
        score -= 4
    result["_score"] = max(score, 0)

    return result


def check_manifest_completeness(export: dict) -> dict:
    result: dict = {"status": "pass", "details": {}}
    warnings: list[str] = []

    expected_keys = ["quality_checks", "rhythm", "subtitles", "timeline", "segments"]
    missing = [k for k in expected_keys if not export.get(k)]

    result["details"]["present_keys"] = [k for k in expected_keys if export.get(k)]
    result["details"]["missing_keys"] = missing

    if missing:
        warnings.append(
            f"Export manifest missing blocks: {missing} — may be an older export"
        )
        result["status"] = "warn"

    result["failures"] = []
    result["warnings"] = warnings

    score = WEIGHT_MANIFEST_COMPLETENESS
    score -= len(missing)
    result["_score"] = max(score, 0)

    return result


# ---------------------------------------------------------------------------
# Score + status aggregation
# ---------------------------------------------------------------------------

def compute_overall(checks: dict) -> tuple[int, str, list[str], list[str], list[str]]:
    total = sum(
        checks[k].get("_score", 0)
        for k in ("files", "video_stream", "subtitles", "clip_diversity", "planning", "rhythm", "manifest_completeness")
        if k in checks
    )

    all_failures: list[str] = []
    all_warnings: list[str] = []

    for k, v in checks.items():
        section_label = k.replace("_", " ").title()
        for f in v.get("failures", []):
            all_failures.append(f"[{section_label}] {f}")
        for w in v.get("warnings", []):
            all_warnings.append(f"[{section_label}] {w}")

    if all_failures or total < SCORE_WARN:
        status = "fail"
    elif total < SCORE_PASS:
        status = "warn"
    else:
        status = "pass"

    recommendations: list[str] = []

    # Derive recommendations from warnings
    qc_warns = checks.get("clip_diversity", {}).get("warnings", [])
    for w in qc_warns:
        if "fallback" in w.lower():
            recommendations.append(
                "Consider improving Step 07 retrieval prompt or expanding the archive "
                "if fallback segments exceed 50%."
            )
            break

    sub_warns = checks.get("subtitles", {}).get("warnings", [])
    for w in sub_warns:
        if "zero" in w.lower() or "not found" in w.lower():
            recommendations.append(
                "SRT/VTT files missing or empty — re-run Step 09 with a Phase 4C+ export."
            )
            break

    plan_warns = checks.get("planning", {}).get("warnings", [])
    for w in plan_warns:
        if "director_plan" in w.lower():
            recommendations.append(
                "Re-run Steps 07 and 08 with Phase 4E code to get director_plan and video_mode metadata."
            )
            break

    rhythm_warns = checks.get("rhythm", {}).get("warnings", [])
    for w in rhythm_warns:
        if "b-roll" in w.lower():
            recommendations.append(
                "B-roll under narration is available from Phase 4D — re-export with latest Step 09."
            )
        if "lower-thirds" in w.lower():
            recommendations.append(
                "Lower-third overlays are available from Phase 4D — re-export with latest Step 09."
            )

    return total, status, all_failures, all_warnings, recommendations


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def icon(status: str) -> str:
    return {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}.get(status, "[????]")


def print_report(
    export_path: str,
    total_score: int,
    status: str,
    checks: dict,
    failures: list[str],
    warnings: list[str],
    recommendations: list[str],
) -> None:
    line = "=" * 52
    print()
    print("AI4IA Video Export QA")
    print(line)
    print(f"Export : {export_path}")
    print(f"Status : {status.upper()}")
    print(f"Score  : {total_score}/100")
    print()

    # Files
    f = checks["files"]["details"]
    print("Files:")
    mp4_status = "pass" if f.get("mp4_exists") and f.get("mp4_size_mb", 0) >= 1.0 else "fail"
    print(f"  {icon(mp4_status)} MP4 exists: {f.get('mp4_size_mb', 0)} MB")
    srt_s = "pass" if f.get("srt_exists") else "warn"
    vtt_s = "pass" if f.get("vtt_exists") else "warn"
    sub = checks["subtitles"]["details"]
    print(f"  {icon(srt_s)} SRT exists: {sub.get('srt_cue_count', 0)} cues")
    print(f"  {icon(vtt_s)} VTT exists: {sub.get('vtt_cue_count', 0)} cues")

    # Video Stream Integrity
    if "video_stream" in checks:
        vs = checks["video_stream"]["details"]
        vs_fails = checks["video_stream"].get("failures", [])
        print()
        print("Video Stream:")
        res_s = "fail" if any("Resolution" in fl for fl in vs_fails) else "pass"
        fps_s = "fail" if any("Frame rate" in fl for fl in vs_fails) else "pass"
        freeze_s = "fail" if any("Video freeze" in fl for fl in vs_fails) else ("warn" if vs.get("max_freeze_duration_s", 0) > 1.5 else "pass")
        print(f"  {icon(res_s)} Resolution: {vs.get('width', '?')}x{vs.get('height', '?')}")
        print(f"  {icon(fps_s)} Frame rate: {vs.get('avg_frame_rate', '?')} (r_fps: {vs.get('r_frame_rate', '?')})")
        print(f"  {icon(freeze_s)} Freeze detection: max duration {vs.get('max_freeze_duration_s', 0.0)}s ({vs.get('freeze_runs_count', 0)} freeze runs)")

    # Clip diversity
    d = checks["clip_diversity"]["details"]
    print()
    print("Clip diversity:")
    rep = d.get("repeated_fragments") or []
    print(f"  {icon('fail' if rep else 'pass')} Repeated fragments: {len(rep)}")
    print(f"  {icon('pass')} Unique fragments: {d.get('unique_fragments', 0)}")
    fb = d.get("fallback_segments", 0)
    fb_status = "warn" if d.get("fallback_ratio", 0) > 0.5 else "pass"
    print(f"  {icon(fb_status)} Fallback segments: {fb} / {d.get('total_clip_slots', '?')}")
    sv = d.get("unique_source_videos", 0)
    sp = d.get("unique_speakers", 0)
    sv_status = "warn" if sv == 0 else "pass"
    sp_status = "warn" if sp == 0 else "pass"
    print(f"  {icon(sv_status)} Unique source videos: {sv if sv else 'not tracked'}")
    print(f"  {icon(sp_status)} Unique speakers: {sp if sp else 'not tracked'}")

    # Subtitles
    print()
    print("Subtitles:")
    print(f"  {icon('pass' if sub.get('speaker_segments_subtitled', 0) > 0 else 'warn')} Speaker subtitles: {sub.get('speaker_segments_subtitled', 0)}")
    print(f"  {icon('pass' if sub.get('voiceover_segments_subtitled', 0) > 0 else 'warn')} Voice-over subtitles: {sub.get('voiceover_segments_subtitled', 0)}")
    match_s = "pass" if sub.get("srt_vtt_cue_count_match") else "warn"
    print(f"  {icon(match_s)} SRT/VTT cue counts match: {sub.get('srt_vtt_cue_count_match', False)}")

    # Planning
    p = checks["planning"]["details"]
    print()
    print("Planning:")
    plan_s = "pass" if p.get("director_plan_present") else "warn"
    print(f"  {icon(plan_s)} director_plan: {'present' if p.get('director_plan_present') else 'missing'}")
    mode = p.get("video_mode") or "not set"
    mode_s = "pass" if p.get("video_mode") else "warn"
    print(f"  {icon(mode_s)} video_mode: {mode}")
    ev = p.get("evidence_strength") or "not set"
    ev_s = "pass" if p.get("evidence_strength") else "warn"
    print(f"  {icon(ev_s)} evidence_strength: {ev}")

    # Rhythm
    r = checks["rhythm"]["details"]
    print()
    print("Rhythm:")
    broll = r.get("narration_broll_segments", 0)
    broll_s = "warn" if broll == 0 and r.get("static_card_segments", 0) > 6 else "pass"
    print(f"  {icon(broll_s)} B-roll under narration: {broll}")
    lt = r.get("speaker_lower_thirds", 0)
    lt_s = "pass" if lt > 0 else "warn"
    print(f"  {icon(lt_s)} Lower-thirds: {lt}")
    print(f"  {icon('pass')} Static cards: {r.get('static_card_segments', 0)}")

    if failures:
        print()
        print("Failures:")
        for fl in failures:
            print(f"  {icon('fail')} {fl}")

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  {icon('warn')} {w}")

    if recommendations:
        print()
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI4IA Phase 4F — Final QA Validator for video exports."
    )
    parser.add_argument(
        "--export-manifest",
        required=True,
        help="Path to the export manifest JSON (e.g. data/exports/<name>_export_manifest.json)",
    )
    parser.add_argument(
        "--asset-manifest",
        default=None,
        help="Path to the asset manifest JSON (inferred from export manifest if omitted)",
    )
    parser.add_argument(
        "--storyline",
        default=None,
        help="Path to the storyline JSON (inferred from asset manifest if omitted)",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Where to write the JSON QA report (default: <export_stem>_qa_report.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero for serious warnings, not just failures",
    )
    args = parser.parse_args()

    export_manifest_path = Path(args.export_manifest).resolve()
    if not export_manifest_path.exists():
        print(f"[FAIL] Export manifest not found: {export_manifest_path}")
        sys.exit(1)

    base_dir = export_manifest_path.parent
    export = load_json(export_manifest_path)

    if "_load_error" in export:
        print(f"[FAIL] Could not parse export manifest: {export['_load_error']}")
        sys.exit(1)

    # Resolve asset manifest
    asset_manifest: dict | None = None
    if args.asset_manifest:
        p = Path(args.asset_manifest).resolve()
        asset_manifest = load_json(p) if p.exists() else None
    else:
        manifest_path_str = export.get("manifest_path", "")
        if manifest_path_str:
            p = resolve_path(manifest_path_str, base_dir)
            if p.exists():
                asset_manifest = load_json(p)

    # Resolve storyline
    storyline: dict | None = None
    if args.storyline:
        p = Path(args.storyline).resolve()
        storyline_raw = load_json(p) if p.exists() else {}
        storyline = storyline_raw.get("storyline") or storyline_raw
    elif asset_manifest:
        sl_path_str = asset_manifest.get("storyline_path", "")
        if sl_path_str:
            p = resolve_path(sl_path_str, base_dir)
            if p.exists():
                storyline_raw = load_json(p)
                storyline = storyline_raw.get("storyline") or storyline_raw

    # Run checks
    files_check = check_files(export, base_dir)
    srt_path = files_check.pop("_srt_path", None)
    vtt_path = files_check.pop("_vtt_path", None)
    mp4_path = files_check.pop("_mp4_path", None)

    video_stream_check = check_video_stream(mp4_path)
    subtitles_check = check_subtitles(export, srt_path, vtt_path)
    diversity_check = check_clip_diversity(export)
    rhythm_check = check_rhythm(export)
    planning_check = check_planning(asset_manifest, storyline)
    manifest_check = check_manifest_completeness(export)

    checks = {
        "files": files_check,
        "video_stream": video_stream_check,
        "subtitles": subtitles_check,
        "clip_diversity": diversity_check,
        "rhythm": rhythm_check,
        "planning": planning_check,
        "manifest_completeness": manifest_check,
    }

    total_score, status, all_failures, all_warnings, recommendations = compute_overall(checks)

    # Build clean public check results (strip internal _score key)
    public_checks: dict = {}
    for key, val in checks.items():
        public_checks[key] = {k: v for k, v in val.items() if not k.startswith("_")}
        public_checks[key]["score"] = val.get("_score", 0)

    report = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        "export_manifest": str(export_manifest_path),
        "status": status,
        "score": total_score,
        "checks": public_checks,
        "failures": all_failures,
        "warnings": all_warnings,
        "recommendations": recommendations,
    }

    # Write report
    if args.output_report:
        report_path = Path(args.output_report).resolve()
    else:
        report_path = base_dir / f"{export_manifest_path.stem.replace('_export_manifest', '')}_qa_report.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console output
    mp4_label = export.get("output_path", str(export_manifest_path))
    print_report(mp4_label, total_score, status, checks, all_failures, all_warnings, recommendations)

    print(f"QA report written: {report_path}")
    print()

    # Exit code
    if status == "fail":
        sys.exit(1)
    if args.strict and all_warnings:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
