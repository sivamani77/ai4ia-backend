from __future__ import annotations
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipeline.media_probe import (
    media_duration_or_raise as get_media_duration,
    video_stream_duration as get_video_stream_duration,
)
from app.pipeline.export_gen.ffmpeg_utils import log, run_cmd, _vcodec_flags
from app.pipeline.export_gen.manifest_io import find_source_video, section_voiceover_by_type_and_index, slugify, source_video_lookup
from app.pipeline.export_gen.clip_render import (
    ARCHIVE_AUDIO_VOLUME,
    CLOSING_CARD_MIN_DURATION,
    DEFAULT_CLIP_DURATION,
    FPS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SEGMENT_AUDIO_CODEC,
    SEGMENT_CONTAINER_SUFFIX,
    STATIC_CARD_DURATION,
    TITLE_CARD_MIN_DURATION,
    _first_present,
    build_lower_third,
    build_video_filter,
    get_clip_candidate_duration,
    make_card_video_with_optional_voiceover,
    render_archive_clip,
)
from app.pipeline.export_gen.timeline_selection import get_canonical_export_timeline, normalize_canonical_timeline_item
from app.pipeline.export_gen.subtitles import format_vtt_timestamp, get_segment_subtitle_text, make_subtitle_entries_for_segment, segment_has_spoken_audio


# A rendered segment's video stream may legitimately end a fraction of a frame
# before its container (last frame vs last audio packet, 1/30s quantisation).
# Two frames of slack; anything beyond that is real missing picture.
_SEGMENT_AV_TOLERANCE = 0.10


def _segment_video_duration(path: Path) -> float:
    """Seconds of actual picture in a rendered segment, from the frame count.

    Not media_probe.video_stream_duration: segments are Matroska (see
    SEGMENT_CONTAINER_SUFFIX), and Matroska carries no per-stream duration, so
    that helper returns 0.0 for every segment this pipeline writes — a check
    built on it silently never runs. Counting packets works on any container,
    and segments are always exactly FPS by construction.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return int(out) / FPS if out else 0.0
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


class ExportIntegrityError(RuntimeError):
    """A timeline segment cannot be rendered as planned.

    Raised instead of silently skipping content: a skipped segment shortens the
    final video below the validated plan, which is a contract violation. The fix
    belongs upstream (re-run Step 08, restore the missing media), never in
    dropping content at render time.
    """


def _build_canonical_rendering_timeline(
    manifest: dict[str, Any],
    video_mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build the rendering timeline directly from the canonical export_timeline stored in the
    manifest (written by Step 07 and enriched by Step 08).

    Cards (title_card / closing_card) are resolved from the manifest cards dict.
    Section voiceovers are attached to narration_broll items where an audio file exists.

    Returns (rendering_timeline, quality_checks).
    """
    canonical = get_canonical_export_timeline(manifest) or []
    sl = source_video_lookup(manifest)
    # Extended map: keyed by section_id, section_type, section_role, and positional index
    sv_map = section_voiceover_by_type_and_index(manifest)
    cards = manifest.get("cards", {})
    title_card_path = str(cards.get("title_card", ""))
    closing_card_path = str(cards.get("closing_card", cards.get("title_card", "")))

    # For chronological_relay: build section_voiceovers list for bridge narration lookup
    _sv_list = (manifest.get("voice") or {}).get("section_voiceovers") or []

    rendering: list[dict[str, Any]] = []
    archive_visual_count = 0
    speaker_clip_count = 0
    narration_broll_count = 0
    card_count = 0
    skipped_count = 0
    fragment_item_index = 0  # 0-based counter for all fragment items (logging only)
    voiceover_slot_index = 0  # 0-based counter for narration/archive items only (positional voiceover matching)
    _relay_section_idx = 0  # track section index for bridge narration in chronological_relay
    used_voiceover_paths: set[str] = set()

    for idx, raw_item in enumerate(canonical, start=1):
        item = normalize_canonical_timeline_item(raw_item, manifest, idx, source_lookup=sl)
        seg_type = item.get("segment_type", "")
        item_type = item.get("type", "")

        if item_type == "year_stamp" or seg_type == "year_stamp" or raw_item.get("type") == "year_stamp":
            year_val = raw_item.get("year") or item.get("year")
            year_stamp_cards = cards.get("year_stamp_cards") or []
            matched = next((c for c in year_stamp_cards if c.get("year") == year_val), None)
            if not matched:
                log(f"  [canonical] year_stamp item {idx} (year={year_val}) — card PNG not in manifest, skipping.")
                skipped_count += 1
                continue
            card_path = matched.get("path", "")
            if not card_path or not Path(card_path).exists():
                log(f"  [canonical] year_stamp item {idx} (year={year_val}) — card PNG missing at {card_path!r}, skipping.")
                skipped_count += 1
                continue
            item["type"] = "card"
            item["segment_type"] = "year_stamp"
            item["path"] = card_path
            item["audio_path"] = None
            item["audio_mode"] = "silent"
            item["muted"] = True
            item.setdefault("duration", float(raw_item.get("duration") or 2.5))
            item.setdefault("section_id", f"YEAR_{year_val}")
            item.setdefault("label", f"year_stamp_{year_val}")
            item["video_year"] = None
            card_count += 1
            rendering.append(item)

        elif item_type == "card" or seg_type in ("title_card", "closing_card"):
            # Resolve card image path
            if seg_type == "title_card":
                card_path = title_card_path
            elif seg_type == "closing_card":
                card_path = closing_card_path
            else:
                card_path = item.get("path", "")

            if not card_path or not Path(card_path).exists():
                raise ExportIntegrityError(
                    f"Card item {idx} ({seg_type}): card PNG missing at {card_path!r}. "
                    f"Re-run Step 08 to regenerate cards."
                )

            item["type"] = "card"
            item["path"] = card_path
            item.setdefault("audio_path", None)
            item.setdefault("duration", TITLE_CARD_MIN_DURATION if seg_type == "title_card" else CLOSING_CARD_MIN_DURATION)
            item.setdefault("section_id", seg_type.upper())
            item.setdefault("label", seg_type)
            item.setdefault("subtitle_text", manifest.get("title", ""))
            card_count += 1
            rendering.append(item)

        elif item_type == "archive_clip" or item.get("fragment_id"):
            item["type"] = "archive_clip"
            item_audio_mode = item.get("audio_mode") or ""

            if seg_type == "speaker_clip" or item_audio_mode == "source_audio":
                # Phase 4Q: speaker_clip plays with original source audio — never attach voiceover
                item["segment_type"] = "speaker_clip"
                item["audio_mode"] = "source_audio"
                item["muted"] = False
                if not item.get("subtitle_text"):
                    item["subtitle_text"] = item.get("moment_text", "")
                item.setdefault("subtitle_kind", "speaker")
                if not item.get("lower_third"):
                    item["lower_third"] = build_lower_third(item)
                seg_type = "speaker_clip"
                speaker_clip_count += 1
                log(f"  [canonical] Speaker clip item {idx}: {item.get('fragment_id', '')[:40]!r}")
                fragment_item_index += 1
                rendering.append(item)

                # chronological_relay: insert narration bridge BETWEEN speaker clips
                if video_mode == "chronological_relay":
                    # Check if the next canonical item is a closing_card — skip bridge at the very end
                    _next_raw = canonical[idx] if idx < len(canonical) else None
                    _next_type = (_next_raw or {}).get("type", "")
                    _next_seg = (_next_raw or {}).get("segment_type", "")
                    _is_final = _next_type == "card" and _next_seg == "closing_card"
                    if not _is_final:
                        _bridge_sv = _sv_list[_relay_section_idx] if _relay_section_idx < len(_sv_list) else None
                        _bridge_audio_path_str = (_bridge_sv or {}).get("audio_path", "")
                        _bridge_audio = Path(_bridge_audio_path_str) if _bridge_audio_path_str else None
                        _bridge_text = (_bridge_sv or {}).get("text", "")
                        if _bridge_audio and _bridge_audio.exists():
                            _bridge_dur = get_media_duration(_bridge_audio)
                            if _bridge_dur <= 0.0:
                                _wc = len(_bridge_text.split()) if _bridge_text else 0
                                _bridge_dur = max(2.5, _wc / 2.5)
                        else:
                            _wc = len(_bridge_text.split()) if _bridge_text else 0
                            _bridge_dur = max(2.5, _wc / 2.5)
                            _bridge_audio = None
                        # Bridges have no archive footage — render over the title card
                        # so they are always renderable (never silently dropped).
                        _bridge_visual = None
                        if title_card_path and Path(title_card_path).exists():
                            _bridge_visual = title_card_path
                        _bridge_seg: dict[str, Any] = {
                            "type": "archive_clip",
                            "segment_type": "narration_broll",
                            "audio_mode": "external",
                            "audio_path": str(_bridge_audio) if _bridge_audio else None,
                            "render_visual_path": _bridge_visual,
                            "visual_path": _bridge_visual,
                            "muted": True,
                            "subtitle_text": _bridge_text,
                            "voiceover_text": _bridge_text,
                            "subtitle_kind": "voiceover",
                            "label": f"narration_bridge_{_relay_section_idx}",
                            "duration": round(_bridge_dur + 0.4, 2),
                            "render_duration": round(_bridge_dur + 0.4, 2),
                            "source_video": None,
                            "fragment_id": None,
                            "video_year": None,
                            "section_id": (_bridge_sv or {}).get("section_id", ""),
                        }
                        rendering.append(_bridge_seg)
                        narration_broll_count += 1
                        log(f"  [canonical] narration_bridge_{_relay_section_idx}: {_bridge_dur:.2f}s, audio={'yes' if _bridge_audio else 'estimated'}")
                    _relay_section_idx += 1
                continue  # skip the common fragment_item_index += 1 / rendering.append(item) below
            else:
                # archive_visual / narration_broll: try to attach a section voiceover.
                # Priority: pre-set audio_path (from Step 08) > section_id > section_role > section_type > positional index.
                audio_path: Path | None = None

                # Check pre-set audio_path written by Step 08's build_hybrid_export_timeline
                preset_audio = item.get("audio_path") or item.get("voiceover_path")
                if preset_audio:
                    preset_p = Path(preset_audio)
                    if preset_p.exists():
                        audio_path = preset_p

                if audio_path is None:
                    for key in (
                        item.get("section_id"),
                        item.get("section_role"),
                        item.get("section_type"),
                        str(voiceover_slot_index),
                    ):
                        if key and key in sv_map:
                            candidate_ap = sv_map[key]
                            if str(candidate_ap.resolve()) not in used_voiceover_paths:
                                audio_path = candidate_ap
                                break

                if audio_path and str(audio_path.resolve()) not in used_voiceover_paths:
                    used_voiceover_paths.add(str(audio_path.resolve()))
                    item["audio_path"] = str(audio_path)
                    item["segment_type"] = "narration_broll"
                    item["audio_mode"] = "ai_voiceover"
                    probed_audio_dur = get_media_duration(audio_path)
                    preset_render_dur = float(item.get("render_duration") or 0.0)
                    if preset_render_dur > 0.0:
                        if probed_audio_dur > preset_render_dur + 1.0:
                            log(
                                f"  [canonical] WARNING: voiceover {audio_path.name} probed as "
                                f"{probed_audio_dur:.1f}s exceeding declared {preset_render_dur:.1f}s — updating duration to {probed_audio_dur + 0.4:.1f}s"
                            )
                            item["duration"] = probed_audio_dur + 0.4
                        else:
                            item["duration"] = preset_render_dur
                    else:
                        if probed_audio_dur <= 0.0:
                            log(f"  [canonical] WARNING: voiceover {audio_path.name} probed as 0s — using fragment duration")
                        narration_duration = max(probed_audio_dur, 0.0) + 0.4
                        frag_dur = float(item.get("duration") or 0.0)
                        item["duration"] = max(frag_dur, narration_duration)
                        item["render_duration"] = item["duration"]
                    if not item.get("subtitle_text") or not item.get("voiceover_text"):
                        for sv in (manifest.get("voice", {}).get("section_voiceovers") or []):
                            sv_ap = sv.get("audio_path", "")
                            if sv_ap and Path(sv_ap) == audio_path:
                                text = sv.get("text", "")
                                if not item.get("subtitle_text"):
                                    item["subtitle_text"] = text
                                if not item.get("voiceover_text"):
                                    item["voiceover_text"] = text
                                item.setdefault("subtitle_kind", "voiceover")
                                break
                    narration_broll_count += 1
                    log(f"  [canonical] Attached voiceover to item {idx} (voiceover_slot_index={voiceover_slot_index}): {audio_path.name} ({probed_audio_dur:.2f}s → render {item['duration']:.2f}s)")
                else:
                    item["audio_path"] = None
                    item["segment_type"] = "archive_visual"
                    item["audio_mode"] = "muted"
                    item["muted"] = True
                    archive_visual_count += 1
                    log(f"  [canonical] Archive visual item {idx}: {item.get('fragment_id', '')[:40]!r}")

            fragment_item_index += 1
            voiceover_slot_index += 1
            rendering.append(item)
        else:
            raise ExportIntegrityError(
                f"Unrecognised timeline item {idx} (type={item_type!r}, seg={seg_type!r}) — "
                f"cannot render it and refusing to drop it. Fix the manifest timeline."
            )

    log(f"Canonical timeline summary:")
    log(f"  total items   : {len(canonical)}")
    log(f"  cards         : {card_count}")
    log(f"  speaker_clips : {speaker_clip_count}")
    log(f"  archive_visuals: {archive_visual_count}")
    log(f"  narration_broll: {narration_broll_count}")
    log(f"  skipped       : {skipped_count}")

    quality_checks: dict[str, Any] = {
        "timeline_source": "canonical_export_timeline",
        "canonical_items": len(canonical),
        "card_count": card_count,
        "speaker_clip_count": speaker_clip_count,
        "archive_visual_count": archive_visual_count,
        "narration_broll_count": narration_broll_count,
        "skipped_count": skipped_count,
    }
    return rendering, quality_checks


@dataclass
class SegmentPlan:
    """A fully-resolved rendering decision for one timeline segment.

    Produced by the PURE planner _plan_segment() (no ffmpeg, unit-testable) and
    consumed by the thin executor _execute_plan(). `kind` selects the executor
    branch; the remaining fields carry everything that branch needs.
    """
    kind: str  # "skip" | "card" | "black_source_audio" | "generated_visual" | "archive"
    segment_path: Path
    seg_type: str
    fragment_id: str | None = None
    # execution inputs (populated per kind)
    image_path: Path | None = None
    source_path: Path | None = None
    audio_path: Path | None = None            # card voiceover / narration external audio
    min_duration: float = 0.0                 # card / generated_visual
    start: float = 0.0
    end: float = 0.0
    target_duration: float | None = None
    role: str = ""
    audio_mode: str = ""                       # archive: source | muted | external
    lower_third: dict[str, str] | None = None
    # info-dict extras
    subtitle_text: str = ""
    muted: bool = False
    visual_mode: str | None = None
    bandaid_note: str | None = None
    stream_loop: bool = False


def _plan_segment(idx: int, item: dict[str, Any], work_dir: Path) -> SegmentPlan:
    """PURE planning: resolve one timeline item into a SegmentPlan.

    Decides start/end/target_duration/audio_mode, detects footage bandaids, and
    picks the executor `kind` — WITHOUT invoking ffmpeg. Raises ExportIntegrityError
    for unrenderable items (the same hard-fail contract as before). May probe media
    durations (get_media_duration / get_video_stream_duration), which are cached.
    """
    segment_label = slugify(item.get("label") or item.get("fragment_id") or f"segment_{idx}", 70)
    segment_path = work_dir / f"{idx:03d}_{segment_label}{SEGMENT_CONTAINER_SUFFIX}"

    item_type = item.get("type", "")
    seg_type = item.get("segment_type", "")
    fragment_id = item.get("fragment_id")

    if item_type == "card":
        image_path = Path(item.get("path", ""))
        audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
        min_duration = float(item.get("duration") or STATIC_CARD_DURATION)

        if not image_path.exists():
            if seg_type == "year_stamp":
                # Cosmetic transition card — safe to drop, but say so loudly.
                log(f"  [render] WARNING: year_stamp card image missing, dropping cosmetic segment: {image_path}")
                return SegmentPlan(kind="skip", segment_path=segment_path, seg_type=seg_type, fragment_id=fragment_id)
            raise ExportIntegrityError(
                f"Segment {idx} ({seg_type or 'card'}): card image missing at {image_path}. "
                f"Re-run Step 08 to regenerate cards."
            )

        return SegmentPlan(
            kind="card", segment_path=segment_path, seg_type=seg_type, fragment_id=fragment_id,
            image_path=image_path, audio_path=audio_path, min_duration=min_duration,
            muted=item.get("muted", False),
        )

    if not (
        item_type == "archive_clip"
        or item.get("fragment_id")
        or seg_type in ("narration_broll", "speaker_clip", "archive_visual", "speaker_visual")
        or item_type in ("narration_broll", "speaker_clip", "archive_visual", "speaker_visual")
    ):
        raise ExportIntegrityError(
            f"Segment {idx}: unknown timeline item type {item_type!r} — cannot render it "
            f"and refusing to drop it. Fix the manifest timeline."
        )

    source_video = item.get("source_video")
    role = item.get("role", seg_type or "archive_visual")
    source_path = find_source_video(source_video, fragment_id)
    item_audio_mode = item.get("audio_mode") or ""

    _has_generated_visual = bool(
        item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path")
    )
    _is_narration = seg_type == "narration_broll" or item_type == "narration_broll" or item_audio_mode == "ai_voiceover"
    if not source_path and not (_is_narration and _has_generated_visual):
        raise ExportIntegrityError(
            f"Segment {idx} ({seg_type}): source video not found for fragment {fragment_id!r} "
            f"(source_video={source_video!r}) and no generated visual to render instead. "
            f"Restore the source video or re-run Step 08."
        )

    if item_type not in ("narration_broll", "speaker_clip"):
        item["type"] = "archive_clip"

    lower_third = item.get("lower_third")
    video_stream_dur = get_video_stream_duration(source_path) if source_path else 0.0

    # ---- speaker_clip / source_audio ----------------------------------------
    if seg_type == "speaker_clip" or item_audio_mode == "source_audio":
        start = _first_present(item.get("render_start"), item.get("start"))
        end = _first_present(item.get("render_end"), item.get("end"), default=start + DEFAULT_CLIP_DURATION)
        if end <= start:
            end = start + DEFAULT_CLIP_DURATION
        target_duration = float(item.get("render_duration") or (end - start))

        bandaid_note = None
        if video_stream_dur > 0 and (start + target_duration) > (video_stream_dur - 0.5):
            if video_stream_dur >= target_duration + 0.5:
                fallback_start = max(0.0, video_stream_dur - target_duration - 0.5)
                bandaid_note = (
                    f"speaker_clip {str(fragment_id)[:40]!r}: window {start:.1f}s..{start+target_duration:.1f}s "
                    f"exceeds video track ({video_stream_dur:.1f}s) — shifted window back to {fallback_start:.1f}s."
                )
                log(f"  [render] WARNING (footage-bandaid): {bandaid_note}")
                start = fallback_start
                end = fallback_start + target_duration
            else:
                bandaid_note = (
                    f"speaker_clip {str(fragment_id)[:40]!r}: video track ({video_stream_dur:.1f}s) "
                    f"shorter than quote ({target_duration:.1f}s) — rendering clean black visual + source audio."
                )
                log(f"  [render] WARNING (footage-bandaid): {bandaid_note}")
                return SegmentPlan(
                    kind="black_source_audio", segment_path=segment_path, seg_type=seg_type, fragment_id=fragment_id,
                    source_path=source_path, start=start, target_duration=target_duration,
                    lower_third=lower_third, muted=item.get("muted", False), bandaid_note=bandaid_note,
                )

        return SegmentPlan(
            kind="archive", segment_path=segment_path, seg_type=seg_type, fragment_id=fragment_id,
            source_path=source_path, start=start, end=end, target_duration=target_duration,
            role=role, audio_mode="source", lower_third=lower_third, muted=item.get("muted", False),
            bandaid_note=bandaid_note,
        )

    # ---- narration_broll / ai_voiceover -------------------------------------
    if seg_type == "narration_broll" or item_audio_mode == "ai_voiceover":
        # No start/end here: narration has no footage window to seek into — its
        # picture is a generated still, so only its duration matters.
        external_audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
        if not (external_audio_path and external_audio_path.exists()):
            if external_audio_path:
                log(f"  [render] External audio not found: {external_audio_path}. Rendering the visual silent.")
            else:
                log(f"  [render] narration_broll item {idx} ({fragment_id!r}) has no audio_path. Rendering the visual silent.")
            external_audio_path = None
        tts_duration = float(
            item.get("render_duration") or item.get("duration") or get_clip_candidate_duration(item)
        )
        if tts_duration <= 0.0 and external_audio_path and external_audio_path.exists():
            tts_duration = get_media_duration(external_audio_path)
        if tts_duration <= 0.0:
            raise RuntimeError(
                f"narration_broll item {idx} ({fragment_id!r}) has zero target_duration — "
                f"audio_path={item.get('audio_path')!r}. Aborting export."
            )
        if external_audio_path and external_audio_path.exists():
            real_audio_dur = get_media_duration(external_audio_path)
            if real_audio_dur > 0.0 and tts_duration > real_audio_dur + 2.0:
                log(
                    f"  [render] WARNING: narration_broll item {idx} ({fragment_id!r}) declares "
                    f"{tts_duration:.1f}s but voiceover is only {real_audio_dur:.1f}s — clamping to "
                    f"{real_audio_dur + 0.4:.1f}s to avoid trailing silence. Re-run Step 08 to fix the manifest."
                )
                tts_duration = real_audio_dur + 0.4
        target_duration = tts_duration

        # Narration ALWAYS plays over its generated visual, never over archive
        # footage — archive footage is reserved exclusively for speaker_clip, so
        # a viewer never sees real footage except when a real person is speaking.
        # Step 08 generates a visual for every narration_broll item, so a missing
        # one is a manifest defect: fail loudly instead of quietly reaching for
        # footage that this pipeline has decided narration must not use.
        _vis_path_str = (
            item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path")
        )
        if not _vis_path_str or not Path(_vis_path_str).exists():
            raise ExportIntegrityError(
                f"Segment {idx} (narration_broll, fragment={fragment_id!r}): no generated "
                f"visual at {_vis_path_str!r}. Narration never renders over archive footage, "
                f"so there is nothing to put on screen. Re-run Step 08 to generate it."
            )
        return SegmentPlan(
            kind="generated_visual", segment_path=segment_path, seg_type="narration_broll", fragment_id=fragment_id,
            image_path=Path(_vis_path_str), audio_path=external_audio_path, min_duration=target_duration,
            subtitle_text=item.get("voiceover_text") or item.get("subtitle_text") or "",
            muted=False, visual_mode="generated_asset",
        )

    # ---- archive_visual / speaker_visual / default --------------------------
    start = float(item.get("start") or 0)
    end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))
    if end <= start:
        end = start + DEFAULT_CLIP_DURATION
    if seg_type == "archive_visual" or item_audio_mode == "silent" or item.get("muted") is True:
        audio_mode = "muted"
        lower_third = None
    else:
        audio_mode = "source"

    bandaid_note = None
    stream_loop = False
    clip_span = max(end - start, 1.0)
    if video_stream_dur > 0 and (start + clip_span) > (video_stream_dur - 0.5):
        if video_stream_dur >= clip_span + 0.5:
            fallback_start = max(0.0, video_stream_dur - clip_span - 0.5)
            bandaid_note = (
                f"segment {idx} ({seg_type or item_type}, {str(fragment_id)[:40]!r}): window {start:.1f}s..{start+clip_span:.1f}s "
                f"exceeds video track ({video_stream_dur:.1f}s) — shifted window to {fallback_start:.1f}s."
            )
            log(f"  [render] WARNING (footage-bandaid): {bandaid_note}")
            end = fallback_start + clip_span
            start = fallback_start
        else:
            stream_loop = True
            bandaid_note = (
                f"segment {idx} ({seg_type or item_type}, {str(fragment_id)[:40]!r}): source video track ({video_stream_dur:.1f}s) "
                f"shorter than clip span ({clip_span:.1f}s) — enabling video stream looping."
            )
            log(f"  [render] WARNING (footage-bandaid): {bandaid_note}")
            start = 0.0
            end = video_stream_dur

    return SegmentPlan(
        kind="archive", segment_path=segment_path, seg_type=seg_type, fragment_id=fragment_id,
        source_path=source_path, start=start, end=end, target_duration=None,
        role=role, audio_mode=audio_mode, lower_third=lower_third,
        muted=item.get("muted", False), bandaid_note=bandaid_note, stream_loop=stream_loop,
    )


def _execute_plan(plan: SegmentPlan) -> None:
    """THIN execution: turn a SegmentPlan into the actual ffmpeg render call(s).

    No planning decisions here — every start/end/duration/audio_mode is already
    resolved. One branch per plan.kind.
    """
    if plan.kind == "card":
        make_card_video_with_optional_voiceover(
            image_path=plan.image_path,
            output_path=plan.segment_path,
            min_duration=plan.min_duration,
            audio_path=plan.audio_path,
        )
    elif plan.kind == "generated_visual":
        log(f"  [render] narration_broll: generated visual {plan.image_path.name} + ai_voiceover (min_dur={plan.min_duration:.1f}s)")
        make_card_video_with_optional_voiceover(
            image_path=plan.image_path,
            output_path=plan.segment_path,
            min_duration=plan.min_duration,
            audio_path=plan.audio_path,
        )
    elif plan.kind == "black_source_audio":
        log(f"  [render] WARNING (footage-bandaid): {plan.bandaid_note}")
        td = plan.target_duration
        vf = build_video_filter(duration=td, lower_third=plan.lower_third)
        run_cmd([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=black:size={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:duration={td:.3f}:rate={FPS}",
            "-ss", str(plan.start),
            "-i", str(plan.source_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(td),
            "-vf", vf,
            "-filter:a", f"volume={ARCHIVE_AUDIO_VOLUME},afade=t=in:st=0:d=0.15,afade=t=out:st={max(td - 0.25, 0):.3f}:d=0.25",
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC, "-ar", "48000", "-ac", "2",
            str(plan.segment_path),
        ])
    elif plan.kind == "archive":
        if plan.bandaid_note is None and plan.audio_mode == "source" and plan.seg_type == "speaker_clip":
            log(
                f"  [render] speaker_clip {plan.fragment_id[:40] if plan.fragment_id else '?'!r}: "
                f"source_audio {plan.start:.2f}→{plan.end:.2f} ({(plan.target_duration or 0.0):.2f}s)"
            )
        render_archive_clip(
            source_path=plan.source_path,
            output_path=plan.segment_path,
            start=plan.start,
            end=plan.end,
            role=plan.role,
            audio_mode=plan.audio_mode,
            external_audio_path=plan.audio_path,
            target_duration=plan.target_duration,
            lower_third=plan.lower_third,
            stream_loop=plan.stream_loop,
        )
    else:
        raise RuntimeError(f"_execute_plan: unhandled plan kind {plan.kind!r}")


class HLSStreamWriter:
    """Progressively mux rendered segments into an HLS stream (playlist +
    .ts chunks) so a client can start playback before the whole export
    finishes, instead of waiting for concat_segments() to produce one
    final mp4.

    Segments are h264 video + LOSSLESS PCM audio (see SEGMENT_AUDIO_CODEC),
    so the video still copies but the audio must be encoded to AAC here —
    MPEG-TS cannot carry PCM. This is the one place that still encodes AAC
    per segment, and that is correct for HLS: each .ts chunk is a standalone
    unit whose priming a player trims from its own container. The final mp4
    takes the other path, where concat_segments encodes AAC exactly once.

    add_segment() must be called in ascending timeline order — the caller
    (_render_timeline_items) buffers out-of-order completions (segments
    render in parallel and can finish in any order) and drains them in
    order.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[tuple[str, float]] = []
        self._subtitle_entries: list[dict[str, Any]] = []
        self._next_seg_num = 0
        self._closed = False

    def add_segment(
        self,
        segment_path: Path,
        duration: float,
        timeline_item: dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        if self._closed:
            return
        seg_name = f"seg_{self._next_seg_num:04d}.ts"
        seg_path = self.output_dir / seg_name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(segment_path),
                    "-c:v", "copy",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-f", "mpegts", str(seg_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            log(f"  [hls] WARNING: failed to remux segment {self._next_seg_num} to .ts, skipping in stream: {exc}")
            return

        self._next_seg_num += 1
        self._entries.append((seg_name, duration))

        if segment_has_spoken_audio(timeline_item):
            text = get_segment_subtitle_text(timeline_item)
            if text:
                self._subtitle_entries.extend(
                    make_subtitle_entries_for_segment(
                        text=text, segment_start=start_time, segment_end=end_time,
                    )
                )

        self._write_playlist()
        self._write_subs()
        log(f"  [hls] {seg_name} ready (stream position {end_time:.1f}s) — playlist updated")

    def _write_playlist(self) -> None:
        target_dur = max([2] + [int(d) + 1 for _, d in self._entries])
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_dur}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:EVENT",
        ]
        if self._subtitle_entries:
            lines.append('#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",DEFAULT=YES,AUTOSELECT=YES,FORCED=NO,LANGUAGE="en",URI="subs.vtt"')
        for i, (name, dur) in enumerate(self._entries):
            if i > 0:
                # Every segment is encoded independently (fresh ffmpeg process
                # per timeline item, starting at PTS 0) — there is no shared
                # timeline across segments, so every boundary is a genuine
                # discontinuity. Without this tag, players' internal
                # transmuxers (raw MPEG-TS isn't MSE-native — this is what
                # actually feeds the browser's SourceBuffer) read the PTS
                # reset at segment 2+ as corrupt/out-of-order timestamps and
                # silently drop audio from there on, while video keeps
                # displaying (audio buffering is stricter about monotonic
                # continuity than video frame display) — confirmed live:
                # exactly "segment 1 audio plays, video fine after that but
                # silent."
                lines.append("#EXT-X-DISCONTINUITY")
            lines.append(f"#EXTINF:{dur:.3f},")
            lines.append(name)
        if self._closed:
            lines.append("#EXT-X-ENDLIST")
        tmp = self.output_dir / "master.m3u8.tmp"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.output_dir / "master.m3u8")

    def _write_subs(self) -> None:
        if not self._subtitle_entries:
            return
        lines = ["WEBVTT", ""]
        for entry in self._subtitle_entries:
            lines.append(f"{format_vtt_timestamp(float(entry['start']))} --> {format_vtt_timestamp(float(entry['end']))}")
            lines.append(entry["text"])
            lines.append("")
        tmp = self.output_dir / "subs.vtt.tmp"
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(self.output_dir / "subs.vtt")

    def finalize(self, success: bool) -> None:
        self._closed = True
        self._write_playlist()
        marker = "_success" if success else "_failed"
        (self.output_dir / marker).write_text("", encoding="utf-8")
        log(f"  [hls] stream finalized (success={success}, segments={len(self._entries)})")


def _render_one_segment(
    idx: int,
    item: dict[str, Any],
    work_dir: Path,
) -> tuple[int, Path | None, dict[str, Any] | None]:
    """Render one timeline segment: plan (pure) → execute (ffmpeg) → info dict.

    Returns (idx, path, partial_info) or (idx, None, None) to skip.
    """
    plan = _plan_segment(idx, item, work_dir)
    if plan.kind == "skip":
        return idx, None, None

    _execute_plan(plan)

    duration = get_media_duration(plan.segment_path)
    if duration == 0.0:
        seg_type_check = item.get("segment_type") or item.get("type") or "?"
        raise RuntimeError(
            f"Rendered segment {idx} ({seg_type_check}, fragment={item.get('fragment_id')!r}) "
            f"has duration 0.0 at {plan.segment_path}. Aborting export."
        )

    # The container duration above is max(video, audio). A segment whose VIDEO
    # stream is shorter than that still reports a healthy duration here, gets a
    # correct-looking slot on the timeline, and then loses exactly that much
    # picture during concat — shifting every later visual earlier while the audio
    # stays put. Catch it on the segment that caused it, by name, instead of
    # discovering a 4.9s whole-export desync at the end (or not at all).
    video_duration = _segment_video_duration(plan.segment_path)
    if video_duration > 0.0 and (duration - video_duration) > _SEGMENT_AV_TOLERANCE:
        raise ExportIntegrityError(
            f"Rendered segment {idx} ({plan.seg_type}, fragment={item.get('fragment_id')!r}) "
            f"has {video_duration:.3f}s of video but a {duration:.3f}s container "
            f"({duration - video_duration:.3f}s of picture missing) at {plan.segment_path}. "
            f"Concat would drop that picture and run every later visual early."
        )

    # Log the frame count for EVERY segment, not just failing ones, and record it
    # in the manifest. When an export comes back short there is otherwise no way
    # to tell afterwards whether the frames were missing as each segment was
    # written or were dropped during concat: the .mkv segments live in a temp dir
    # that is cleaned up, and the HLS .ts copies are truncated in their own right.
    # One number per segment in the log settles that question from the job record
    # alone, without needing to reproduce the run.
    log(
        f"  [segment-frames] segment {idx} ({plan.seg_type}): "
        f"{round(video_duration * FPS)} frames = {video_duration:.3f}s video / "
        f"{duration:.3f}s container"
    )

    return idx, plan.segment_path, {
        "index": idx,
        "path": str(plan.segment_path),
        "timeline_item": item,
        "duration": duration,
        "video_duration": round(video_duration, 3),
        "subtitle_text": plan.subtitle_text,
        "rendered": True,
        "segment_type": plan.seg_type,
        "fragment_id": item.get("fragment_id"),
        "muted": plan.muted,
        "footage_bandaid": plan.bandaid_note,
        **({"visual_mode": plan.visual_mode} if plan.visual_mode else {}),
    }


def _render_timeline_items(
    timeline: list[dict[str, Any]],
    work_dir: Path,
    target_sec_per_segment: float = 0.0,
    hls_writer: "HLSStreamWriter | None" = None,
) -> tuple[list[Path], list[dict[str, Any]], float]:
    """
    Render all timeline segments in parallel (up to 4 concurrent ffmpeg processes),
    then assemble results in original order with sequential timestamps.

    When hls_writer is given, segments are also fed to it in strict timeline
    order as they become available (segments can *finish* rendering out of
    order even though they must be *streamed* in order) — this is what lets a
    client start watching before the whole export is done.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _render_one_segment_timed(idx, item, work_dir):
        _t0 = time.time()
        result = _render_one_segment(idx, item, work_dir)
        log(f"  [segment-timing] segment {idx} ({item.get('segment_type') or item.get('type')}) rendered in {time.time() - _t0:.1f}s")
        return result

    MAX_WORKERS = min(4, max(1, len(timeline)))
    raw_results: dict[int, tuple[Path | None, dict[str, Any] | None]] = {}

    # HLS sequencing state: segments must be handed to hls_writer in ascending
    # idx order even though as_completed() yields them in completion order.
    _hls_next_idx = 1
    _hls_cumulative_time = 0.0

    _render_all_t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_render_one_segment_timed, idx, item, work_dir): idx
            for idx, item in enumerate(timeline, start=1)
        }
        for future in as_completed(futures):
            idx, path, info = future.result()  # re-raises any RuntimeError from render
            raw_results[idx] = (path, info)

            if hls_writer is not None:
                while _hls_next_idx in raw_results:
                    _p, _info = raw_results[_hls_next_idx]
                    if _p is not None and _info is not None:
                        _start = _hls_cumulative_time
                        _end = _start + _info["duration"]
                        hls_writer.add_segment(_p, _info["duration"], _info["timeline_item"], _start, _end)
                        _hls_cumulative_time = _end
                    _hls_next_idx += 1
    log(f"  [render] all {len(timeline)} segments rendered (wall-clock, {MAX_WORKERS} workers) in {time.time() - _render_all_t0:.1f}s")

    segment_paths: list[Path] = []
    segment_infos: list[dict[str, Any]] = []
    current_time = 0.0
    rendered_archive_visual = 0
    rendered_speaker = 0

    for idx in sorted(raw_results.keys()):
        path, info = raw_results[idx]
        if path is None or info is None:
            continue
        info["start_time"] = current_time
        info["end_time"] = current_time + info["duration"]
        current_time += info["duration"]
        seg_type = info.get("segment_type", "")
        if seg_type == "archive_visual":
            rendered_archive_visual += 1
        elif seg_type == "speaker_clip":
            rendered_speaker += 1
        segment_paths.append(path)
        segment_infos.append(info)

    log(f"  [render] rendered_archive_visual={rendered_archive_visual}  rendered_speaker={rendered_speaker}  total_segments={len(segment_paths)}")
    return segment_paths, segment_infos, current_time
