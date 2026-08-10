from __future__ import annotations
from pathlib import Path
from typing import Any

from app.core.settings import settings
from app.pipeline.media_probe import media_duration_or_raise as get_media_duration
from app.pipeline.export_gen.ffmpeg_utils import log
from app.pipeline.export_gen.manifest_io import get_story, section_card_map, section_voiceover_map, source_video_lookup
from app.pipeline.export_gen.clip_render import (
    CLOSING_CARD_MIN_DURATION,
    MIN_CARD_DURATION,
    STATIC_CARD_DURATION,
    TITLE_CARD_MIN_DURATION,
    build_lower_third,
    get_clip_candidate_duration,
)

MAX_USES_PER_SOURCE_VIDEO = settings.max_uses_per_source_video
MAX_USES_PER_SPEAKER = settings.max_uses_per_speaker
ALLOW_REPEAT_FRAGMENT = False


class ClipUsageTracker:
    def __init__(self) -> None:
        self.used_fragment_ids: set[str] = set()
        self.used_source_videos: dict[str, int] = {}
        self.used_speakers: dict[str, int] = {}
        self.last_source_video: str | None = None

    def _get_ids(self, clip: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
        source_video = clip.get("source_video") or clip.get("source_video_file") or clip.get("video_id")
        raw_fid = clip.get("fragment_id") or clip.get("id")
        if raw_fid:
            fragment_id = str(raw_fid)
        elif source_video:
            start = clip.get("start", "")
            end = clip.get("end", "")
            fragment_id = f"{source_video}:{start}:{end}"
        else:
            fragment_id = None
        speaker = clip.get("speaker") or clip.get("speaker_name")
        return fragment_id, source_video, speaker

    def can_use(self, clip: dict[str, Any]) -> tuple[bool, str | None]:
        fragment_id, source_video, speaker = self._get_ids(clip)

        if not ALLOW_REPEAT_FRAGMENT and fragment_id and fragment_id in self.used_fragment_ids:
            return False, "fragment_already_used"

        if source_video and source_video == self.last_source_video:
            return False, "same_source_video_as_previous"

        if source_video and self.used_source_videos.get(source_video, 0) >= MAX_USES_PER_SOURCE_VIDEO:
            return False, "source_video_overused"

        if speaker and self.used_speakers.get(speaker, 0) >= MAX_USES_PER_SPEAKER:
            return False, "speaker_overused"

        return True, None

    def mark_used(self, clip: dict[str, Any]) -> None:
        fragment_id, source_video, speaker = self._get_ids(clip)
        if fragment_id:
            self.used_fragment_ids.add(fragment_id)
        if source_video:
            self.used_source_videos[source_video] = self.used_source_videos.get(source_video, 0) + 1
            self.last_source_video = source_video
        if speaker:
            self.used_speakers[speaker] = self.used_speakers.get(speaker, 0) + 1


def score_clip_for_export(clip: dict[str, Any]) -> float:
    score = 0.0
    score += float(clip.get("quote_strength") or 0) * 3
    score += float(clip.get("clarity_score") or 0) * 2
    score += float(clip.get("narration_support_score") or 0)
    return score


def select_best_unique_clip(
    candidates: list[dict[str, Any]],
    tracker: ClipUsageTracker,
    role: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Return (clip, selection_reason, fallback_reason)."""
    ranked = sorted(candidates, key=score_clip_for_export, reverse=True)
    last_rejection: str | None = None
    for clip in ranked:
        ok, reason = tracker.can_use(clip)
        if ok:
            chosen = dict(clip)
            chosen["role"] = role
            return chosen, "unique_" + role, None
        last_rejection = reason
    return None, None, last_rejection or "no_candidates"


def choose_speaker_soundbite(
    section: dict[str, Any],
    tracker: ClipUsageTracker,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    soundbites = section.get("speaker_soundbites", [])
    return select_best_unique_clip(soundbites, tracker, "speaker_soundbite")


def choose_archive_visual(
    section: dict[str, Any],
    tracker: ClipUsageTracker,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    visuals = section.get("archive_visual_fragments", [])
    return select_best_unique_clip(visuals, tracker, "archive_visual")


def choose_broll_for_narration(
    section: dict[str, Any],
    tracker: ClipUsageTracker,
    target_duration: float,
    excluded_fragment_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    visuals = []
    excluded_fragment_ids = excluded_fragment_ids or set()

    for item in section.get("archive_visual_fragments", []):
        fragment_id = str(item.get("fragment_id") or item.get("id") or "")
        if fragment_id and fragment_id in excluded_fragment_ids:
            continue
        if get_clip_candidate_duration(item) >= target_duration:
            visuals.append(item)

    return select_best_unique_clip(visuals, tracker, "narration_broll")


def get_canonical_export_timeline(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Return the canonical export_timeline from the storyline embedded in the manifest.

    Preference order:
      1. manifest["storyline"]["export_timeline"]
      2. manifest["export_timeline"]

    Returns None if no canonical timeline exists or it has no fragment-linked items.
    """
    storyline = manifest.get("storyline")
    if isinstance(storyline, dict):
        tl = storyline.get("export_timeline")
        if isinstance(tl, list) and tl:
            return tl

    tl = manifest.get("export_timeline")
    if isinstance(tl, list) and tl:
        return tl

    return None


def normalize_canonical_timeline_item(
    item: dict[str, Any],
    manifest: dict[str, Any],
    index: int,
    source_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Normalize a raw canonical timeline item into a fully-resolved rendering descriptor.

    Handles field aliases (type/timeline_type, duration/duration_seconds) and
    sets rendering defaults for each segment class.
    """
    out = dict(item)

    # Resolve type / timeline_type alias
    raw_type = item.get("type") or item.get("timeline_type") or ""

    # Resolve segment_type
    raw_seg = item.get("segment_type") or raw_type or ""

    # Map title_card / closing_card top-level types to card segment_type
    if raw_type in ("title_card", "closing_card"):
        out["type"] = "card"
        out["segment_type"] = raw_type
    elif raw_seg in ("title_card", "closing_card"):
        out["type"] = "card"
        out["segment_type"] = raw_seg
    elif raw_seg in ("speaker_clip", "archive_visual", "narration_broll"):
        out["type"] = "archive_clip"
        out["segment_type"] = raw_seg
    elif raw_type in ("speaker_clip", "archive_visual", "narration_broll"):
        out["type"] = "archive_clip"
        out["segment_type"] = raw_type
    elif item.get("fragment_id"):
        # Fragment-linked item without explicit segment_type → default archive_visual
        out["type"] = "archive_clip"
        out["segment_type"] = raw_seg or "archive_visual"
    else:
        out["type"] = item.get("type") or "unknown"
        out["segment_type"] = raw_seg

    # Resolve duration alias
    if out.get("duration") is None and item.get("duration_seconds") is not None:
        out["duration"] = item["duration_seconds"]

    # Resolve source_video from lookup when missing
    fragment_id = item.get("fragment_id")
    if fragment_id and not out.get("source_video") and source_lookup:
        out["source_video"] = source_lookup.get(fragment_id)

    seg_type = out["segment_type"]

    # Apply rendering defaults per segment class
    if seg_type == "speaker_clip":
        out.setdefault("muted", False)
        # subtitle_text = moment_text or subtitle_text (not full fragment text)
        moment_text = item.get("moment_text") or ""
        subtitle_text = item.get("subtitle_text") or moment_text
        out["subtitle_text"] = subtitle_text
        out.setdefault("subtitle_kind", "speaker")

    elif seg_type == "archive_visual":
        out["muted"] = True
        out["subtitle_text"] = ""
        out["subtitle_kind"] = "none"

    elif seg_type == "narration_broll":
        out["muted"] = True
        # Prefer pre-set subtitle_text; fall back to voiceover_text written by Step 08
        if not out.get("subtitle_text"):
            out["subtitle_text"] = item.get("voiceover_text") or item.get("subtitle_text") or ""
        out.setdefault("subtitle_kind", "voiceover")

    # Ensure label for segment naming
    if not out.get("label"):
        out["label"] = item.get("fragment_id") or f"segment_{index}"

    return out


def ensure_requested_speaker(manifest: dict[str, Any], timeline: list[dict[str, Any]]) -> list[str]:
    """If the prompt requested a specific speaker, ensure the timeline shows them.

    When the timeline has no speaker_clip/speaker_visual segment and a
    speaker_visual_allowed candidate exists, the best candidate is inserted
    after the title card. Runs on both the canonical and legacy timeline paths.
    Returns validation messages for quality_checks.
    """
    story = manifest.get("storyline") or {}
    prompt_intent = manifest.get("prompt_intent") or story.get("prompt_intent") or story.get("prompt_profile") or {}
    requested_speaker = prompt_intent.get("requested_speaker")
    issues: list[str] = []
    if not requested_speaker:
        return issues

    selected_candidates = manifest.get("selected_candidates") or story.get("selected_candidates") or []
    speaker_visual_candidates = [
        c for c in selected_candidates
        if c.get("speaker_visual_allowed") is True and c.get("final_video_allowed") is True
    ]
    timeline_has_speaker = any(
        item.get("segment_type") in ("speaker_clip", "speaker_visual")
        for item in timeline
        if item.get("type") == "archive_clip"
    )
    if not timeline_has_speaker and speaker_visual_candidates:
        # Repair: insert best speaker_visual candidate after title card
        best = speaker_visual_candidates[0]
        fragment_id = best.get("fragment_id")
        source_video = best.get("source_video") or best.get("video_id")
        repair_item = {
            "type": "archive_clip",
            "segment_type": "speaker_visual",
            "label": f"repaired_speaker_visual_{fragment_id}",
            "fragment_id": fragment_id,
            "source_video": source_video,
            "start": best.get("start", 0),
            "end": best.get("end", 0),
            "duration": best.get("duration"),
            "role": best.get("relevance_role", "exact"),
            "speaker": best.get("speaker") or best.get("speaker_name"),
            "subtitle_text": best.get("summary", ""),
            "selection_reason": "repair_speaker_visual_missing",
            "fallback_reason": None,
            "muted": False,
            "lower_third": None,
            "section_id": "REPAIR",
        }
        # Insert after title card (index 1)
        insert_pos = 1 if len(timeline) > 1 else len(timeline)
        timeline.insert(insert_pos, repair_item)
        msg = (
            f"[speaker-visual-repair] Timeline had no speaker segments for requested speaker "
            f"'{requested_speaker}'. Inserted speaker_visual fragment {fragment_id}."
        )
        log(msg)
        issues.append(msg)
    elif not timeline_has_speaker and not speaker_visual_candidates:
        msg = (
            f"[speaker-visual-validation] No speaker segments in timeline and no "
            f"speaker_visual_allowed candidates available for '{requested_speaker}'. "
            f"Narration-only output is expected."
        )
        log(msg)
        issues.append(msg)

    return issues


def build_refined_timeline(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Refined documentary rhythm:
    - title card
    - section card with voice-over
    - archive visual or speaker clip (unique, ranked)
    - fallback narration card if no unique clip is available
    - closing card

    Returns (timeline, quality_checks).
    """

    story = get_story(manifest)

    cards = manifest.get("cards", {})
    title_card = Path(cards.get("title_card", ""))
    closing_card = Path(cards.get("closing_card", cards.get("title_card", "")))

    section_cards = section_card_map(manifest)
    section_voiceovers = section_voiceover_map(manifest)
    source_lookup = source_video_lookup(manifest)

    tracker = ClipUsageTracker()
    timeline: list[dict[str, Any]] = []
    fallback_segments = 0
    repeated_fragments: list[str] = []

    timeline.append(
        {
            "type": "card",
            "segment_type": "title_card",
            "label": "title_card",
            "path": str(title_card),
            "duration": TITLE_CARD_MIN_DURATION,
            "audio_path": None,
            "subtitle_text": manifest.get("title", ""),
            "section_id": "TITLE",
        }
    )

    for section in story.get("sections", []):
        section_id = section.get("section_id")
        section_type = section.get("section_type", "section")
        narration = section.get("narration", "")

        section_card = section_cards.get(section_id)
        section_audio = section_voiceovers.get(section_id)
        narration_duration = 0.0
        if section_audio:
            narration_duration = max(get_media_duration(section_audio) + 0.4, MIN_CARD_DURATION)

        # Prefer speaker soundbite; fall back to archive visual.
        speaker, speaker_reason, speaker_fallback = choose_speaker_soundbite(section, tracker)

        # Safety gate: separate speaker_soundbite from speaker_visual rendering
        speaker_render_mode = None  # "speaker_clip" | "speaker_visual" | None (demote)
        if speaker is not None:
            relevance_role = speaker.get("relevance_role", "exact")
            speaker_soundbite_allowed = speaker.get("speaker_soundbite_allowed", speaker.get("speaker_clip_allowed", True))
            speaker_visual_allowed = speaker.get("speaker_visual_allowed", False)
            use_as = speaker.get("use_as", "speaker_soundbite")
            final_video_allowed = speaker.get("final_video_allowed", True)

            if not final_video_allowed or relevance_role in ("reject",):
                # Hard reject: not for final video at all
                log(f"  [safety-gate] Rejecting speaker clip {speaker.get('fragment_id')} (role={relevance_role}, final_video_allowed={final_video_allowed})")
                speaker = None
                speaker_reason = None
            elif speaker_soundbite_allowed and use_as == "speaker_soundbite":
                # Full speaker soundbite with original audio
                speaker_render_mode = "speaker_clip"
                log(f"  [safety-gate] Approved speaker_soundbite {speaker.get('fragment_id')} (role={relevance_role})")
            elif speaker_visual_allowed or use_as == "speaker_visual":
                # Speaker visual: show the clip muted under narration
                speaker_render_mode = "speaker_visual"
                log(f"  [safety-gate] Approved speaker_visual {speaker.get('fragment_id')} (role={relevance_role}, use_as={use_as}) — will render muted under narration")
            else:
                # Not good enough for soundbite and not a speaker_visual — demote to archive
                log(f"  [safety-gate] Demoting speaker clip {speaker.get('fragment_id')} (role={relevance_role}, use_as={use_as}, soundbite_allowed={speaker_soundbite_allowed}, visual_allowed={speaker_visual_allowed}) to archive_visual")
                speaker = None
                speaker_reason = None

        visual, visual_reason, visual_fallback = choose_archive_visual(section, tracker)

        chosen_clip = speaker or visual
        selection_reason = speaker_reason or visual_reason
        if speaker and speaker_render_mode == "speaker_clip":
            chosen_clip_type = "speaker_clip"
        elif speaker and speaker_render_mode == "speaker_visual":
            chosen_clip_type = "speaker_visual"
        else:
            chosen_clip_type = "archive_visual"

        if chosen_clip:
            fragment_id = chosen_clip.get("fragment_id") or chosen_clip.get("id")
            source_video = chosen_clip.get("source_video") or chosen_clip.get("source_video_file") or chosen_clip.get("video_id")
            if not source_video and fragment_id:
                source_video = source_lookup.get(fragment_id)

            chosen_clip["type"] = "archive_clip"
            chosen_clip["section_id"] = section_id
            chosen_clip["source_video"] = source_video
            chosen_clip["selection_reason"] = selection_reason
            chosen_clip["fallback_reason"] = None

            # Phase 4O: respect moment-level muted/subtitle decisions from Step 07/08
            # These fields are written by select_best_moment_for_section and
            # enrich_timeline_with_moment_fields.
            p4o_muted = chosen_clip.get("muted")
            p4o_subtitle = chosen_clip.get("subtitle_text")
            p4o_moment_text = chosen_clip.get("moment_text", "")
            p4o_seg_type = chosen_clip.get("segment_type") or chosen_clip_type

            # Resolve segment_type from Phase 4O if present, fall back to speaker-gate decision
            if p4o_seg_type in ("speaker_clip", "archive_visual", "narration_broll"):
                chosen_clip["segment_type"] = p4o_seg_type
            else:
                chosen_clip["segment_type"] = chosen_clip_type

            final_seg_type = chosen_clip["segment_type"]

            # Muted: Phase 4O wins; fall back to legacy rules
            if p4o_muted is not None:
                chosen_clip["muted"] = bool(p4o_muted)
            else:
                # archive_visual → muted; speaker_clip → source audio
                chosen_clip["muted"] = (final_seg_type == "archive_visual")

            # Subtitle text:
            # - speaker_clip: use moment_text (not full fragment text) if available
            # - archive_visual / narration_broll: no speaker subtitle
            if final_seg_type == "speaker_clip":
                if p4o_subtitle is not None:
                    chosen_clip["subtitle_text"] = p4o_subtitle
                elif p4o_moment_text:
                    chosen_clip["subtitle_text"] = p4o_moment_text
                else:
                    chosen_clip["subtitle_text"] = chosen_clip.get("quote_summary", "")
            else:
                chosen_clip["subtitle_text"] = ""

            chosen_clip["lower_third"] = build_lower_third(chosen_clip)

        # Prefer narration over unique b-roll when a section voice-over exists.
        narration_segment: dict[str, Any] | None = None
        if section_audio:
            excluded_ids = set()
            if chosen_clip and (chosen_clip.get("fragment_id") or chosen_clip.get("id")):
                excluded_ids.add(str(chosen_clip.get("fragment_id") or chosen_clip.get("id")))
            broll, _, _ = choose_broll_for_narration(section, tracker, narration_duration, excluded_fragment_ids=excluded_ids)
            if broll:
                fragment_id = broll.get("fragment_id") or broll.get("id")
                source_video = broll.get("source_video") or broll.get("source_video_file") or broll.get("video_id")
                if not source_video and fragment_id:
                    source_video = source_lookup.get(fragment_id)

                narration_segment = dict(broll)
                narration_segment.update(
                    {
                        "type": "archive_clip",
                        "segment_type": "narration_broll",
                        "label": f"{section_id}_{section_type}_narration_broll",
                        "section_id": section_id,
                        "source_video": source_video,
                        "audio_path": str(section_audio),
                        "duration": narration_duration,
                        "subtitle_text": narration,
                        "voiceover_text": narration,
                        "broll_fragment_id": fragment_id,
                        "broll_source_video": source_video,
                        "selection_reason": "unique_narration_broll",
                        "fallback_reason": None,
                    }
                )
                tracker.mark_used(narration_segment)
            elif section_card:
                segment_type = "section_card" if chosen_clip else "fallback_narration"
                narration_segment = {
                    "type": "card",
                    "segment_type": segment_type,
                    "label": f"{section_id}_{section_type}_{segment_type}",
                    "section_id": section_id,
                    "path": str(section_card),
                    "duration": narration_duration,
                    "audio_path": str(section_audio),
                    "subtitle_text": narration,
                    "voiceover_text": narration,
                    "fallback_reason": None if chosen_clip else (speaker_fallback or visual_fallback or "no_unique_speaker_or_visual_clip_available"),
                }
                if not chosen_clip:
                    fallback_segments += 1
                    log(f"[Phase4A] Section {section_id}: using fallback card — {narration_segment['fallback_reason']}")
        elif section_card:
            segment_type = "section_card" if chosen_clip else "fallback_narration"
            narration_segment = {
                "type": "card",
                "segment_type": segment_type,
                "label": f"{section_id}_{section_type}_{segment_type}",
                "section_id": section_id,
                "path": str(section_card),
                "duration": STATIC_CARD_DURATION,
                "audio_path": None,
                "subtitle_text": narration,
                "fallback_reason": None if chosen_clip else (speaker_fallback or visual_fallback or "no_unique_speaker_or_visual_clip_available"),
            }
            if not chosen_clip:
                fallback_segments += 1
                log(f"[Phase4A] Section {section_id}: using fallback card — {narration_segment['fallback_reason']}")

        if narration_segment:
            timeline.append(narration_segment)

        if chosen_clip:
            ok, reason = tracker.can_use(chosen_clip)
            if not ok:
                log(f"[Phase4D] Section {section_id}: dropping chosen clip after rhythm planning — {reason}")
                chosen_clip = None

        if chosen_clip:
            tracker.mark_used(chosen_clip)
            timeline.append(chosen_clip)

    timeline.append(
        {
            "type": "card",
            "segment_type": "closing_card",
            "label": "closing_card",
            "path": str(closing_card),
            "duration": CLOSING_CARD_MIN_DURATION,
            "audio_path": None,
            "subtitle_text": "AI can become a bridge to knowledge when inclusion, trust and responsibility guide its design.",
            "section_id": "CLOSING",
        }
    )

    speaker_visual_issues = ensure_requested_speaker(manifest, timeline)

    quality_checks = {
        "unique_fragments": len(tracker.used_fragment_ids),
        "repeated_fragments": repeated_fragments,
        "source_video_use_counts": dict(tracker.used_source_videos),
        "speaker_use_counts": dict(tracker.used_speakers),
        "fallback_segments": fallback_segments,
        "speaker_visual_validation": speaker_visual_issues,
    }

    return timeline, quality_checks
