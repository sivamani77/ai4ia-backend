from __future__ import annotations
from pathlib import Path
from typing import Any

from app.pipeline.timeline_budget import segment_effective_duration
from app.pipeline.assets_gen.export_timeline import _detect_intro_admin_in_moment
from app.pipeline.assets_gen.text_copy import (
    detect_generic_card_title,
    detect_generic_voiceover,
    detect_internal_pipeline_language_08,
)

STRICT_NO_GENERIC_FALLBACK = True


def validate_asset_manifest_before_save(
    manifest: dict[str, Any],
    prompt_profile: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
) -> list[str]:
    """Strict pre-save validation for asset manifest."""
    if not STRICT_NO_GENERIC_FALLBACK:
        return []

    issues = []
    cards = manifest.get("cards") or {}
    for sc in (cards.get("section_cards") or []):
        headline = sc.get("headline", "")
        subtitle = sc.get("subtitle", "")
        sid = sc.get("section_id", "?")
        if detect_generic_card_title(headline):
            issues.append(f"generic_card_title_detected: '{headline}' in section {sid}")
        pipeline_issues = detect_internal_pipeline_language_08(headline)
        if pipeline_issues:
            issues.append(f"card_not_documentary_style: internal language in section {sid} headline")
        pipeline_issues_sub = detect_internal_pipeline_language_08(subtitle)
        if pipeline_issues_sub:
            issues.append(f"card_not_documentary_style: internal language in section {sid} subtitle")

    voice = manifest.get("voice") or {}
    for sv in (voice.get("section_voiceovers") or []):
        text = sv.get("text", "")
        sid = sv.get("section_id", "?")
        if detect_generic_voiceover(text):
            issues.append(f"generic_voiceover_detected in section {sid}")
        pipeline_issues_vo = detect_internal_pipeline_language_08(text)
        if pipeline_issues_vo:
            issues.append(f"internal_pipeline_language_detected in voiceover section {sid}")

    storyline = manifest.get("storyline") or {}
    export_timeline = storyline.get("export_timeline") or []
    timeline_fragment_ids = {item.get("fragment_id") for item in export_timeline if item.get("fragment_id")}
    if selected_candidates and not timeline_fragment_ids:
        issues.append("missing_evidence_links: export_timeline has no fragment_ids despite having selected_candidates")

    closing_headline = manifest.get("closing_headline") or ""
    if closing_headline:
        pipeline_close = detect_internal_pipeline_language_08(closing_headline)
        if pipeline_close:
            issues.append(f"closing_not_prompt_specific: internal language in closing headline")

    # Phase 4O: moment-level validation on export_timeline
    storyline = manifest.get("storyline") or {}
    manifest_timeline = (storyline.get("export_timeline") or [])
    for titem in manifest_timeline:
        seg_type = titem.get("type", "")
        fid = titem.get("fragment_id")
        if not fid or seg_type in ("title_card", "closing_card"):
            continue

        moment_text = titem.get("moment_text", "")
        if seg_type == "speaker_clip" and not moment_text:
            issues.append(f"selected_clip_missing_moment_text: {fid}")

        if seg_type == "speaker_clip" and moment_text and _detect_intro_admin_in_moment(moment_text):
            issues.append(f"speaker_clip_has_intro_language: {fid}")

        speech_q = titem.get("speech_quality") or {}
        if seg_type == "speaker_clip" and speech_q and not speech_q.get("is_usable_speech", True):
            issues.append(f"speaker_clip_has_low_speech_ratio: {fid}")

        if seg_type == "archive_visual" and titem.get("muted") is False:
            issues.append(f"archive_visual_not_muted: {fid}")

    return issues


def preflight_audit_manifest(manifest: dict[str, Any], skip_tts: bool = False) -> list[str]:
    """Verify every timeline item is renderable BEFORE handing off to Step 09.

    Returns a list of failure strings; empty means the manifest is render-safe.
    Checks: card PNGs exist, every content item has a positive duration, voiceover
    audio files exist on disk, and narration items without archive footage carry a
    generated visual PNG that exists.
    """
    failures: list[str] = []
    cards = manifest.get("cards") or {}

    for card_key in ("title_card", "closing_card"):
        card_path = cards.get(card_key)
        if not card_path or not Path(card_path).exists():
            failures.append(f"{card_key}: PNG missing at {card_path!r}")

    timeline = manifest.get("timeline") or manifest.get("export_timeline") or []
    for idx, item in enumerate(timeline, start=1):
        t = item.get("segment_type") or item.get("type") or ""
        label = f"item {idx} ({t}, fragment={str(item.get('fragment_id') or '')[:40]!r})"

        if t in ("title_card", "closing_card", "year_stamp"):
            continue

        if segment_effective_duration(item) <= 0.0:
            failures.append(f"{label}: no measurable duration")

        audio_mode = item.get("audio_mode") or ""
        audio_path = item.get("audio_path") or item.get("voiceover_path")
        if audio_mode in ("ai_voiceover", "external") and not skip_tts:
            if not audio_path:
                failures.append(f"{label}: audio_mode={audio_mode} but no audio_path")
            elif not Path(audio_path).exists():
                failures.append(f"{label}: voiceover audio missing at {audio_path!r}")

        if t == "narration_broll" and not item.get("source_video"):
            vis = item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path")
            if not vis:
                failures.append(f"{label}: no archive footage and no generated visual")
            elif not Path(vis).exists():
                failures.append(f"{label}: generated visual missing at {vis!r}")

        if t == "speaker_clip":
            if not item.get("source_video") and not item.get("fragment_id"):
                failures.append(f"{label}: speaker_clip without source_video/fragment_id")

    return failures
