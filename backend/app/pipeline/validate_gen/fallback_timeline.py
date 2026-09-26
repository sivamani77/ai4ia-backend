"""Deterministic, score-based final-timeline construction used when the LLM is unavailable."""
from typing import Any


def build_fallback_final_timeline(
    validated_candidates: list[dict[str, Any]],
    narrative_sections: list[dict[str, Any]],
    storyline_data: dict[str, Any],
    target_duration: int = 120,
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Build a minimal final_timeline from scored candidates when LLM fails.
    Clips go only in middle sections; `max_clips` caps them (None: no cap)."""
    approved = [c for c in validated_candidates if c.get("approved")]
    speaker_clips = [c for c in approved if c.get("decision") == "speaker_clip"][:max_clips]
    clip_sections = {s["section_id"] for s in narrative_sections[1:-1]}
    brolls = [c for c in approved if c.get("decision") in ("narration_broll", "visual_support")]

    title = storyline_data.get("storyline", {}).get("title") or "AI4IA Documentary"
    section_ids = [s["section_id"] for s in narrative_sections]

    # Extract narration from script-07-generated sections (already topic-correct and proportionally long)
    sl_sections = (storyline_data.get("storyline") or {}).get("sections") or []
    sl_narration: list[str] = []
    for sec in sl_sections:
        narration = sec.get("narration") or sec.get("voiceover_text") or ""
        if narration:
            sl_narration.append(narration)

    final_timeline: list[dict[str, Any]] = [
        {"type": "title_card", "title": title, "duration": 5.0}
    ]

    vo_idx = 0
    sc_idx = 0
    sections_used: set[str] = set()

    for section_idx, s in enumerate(narrative_sections):
        sid = s["section_id"]
        # EVERY section gets an AI narration bridge — this is the documentary's
        # connective voice. It renders over an archive fragment when one is
        # available, otherwise over a generated visual (Step 08 makes one for
        # footage-less narration). This guarantees narration is never missing,
        # even when the approved pool happens to be all speaker clips.
        broll_cand = brolls[vo_idx] if vo_idx < len(brolls) else None
        narration_text = sl_narration[section_idx] if section_idx < len(sl_narration) else ""
        narration_item: dict[str, Any] = {
            "type": "narration_broll",
            "segment_type": "narration_broll",
            "audio_mode": "ai_voiceover",
            "section_id": sid,
            "voiceover_text": narration_text,
            "subtitle_text": narration_text,
            "subtitle_kind": "voiceover",
            "visual_direction": s.get("card_text", ""),
        }
        if broll_cand:
            fragment_id = broll_cand.get("fragment_id")
            source_video = broll_cand.get("source_video") or (
                "_".join(str(fragment_id).rsplit("_", 1)[:-1]) if fragment_id else None
            )
            narration_item["fragment_id"] = fragment_id
            narration_item["source_video"] = source_video
            narration_item["start"] = float(broll_cand.get("quote_start") or broll_cand.get("start") or 0)
            narration_item["end"] = float(broll_cand.get("quote_end") or broll_cand.get("end") or 10)
            vo_idx += 1
        else:
            # No archive fragment for this section's bridge — footage-less
            # narration over a generated visual (Step 08 renders the visual).
            narration_item["fragment_id"] = None
            narration_item["source_video"] = None
        final_timeline.append(narration_item)
        sections_used.add(sid)

        # Assign a speaker_clip to this section if available
        sc_cand = speaker_clips[sc_idx] if sc_idx < len(speaker_clips) and sid in clip_sections else None
        if sc_cand:
            fragment_id = sc_cand.get("fragment_id")
            source_video = sc_cand.get("source_video") or (
                "_".join(str(fragment_id).rsplit("_", 1)[:-1]) if fragment_id else None
            )
            qs = float(sc_cand.get("quote_start") or sc_cand.get("start") or 0)
            qe = float(sc_cand.get("quote_end") or sc_cand.get("end") or qs + 10)
            final_timeline.append({
                "type": "speaker_clip",
                "segment_type": "speaker_clip",
                "audio_mode": "source_audio",
                "section_id": sid,
                "fragment_id": fragment_id,
                "source_video": source_video,
                "speaker": sc_cand.get("speaker_label") or sc_cand.get("speaker") or "",
                "render_start": qs,
                "render_end": qe,
                "render_duration": round(qe - qs, 3),
                "quote_text": sc_cand.get("quote_text") or sc_cand.get("text") or "",
                "subtitle_text": sc_cand.get("subtitle_text") or sc_cand.get("quote_text") or "",
                "quote_complete": sc_cand.get("quote_complete", False),
                "muted": False,
            })
            sc_idx += 1
            sections_used.add(sid)

    final_timeline.append({"type": "closing_card", "title": title, "duration": 5.0})

    rejected = [
        {"fragment_id": c.get("fragment_id"), "reason": c.get("reason", "Not used in fallback timeline")}
        for c in validated_candidates
        if not c.get("approved") or c.get("decision") == "reject"
    ]

    return {
        "title": title,
        "logline": f"A documentary on {title}.",
        "narrative_arc": [],
        "final_timeline": final_timeline,
        "rejected_candidates": rejected,
        "editorial_quality_score": 0.5,
        "warnings": ["LLM narrative assembly unavailable — used score-based fallback."],
        "_fallback": True,
    }
