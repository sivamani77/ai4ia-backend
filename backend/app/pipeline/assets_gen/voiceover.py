from __future__ import annotations
from pathlib import Path
from typing import Any

import edge_tts

from app.pipeline.media_probe import media_duration_or_zero as get_media_duration
from app.pipeline.assets_gen.text_copy import make_display_title


def build_voiceover_script(storyline: dict[str, Any]) -> list[dict[str, str]]:
    sections = storyline.get("sections", [])
    voiceover_script = []

    for section in sections:
        section_id = section.get("section_id", "")
        section_type = section.get("section_type", "")
        # Phase 4O.3: use narration first, fall back to viewer_narration
        narration = (section.get("narration") or section.get("viewer_narration") or "").strip()

        if narration:
            voiceover_script.append(
                {
                    "section_id": section_id,
                    "section_type": section_type,
                    "text": narration,
                    "tone": "calm cinematic documentary narrator",
                }
            )

    # Phase 4O.3: if no sections produced narration, build a fallback from top-level fields
    if not voiceover_script:
        logline = (storyline.get("logline") or "").strip()
        closing = (storyline.get("closing_meaning") or "").strip()
        thesis = (storyline.get("narrative_thesis") or "").strip()
        central_q = (storyline.get("central_question") or "").strip()

        fallback_parts = [p for p in [logline, thesis or central_q, closing] if p]
        if fallback_parts:
            voiceover_script.append(
                {
                    "section_id": "S_FALLBACK",
                    "section_type": "narration_fallback",
                    "text": "  ".join(fallback_parts),
                    "tone": "calm cinematic documentary narrator",
                }
            )

    return voiceover_script


def voiceover_text_for_tts(voiceover_script: list[dict[str, str]]) -> str:
    parts = []

    for item in voiceover_script:
        text = item.get("text", "").strip()
        if text:
            parts.append(text)

    return "\n\n".join(parts).strip()


def reconcile_narration_duration(item: dict[str, Any], audio_path: str | None) -> None:
    """Set render_duration on a narration item from its real TTS audio length.

    A narration_broll segment is a voiceover over a STILL visual — it has no
    footage, so its length is the voiceover length (plus a short tail), full stop.
    It must NOT be stretched to any source-video start/end window: narration items
    sometimes carry a stray start/end inherited from a speaker slot, and honouring
    that window pads the segment with dead silence (e.g. a 5.9s voiceover forced to
    a 58s window = 52s of silence over a frozen image). The window only wins when
    there is no usable audio at all, as a last-resort fallback duration.
    """
    audio_dur = 0.0
    if audio_path and Path(audio_path).exists():
        audio_dur = get_media_duration(Path(audio_path))

    if audio_dur > 0:
        render_duration = audio_dur + 0.4
    else:
        # No usable voiceover — fall back to the fragment window so the segment
        # still has a non-zero duration rather than collapsing to nothing.
        frag_dur = float(item.get("duration") or 0.0)
        if frag_dur <= 0.0:
            start_v, end_v = item.get("start"), item.get("end")
            try:
                frag_dur = max(0.0, float(end_v) - float(start_v))
            except (TypeError, ValueError):
                frag_dur = 0.0
        render_duration = frag_dur

    if render_duration > 0:
        item["render_duration"] = round(render_duration, 3)
        item["duration"] = item["render_duration"]
        item["tts_audio_seconds"] = round(audio_dur, 3)


async def generate_tts(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    volume: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
    )

    await communicate.save(str(output_path))


def write_voiceover_script_txt(
    output_path: Path,
    title: str,
    voiceover_script: list[dict[str, str]],
) -> None:
    lines = []
    lines.append(make_display_title(title))
    lines.append("=" * len(make_display_title(title)))
    lines.append("")

    for item in voiceover_script:
        section_id = item.get("section_id", "")
        section_type = item.get("section_type", "").replace("_", " ").title()
        text = item.get("text", "")

        lines.append(f"{section_id} - {section_type}")
        lines.append("-" * 60)
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
