from __future__ import annotations
import re
from math import ceil
from pathlib import Path
from typing import Any

from app.pipeline.export_gen.ffmpeg_utils import log
from app.pipeline.export_gen.transcription import DEFAULT_SUBTITLE_CHUNK_WORDS, _real_timed_entries_from_words, _transcribe_words

MIN_SUBTITLE_ENTRY_DURATION = 1.2


def format_srt_timestamp(seconds: float) -> str:
    total_millis = max(int(round(seconds * 1000)), 0)
    total_seconds, millis = divmod(total_millis, 1000)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    total_millis = max(int(round(seconds * 1000)), 0)
    total_seconds, millis = divmod(total_millis, 1000)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def normalize_subtitle_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text_for_subtitles(text: str, max_words: int = DEFAULT_SUBTITLE_CHUNK_WORDS) -> list[str]:
    cleaned = normalize_subtitle_text(text)
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue

        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start:start + max_words]))

    return chunks


def make_subtitle_entries_for_segment(
    text: str,
    segment_start: float,
    segment_end: float,
    max_words: int = DEFAULT_SUBTITLE_CHUNK_WORDS,
) -> list[dict[str, Any]]:
    chunks = split_text_for_subtitles(text, max_words=max_words)
    if not chunks:
        return []

    duration = max(segment_end - segment_start, 0.0)
    if duration <= 0:
        return []

    entry_count = len(chunks)
    max_entries_by_duration = max(1, int(duration / MIN_SUBTITLE_ENTRY_DURATION))

    if entry_count > max_entries_by_duration:
        regroup_size = ceil(entry_count / max_entries_by_duration)
        regrouped_chunks = []
        for idx in range(0, len(chunks), regroup_size):
            regrouped_chunks.append(" ".join(chunks[idx:idx + regroup_size]))
        chunks = regrouped_chunks
        entry_count = len(chunks)

    chunk_duration = duration / entry_count
    entries = []

    for idx, chunk in enumerate(chunks):
        start = segment_start + (idx * chunk_duration)
        if idx == len(chunks) - 1:
            end = segment_end
        else:
            end = segment_start + ((idx + 1) * chunk_duration)

        if end <= start:
            end = start + 0.1

        entries.append(
            {
                "start": start,
                "end": min(end, segment_end) if segment_end > start else end,
                "text": chunk,
            }
        )

    if entries:
        entries[-1]["end"] = max(entries[-1]["end"], segment_end)

    return entries


def classify_segment_subtitle_type(segment: dict[str, Any]) -> str:
    role = str(segment.get("role") or "").lower()
    segment_type = str(segment.get("segment_type") or "").lower()
    item_type = str(segment.get("type") or "").lower()

    if segment_type == "narration_broll" or "narration" in segment_type or item_type == "card" or "voiceover" in segment_type:
        return "voiceover"
    if "speaker" in role or "speaker" in segment_type or segment_type == "speaker_visual":
        return "speaker"
    if item_type == "archive_clip":
        return "speaker"
    return "other"


def segment_has_spoken_audio(segment: dict[str, Any]) -> bool:
    item_type = segment.get("type")
    if item_type == "archive_clip":
        seg_type = str(segment.get("segment_type") or "").lower()
        audio_mode = str(segment.get("audio_mode") or "").lower()
        if seg_type == "archive_visual":
            return False
        # narration_broll with ai_voiceover has external spoken audio even though muted=True
        if seg_type == "narration_broll" and audio_mode == "ai_voiceover":
            return bool(segment.get("audio_path"))
        if segment.get("muted") is True:
            return False
        return True
    if item_type == "card":
        return bool(segment.get("audio_path"))
    return False


def get_segment_subtitle_text(segment: dict[str, Any]) -> str:
    subtitle_type = classify_segment_subtitle_type(segment)
    seg_type = str(segment.get("segment_type") or "").lower()

    # Phase 4O: archive_visual clips must not produce speaker subtitles
    if seg_type == "archive_visual":
        return ""
    # muted=True suppresses subtitles only for truly silent items, not narration_broll with ai_voiceover
    if segment.get("muted") is True and not (
        seg_type == "narration_broll"
        and segment.get("audio_mode") in ("ai_voiceover", "external")
    ):
        return ""

    if subtitle_type == "speaker":
        # Phase 4O: prefer moment_text (the selected window text) over full fragment text
        keys = [
            "moment_text",
            "subtitle_text",
            "quote_text",
            "quote_summary",
        ]
    else:
        keys = [
            "subtitle_text",
            "voiceover_text",
            "narration",
            "card_text",
            "section_narration",
            "text",
            "transcript_text",
            "quote_text",
            "quote_summary",
            "summary",
        ]

    for key in keys:
        value = segment.get(key)
        if isinstance(value, str):
            cleaned = normalize_subtitle_text(value)
            if cleaned:
                return cleaned

    return ""


def build_subtitle_artifacts(segment_infos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    coverage = {
        "segments_with_subtitles": 0,
        "segments_without_subtitles": 0,
        "speaker_segments_subtitled": 0,
        "voiceover_segments_subtitled": 0,
    }
    warnings: list[dict[str, Any]] = []

    for info in segment_infos:
        timeline_item = info.get("timeline_item", {})
        if not segment_has_spoken_audio(timeline_item):
            continue

        subtitle_type = classify_segment_subtitle_type(timeline_item)
        subtitle_text = get_segment_subtitle_text(timeline_item)
        info["subtitle_text"] = subtitle_text
        info["subtitle_kind"] = subtitle_type

        if not subtitle_text:
            coverage["segments_without_subtitles"] += 1
            warnings.append(
                {
                    "index": info.get("index"),
                    "label": timeline_item.get("label"),
                    "fragment_id": timeline_item.get("fragment_id"),
                    "reason": "no_subtitle_text_for_spoken_segment",
                }
            )
            continue

        caption_start = float(info["start_time"])
        caption_end = float(info["end_time"])
        segment_entries: list[dict[str, Any]] = []

        # A speaker_clip's render window (start_time/end_time) can span far
        # more of the archive fragment than what subtitle_text (our curated
        # pull-quote) covers — captioning only the matched quote and going
        # blank for the rest of the clip reads as broken playback. Instead,
        # transcribe the ACTUAL rendered segment file with Whisper and caption
        # the WHOLE clip from that real transcript — continuous, accurate,
        # word-level-synced captions for as long as the clip actually plays,
        # not just for the curated excerpt. Falls back to the even-pace
        # estimate (clamped to the quote's own plausible span) only if
        # transcription itself fails.
        if subtitle_type == "speaker":
            segment_path = info.get("path")
            try:
                if not segment_path:
                    raise ValueError("no rendered segment path to transcribe")
                words = _transcribe_words(Path(segment_path))
                if words:
                    real_entries = _real_timed_entries_from_words(words)
                    segment_entries = [
                        {
                            "start": caption_start + e["start"],
                            "end": caption_start + e["end"],
                            "text": e["text"],
                        }
                        for e in real_entries
                    ]
                    log(
                        f"  [subtitles] speaker_clip {str(timeline_item.get('fragment_id'))[:40]!r}: "
                        f"real word-level captions for full clip ({len(words)} words transcribed)"
                    )
            except Exception as exc:
                log(f"  [subtitles] word-level transcription failed for speaker_clip, falling back to estimate: {exc}")

            if not segment_entries:
                quote_words = len(subtitle_text.split())
                if quote_words:
                    estimated_speech_dur = max(3.0, quote_words / 2.5)
                    caption_end = min(caption_end, caption_start + estimated_speech_dur)

        if not segment_entries:
            segment_entries = make_subtitle_entries_for_segment(
                text=subtitle_text,
                segment_start=caption_start,
                segment_end=caption_end,
            )

        if not segment_entries:
            coverage["segments_without_subtitles"] += 1
            warnings.append(
                {
                    "index": info.get("index"),
                    "label": timeline_item.get("label"),
                    "fragment_id": timeline_item.get("fragment_id"),
                    "reason": "subtitle_chunking_produced_no_entries",
                }
            )
            continue

        coverage["segments_with_subtitles"] += 1
        if subtitle_type == "speaker":
            coverage["speaker_segments_subtitled"] += 1
        elif subtitle_type == "voiceover":
            coverage["voiceover_segments_subtitled"] += 1

        for entry in segment_entries:
            entries.append(
                {
                    "segment_index": info.get("index"),
                    "segment_label": timeline_item.get("label"),
                    "segment_type": subtitle_type,
                    **entry,
                }
            )

    return entries, {"coverage": coverage, "warnings": warnings}


def write_srt(entries: list[dict[str, Any]], path: Path) -> None:
    lines = []
    for index, entry in enumerate(entries, start=1):
        lines.append(str(index))
        lines.append(f"{format_srt_timestamp(float(entry['start']))} --> {format_srt_timestamp(float(entry['end']))}")
        lines.append(entry["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(entries: list[dict[str, Any]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for entry in entries:
        lines.append(f"{format_vtt_timestamp(float(entry['start']))} --> {format_vtt_timestamp(float(entry['end']))}")
        lines.append(entry["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
