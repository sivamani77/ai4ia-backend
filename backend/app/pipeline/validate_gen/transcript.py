"""Transcript lookup and context-window extraction for Step 07b fragment validation."""
import json
from pathlib import Path
from typing import Any

from app.pipeline.validate_gen.llm_client import log

TRANSCRIPT_CONTEXT_SECONDS = 45.0
MIN_CONTEXT_SEGMENTS_EACH_SIDE = 2


def find_transcript_file(
    source_video: str,
    transcripts_dir: Path,
    fallback_dir: Path | None = None,
) -> Path | None:
    """Find the transcript JSON for a source_video by name."""
    name = str(source_video).strip()
    for d in [transcripts_dir, fallback_dir]:
        if not d or not d.exists():
            continue
        candidate = d / (name + ".json")
        if candidate.exists():
            return candidate
        # Fuzzy: find any file whose stem starts with the video_id prefix (first 16 chars)
        prefix = name[:16].lower()
        for f in d.glob("*.json"):
            if f.stem.lower().startswith(prefix):
                return f
    return None


def load_transcript_segments(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = data.get("segments") or []
        return [s for s in segs if isinstance(s, dict) and "start" in s and "end" in s]
    except Exception as exc:
        log(f"  [transcript] Failed to load {path}: {exc}")
        return []


def extract_context_window(
    segments: list[dict[str, Any]],
    frag_start: float,
    frag_end: float,
    context_seconds: float = TRANSCRIPT_CONTEXT_SECONDS,
    min_each_side: int = MIN_CONTEXT_SEGMENTS_EACH_SIDE,
) -> list[dict[str, Any]]:
    """Return segments from context_seconds before frag_start to context_seconds after frag_end."""
    if not segments:
        return []
    win_start = frag_start - context_seconds
    win_end = frag_end + context_seconds

    window = [s for s in segments if s.get("end", 0) >= win_start and s.get("start", 0) <= win_end]

    # Ensure minimum segments on each side
    frag_segs = [i for i, s in enumerate(segments) if s.get("start", 0) >= frag_start and s.get("end", 0) <= frag_end]
    if frag_segs:
        first_idx = frag_segs[0]
        last_idx = frag_segs[-1]
        extra_before = segments[max(0, first_idx - min_each_side):first_idx]
        extra_after = segments[last_idx + 1:last_idx + 1 + min_each_side]
        all_ids = {id(s) for s in window}
        for s in extra_before + extra_after:
            if id(s) not in all_ids:
                window.append(s)
        window.sort(key=lambda s: s.get("start", 0))

    return window


def context_window_to_prompt_text(window: list[dict[str, Any]], frag_start: float, frag_end: float) -> str:
    lines = []
    for s in window:
        ts = s.get("start", 0)
        te = s.get("end", 0)
        text = s.get("text", "").strip()
        marker = " <<<FRAGMENT>>>" if ts >= frag_start and te <= frag_end + 0.5 else ""
        lines.append(f"  [{ts:.1f}s–{te:.1f}s]{marker} {text}")
    return "\n".join(lines)
