import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from math import ceil
from pathlib import Path
from datetime import datetime
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VIDEO_DIR = Path(os.environ.get("AI4IA_RAW_VIDEO_DIR", str(PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel")))
GENERATED_ASSETS_DIR = Path(os.environ.get("AI4IA_LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "data" / "generated_assets")))
EXPORTS_DIR = Path(os.environ.get("AI4IA_EXPORTS_DIR", str(PROJECT_ROOT / "data" / "exports")))
TEMP_DIR = PROJECT_ROOT / "temp" / "video_export"

LATEST_MANIFEST_PATH = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"
FINAL_SUBTITLE_SCRIPT = PROJECT_ROOT / "scripts" / "11_generate_final_subtitles.py"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
Path("/tmp/ai4ia_pipeline/video_export").mkdir(parents=True, exist_ok=True)
TEMP_DIR = Path("/tmp/ai4ia_pipeline/video_export")

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 30

TITLE_CARD_MIN_DURATION = 3.5
STATIC_CARD_DURATION = 2.5
MIN_CARD_DURATION = 2.0
CLOSING_CARD_MIN_DURATION = 3.5

DEFAULT_CLIP_DURATION = 8.0
MAX_SPEAKER_CLIP_DURATION = 13.0
MAX_ARCHIVE_VISUAL_DURATION = 9.0
LOWER_THIRD_DURATION = 3.8

FADE_DURATION = 0.45

ARCHIVE_AUDIO_VOLUME = 1.0
SECTION_VOICEOVER_VOLUME = 1.0
DEFAULT_SUBTITLE_CHUNK_WORDS = 10
MIN_SUBTITLE_ENTRY_DURATION = 1.2


def log(message: str) -> None:
    print(message, flush=True)


def run_cmd(cmd: list[str]) -> None:
    log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        if result.stdout.strip():
            log(result.stdout.rstrip())
        if result.stderr.strip():
            log(result.stderr.rstrip())
        detail = (result.stderr.strip() or result.stdout.strip() or "no output")[-3000:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")


def run_cmd_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def run_cmd_logged(cmd: list[str]) -> None:
    log("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.stdout.strip():
        log(result.stdout.rstrip())
    if result.stderr.strip():
        log(result.stderr.rstrip())

    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip() or "no output")[-2000:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "video"


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/08_generate_assets.py first.")

    return json.loads(path.read_text(encoding="utf-8"))


def get_media_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    output = run_cmd_capture(cmd)

    try:
        return float(output)
    except ValueError:
        raise RuntimeError(f"ffprobe returned non-numeric duration for {path}: {output!r}")


def find_source_video(source_video: str | None, fragment_id: str | None = None) -> Path | None:
    if source_video:
        exact = RAW_VIDEO_DIR / f"{source_video}.mp4"
        if exact.exists():
            return exact

        matches = sorted(RAW_VIDEO_DIR.glob(f"{source_video}*.mp4"))
        if matches:
            return matches[0]

    if fragment_id:
        inferred = re.sub(r"_\d{4}$", "", fragment_id)
        exact = RAW_VIDEO_DIR / f"{inferred}.mp4"
        if exact.exists():
            return exact

        matches = sorted(RAW_VIDEO_DIR.glob(f"{inferred}*.mp4"))
        if matches:
            return matches[0]

    return None


def get_story(manifest: dict[str, Any]) -> dict[str, Any]:
    embedded_storyline = manifest.get("storyline")
    if isinstance(embedded_storyline, dict):
        return embedded_storyline

    storyline_path = Path(manifest["storyline_path"])
    payload = json.loads(storyline_path.read_text(encoding="utf-8"))
    return payload["storyline"]


def section_voiceover_map(manifest: dict[str, Any]) -> dict[str, Path]:
    mapping = {}

    voice = manifest.get("voice", {})
    for item in voice.get("section_voiceovers", []):
        section_id = item.get("section_id")
        audio_path = item.get("audio_path")

        if section_id and audio_path:
            path = Path(audio_path)
            if path.exists():
                mapping[section_id] = path

    return mapping


def section_voiceover_by_type_and_index(manifest: dict[str, Any]) -> dict[str, Path]:
    """
    Extended voiceover lookup: keys are section_id, section_type, section_role, AND
    positional index (str(0), str(1), ...) so that canonical timeline items with only
    section_role can still be matched.
    """
    mapping: dict[str, Path] = {}
    voice = manifest.get("voice", {})
    svlist = voice.get("section_voiceovers", [])

    for pos_idx, item in enumerate(svlist):
        audio_path = item.get("audio_path")
        if not audio_path:
            continue
        path = Path(audio_path)
        if not path.exists():
            continue
        section_id = item.get("section_id", "")
        section_type = item.get("section_type", "")
        if section_id:
            mapping[section_id] = path
        if section_type:
            mapping[section_type] = path
        # positional index (0-based): canonical timeline fragment items are also 0-based
        mapping[str(pos_idx)] = path

    return mapping


def section_card_map(manifest: dict[str, Any]) -> dict[str, Path]:
    cards = manifest.get("cards", {})
    mapping = {}

    for item in cards.get("section_cards", []):
        section_id = item.get("section_id")
        path = item.get("path")

        if section_id and path:
            card_path = Path(path)
            if card_path.exists():
                mapping[section_id] = card_path

    return mapping


def source_video_lookup(manifest: dict[str, Any]) -> dict[str, str]:
    lookup = {}

    for item in manifest.get("archive_fragments", []):
        fragment_id = item.get("fragment_id")
        source_video = item.get("source_video")

        if fragment_id and source_video:
            lookup[fragment_id] = source_video

    return lookup


def get_clip_candidate_duration(clip: dict[str, Any]) -> float:
    duration = clip.get("duration")
    if duration not in (None, ""):
        try:
            return float(duration)
        except (TypeError, ValueError):
            pass  # fall through to start/end calculation

    start_raw = clip.get("start")
    end_raw = clip.get("end")
    try:
        start = float(start_raw or 0)
        end = float(end_raw or 0)
    except (TypeError, ValueError):
        raise ValueError(
            f"Clip has non-numeric start/end (start={start_raw!r}, end={end_raw!r})"
            f" and no usable duration field — fragment_id={clip.get('fragment_id', '?')!r}"
        )

    if end > start:
        return end - start

    return DEFAULT_CLIP_DURATION


def find_font_file() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return None


def escape_drawtext_text(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def build_lower_third(clip: dict[str, Any]) -> dict[str, str] | None:
    if clip.get("role") != "speaker_soundbite":
        return None

    speaker = (clip.get("speaker") or clip.get("speaker_name") or "").strip()
    context = (
        (clip.get("primary_chapter") or "").strip()
        or (clip.get("primary_theme") or "").strip()
    )

    if not speaker:
        return None

    result = {"speaker": speaker}
    if context:
        result["context"] = context
    return result


def make_card_video_with_optional_voiceover(
    image_path: Path,
    output_path: Path,
    min_duration: float,
    audio_path: Path | None = None,
) -> None:
    """
    Creates a cinematic card segment:
    - slow zoom
    - fade in/out
    - optional section voice-over
    """

    audio_duration = 0.0

    if audio_path and audio_path.exists():
        audio_duration = get_media_duration(audio_path)

    duration = max(min_duration, audio_duration + 0.8)

    # scale+crop is ~5× faster than zoompan (no per-frame computation) while still
    # giving a slightly zoomed-in cinematic look with fade in/out.
    zoom_filter = (
        f"scale={OUTPUT_WIDTH + 120}:{OUTPUT_HEIGHT + 68}:flags=lanczos,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(iw-ow)/2:(ih-oh)/2,"
        f"setsar=1,"
        f"fps={FPS},"
        f"fade=t=in:st=0:d={FADE_DURATION},"
        f"fade=t=out:st={max(duration - FADE_DURATION, 0)}:d={FADE_DURATION},"
        f"format=yuv420p"
    )

    if audio_path and audio_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-t", str(duration),
            "-vf", zoom_filter,
            "-filter:a", f"volume={SECTION_VOICEOVER_VOLUME},afade=t=in:st=0:d=0.2,afade=t=out:st={max(audio_duration - 0.25, 0)}:d=0.25,apad=pad_dur=0.6",
            "-shortest",
            "-c:v", "libx264",
            "-threads", "2",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", str(duration),
            "-vf", zoom_filter,
            "-shortest",
            "-c:v", "libx264",
            "-threads", "2",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

    run_cmd(cmd)


def build_video_filter(
    duration: float,
    lower_third: dict[str, str] | None = None,
) -> str:
    fade_out_start = max(duration - FADE_DURATION, 0)
    filter_parts = [
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease",
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        f"fps={FPS}",
        f"fade=t=in:st=0:d={FADE_DURATION}",
        f"fade=t=out:st={fade_out_start}:d={FADE_DURATION}",
    ]

    if lower_third:
        font_file = find_font_file()
        speaker = escape_drawtext_text(lower_third.get("speaker", ""))
        context = escape_drawtext_text(lower_third.get("context", ""))
        enable = f"lt(t\\,{LOWER_THIRD_DURATION})"

        filter_parts.append(
            "drawbox="
            "x=50:y=760:w=1040:h=170:"
            "color=black@0.52:t=fill:"
            f"enable='{enable}'"
        )

        if speaker:
            drawtext = (
                "drawtext="
                f"text='{speaker}':"
                f"fontfile={font_file}:"
                "fontcolor=white:"
                "fontsize=38:"
                "x=72:y=790:"
                "shadowcolor=black@0.7:shadowx=2:shadowy=2:"
                f"enable='{enable}'"
            )
            filter_parts.append(drawtext)

        if context:
            drawtext = (
                "drawtext="
                f"text='{context}':"
                f"fontfile={font_file}:"
                "fontcolor=white@0.92:"
                "fontsize=26:"
                "x=74:y=842:"
                "shadowcolor=black@0.7:shadowx=2:shadowy=2:"
                f"enable='{enable}'"
            )
            filter_parts.append(drawtext)

    filter_parts.append("format=yuv420p")
    return ",".join(filter_parts)


def render_archive_clip(
    source_path: Path,
    output_path: Path,
    start: float,
    end: float,
    role: str,
    audio_mode: str = "source",
    external_audio_path: Path | None = None,
    target_duration: float | None = None,
    lower_third: dict[str, str] | None = None,
) -> None:
    """
    Renders archive-derived segments for:
    - speaker clips with original audio and optional lower-third
    - archive visual clips with original audio
    - narration b-roll with muted source audio plus external narration audio
    """

    start = float(start or 0)
    end = float(end or 0)

    if end <= start:
        end = start + DEFAULT_CLIP_DURATION

    original_duration = end - start

    if target_duration is not None:
        duration = float(target_duration)
    elif role == "speaker_soundbite":
        duration = min(original_duration, MAX_SPEAKER_CLIP_DURATION)
    else:
        duration = min(original_duration, MAX_ARCHIVE_VISUAL_DURATION)

    if duration <= 0:
        duration = DEFAULT_CLIP_DURATION

    vf = build_video_filter(duration=duration, lower_third=lower_third)

    if audio_mode == "external":
        if not external_audio_path or not external_audio_path.exists():
            raise FileNotFoundError(f"External audio not found: {external_audio_path}")

        audio_duration = get_media_duration(external_audio_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-i", str(external_audio_path),
            "-t", str(duration),
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a", f"volume={SECTION_VOICEOVER_VOLUME},afade=t=in:st=0:d=0.2,afade=t=out:st={max(audio_duration - 0.25, 0)}:d=0.25,apad=pad_dur=0.4",
            "-shortest",
            "-c:v", "libx264",
            "-threads", "2",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
    elif audio_mode == "muted":
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(duration),
            "-vf", vf,
            "-an",
            "-c:v", "libx264",
            "-threads", "2",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
    else:
        af = (
            f"volume={ARCHIVE_AUDIO_VOLUME},"
            f"afade=t=in:st=0:d=0.15,"
            f"afade=t=out:st={max(duration - 0.25, 0)}:d=0.25"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(duration),
            "-vf", vf,
            "-af", af,
            "-c:v", "libx264",
            "-threads", "2",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

    run_cmd(cmd)


def concat_segments(segment_paths: list[Path], output_path: Path) -> None:
    concat_file = output_path.parent / "concat_list.txt"

    with concat_file.open("w", encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{path.resolve()}'\n")

    # Re-encode on concat to fix timestamp discontinuities from mixed segments.
    # -c copy causes freeze/drift when segments have differing pts/dts origins.
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+genpts",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-vf", "fps=30,setpts=N/(30*TB),format=yuv420p",
        "-af", "aresample=async=1:first_pts=0",
        "-c:v", "libx264",
        "-threads", "0",
        "-preset", "veryfast",
        "-r", "30",
        "-vsync", "cfr",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    run_cmd(cmd)


_VIDEO_STREAM_DUR_CACHE: dict[str, float] = {}


def get_video_stream_duration(source_path: Path) -> float:
    """Return the video stream duration (NOT format/audio duration). Cached per path."""
    key = str(source_path)
    if key in _VIDEO_STREAM_DUR_CACHE:
        return _VIDEO_STREAM_DUR_CACHE[key]
    try:
        result = run_cmd_capture([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ])
        dur = float(result.strip() or "0")
    except Exception:
        dur = 0.0
    _VIDEO_STREAM_DUR_CACHE[key] = dur
    return dur


MAX_USES_PER_SOURCE_VIDEO = 2
MAX_USES_PER_SPEAKER = 2
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

    # Validation: if a requested speaker was specified and speaker_visual_allowed candidates exist,
    # ensure the timeline contains at least one speaker_clip or speaker_visual segment.
    prompt_intent = manifest.get("prompt_intent") or {}
    requested_speaker = prompt_intent.get("requested_speaker")
    speaker_visual_issues = []
    if requested_speaker:
        selected_candidates = manifest.get("selected_candidates") or []
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
                "speaker": best.get("speaker"),
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
            speaker_visual_issues.append(msg)
        elif not timeline_has_speaker and not speaker_visual_candidates:
            msg = (
                f"[speaker-visual-validation] No speaker segments in timeline and no "
                f"speaker_visual_allowed candidates available for '{requested_speaker}'. "
                f"Narration-only output is expected."
            )
            log(msg)
            speaker_visual_issues.append(msg)

    quality_checks = {
        "unique_fragments": len(tracker.used_fragment_ids),
        "repeated_fragments": repeated_fragments,
        "source_video_use_counts": dict(tracker.used_source_videos),
        "speaker_use_counts": dict(tracker.used_speakers),
        "fallback_segments": fallback_segments,
        "speaker_visual_validation": speaker_visual_issues,
    }

    return timeline, quality_checks


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

        segment_entries = make_subtitle_entries_for_segment(
            text=subtitle_text,
            segment_start=float(info["start_time"]),
            segment_end=float(info["end_time"]),
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


def burn_subtitles(input_video: Path, subtitle_path: Path, output_path: Path) -> None:
    subtitle_filter_path = str(subtitle_path.resolve()).replace(":", "\\:")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", f"subtitles='{subtitle_filter_path}':force_style='Fontsize=16,Outline=1,Shadow=1,MarginV=32,Alignment=2'",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    run_cmd(cmd)


def generate_final_audio_subtitles(output_path: Path) -> None:
    if not FINAL_SUBTITLE_SCRIPT.exists():
        raise FileNotFoundError(f"Final subtitle script not found: {FINAL_SUBTITLE_SCRIPT}")

    cmd = [
        sys.executable,
        str(FINAL_SUBTITLE_SCRIPT),
        "--video",
        str(output_path),
        "--model",
        "small",
        "--update-manifest",
    ]
    run_cmd_logged(cmd)


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
    fragment_item_index = 0  # 0-based counter for fragment items (for positional voiceover matching)
    _relay_section_idx = 0  # track section index for bridge narration in chronological_relay

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

            if not card_path:
                log(f"  [canonical] Card item {idx} ({seg_type}) has no path — skipping.")
                skipped_count += 1
                continue

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
                        _bridge_seg: dict[str, Any] = {
                            "type": "archive_clip",
                            "segment_type": "narration_broll",
                            "audio_mode": "external",
                            "audio_path": str(_bridge_audio) if _bridge_audio else None,
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
                        str(fragment_item_index),
                    ):
                        if key and key in sv_map:
                            audio_path = sv_map[key]
                            break

                if audio_path:
                    item["audio_path"] = str(audio_path)
                    item["segment_type"] = "narration_broll"
                    item["audio_mode"] = "ai_voiceover"
                    probed_audio_dur = get_media_duration(audio_path)
                    if probed_audio_dur <= 0.0:
                        log(f"  [canonical] WARNING: voiceover {audio_path.name} probed as 0s — using fragment duration")
                    narration_duration = max(probed_audio_dur, 0.0) + 0.4
                    frag_dur = float(item.get("duration") or 0.0)
                    item["duration"] = max(frag_dur, narration_duration)
                    item["render_duration"] = item["duration"]
                    # Propagate subtitle text from section_voiceovers if not already set
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
                    seg_type = "narration_broll"
                    narration_broll_count += 1
                    log(f"  [canonical] Attached voiceover to item {idx} (fragment_item_index={fragment_item_index}): {audio_path.name} ({probed_audio_dur:.2f}s → render {item['duration']:.2f}s)")
                else:
                    item["audio_mode"] = "silent"
                    archive_visual_count += 1

            fragment_item_index += 1
            rendering.append(item)
        else:
            log(f"  [canonical] Unrecognised item {idx} (type={item_type!r}, seg={seg_type!r}) — skipping.")
            skipped_count += 1

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


def _render_one_segment(
    idx: int,
    item: dict[str, Any],
    work_dir: Path,
    target_sec_per_segment: float,
) -> tuple[int, Path | None, dict[str, Any] | None]:
    """Render one timeline segment to a file. Returns (idx, path, partial_info) or (idx, None, None) to skip."""
    segment_label = slugify(item.get("label") or item.get("fragment_id") or f"segment_{idx}", 70)
    segment_path = work_dir / f"{idx:03d}_{segment_label}.mp4"

    item_type = item.get("type", "")
    seg_type = item.get("segment_type", "")

    if item_type == "card":
        image_path = Path(item.get("path", ""))
        audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
        min_duration = float(item.get("duration") or STATIC_CARD_DURATION)

        if not image_path.exists():
            log(f"  [render] Missing card image, skipping: {image_path}")
            return idx, None, None

        make_card_video_with_optional_voiceover(
            image_path=image_path,
            output_path=segment_path,
            min_duration=min_duration,
            audio_path=audio_path,
        )

    elif (
        item_type == "archive_clip"
        or item.get("fragment_id")
        or seg_type in ("narration_broll", "speaker_clip", "archive_visual", "speaker_visual")
        or item_type in ("narration_broll", "speaker_clip", "archive_visual", "speaker_visual")
    ):
        fragment_id = item.get("fragment_id")
        source_video = item.get("source_video")
        role = item.get("role", seg_type or "archive_visual")
        source_path = find_source_video(source_video, fragment_id)

        item_audio_mode = item.get("audio_mode") or ""

        _has_generated_visual = bool(
            item.get("render_visual_path") or item.get("visual_path") or item.get("asset_path")
        )
        _is_narration = seg_type == "narration_broll" or item_type == "narration_broll" or item_audio_mode == "ai_voiceover"
        if not source_path and not (_is_narration and _has_generated_visual):
            log(f"  [render] Source video not found for fragment {fragment_id!r} (source_video={source_video!r}). Skipping.")
            return idx, None, None

        if item_type not in ("narration_broll", "speaker_clip"):
            item["type"] = "archive_clip"

        external_audio_path = None
        target_duration = None
        lower_third = item.get("lower_third")
        ffmpeg_audio_mode: str

        video_stream_dur = get_video_stream_duration(source_path) if source_path else 0.0

        if seg_type == "speaker_clip" or item_audio_mode == "source_audio":
            start = float(item.get("render_start") or item.get("start") or 0)
            end = float(item.get("render_end") or item.get("end") or (start + DEFAULT_CLIP_DURATION))
            if end <= start:
                end = start + DEFAULT_CLIP_DURATION
            target_duration = float(item.get("render_duration") or (end - start))

            no_video_frames = video_stream_dur > 0 and start >= (video_stream_dur - 0.5)
            if no_video_frames:
                log(
                    f"  [render] speaker_clip {fragment_id[:40] if fragment_id else '?'!r}: "
                    f"video ends at {video_stream_dur:.1f}s — ts={start:.1f}s beyond video track. "
                    f"Rendering with black video + source audio."
                )
                vf = build_video_filter(duration=target_duration, lower_third=lower_third)
                run_cmd([
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=black:size={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:duration={target_duration:.3f}:rate={FPS}",
                    "-ss", str(start),
                    "-i", str(source_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-t", str(target_duration),
                    "-vf", vf,
                    "-filter:a", f"volume={ARCHIVE_AUDIO_VOLUME},afade=t=in:st=0:d=0.15,afade=t=out:st={max(target_duration - 0.25, 0):.3f}:d=0.25",
                    "-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-pix_fmt", "yuv420p",
                    str(segment_path),
                ])
            else:
                ffmpeg_audio_mode = "source"
                log(
                    f"  [render] speaker_clip {fragment_id[:40] if fragment_id else '?'!r}: "
                    f"source_audio {start:.2f}→{end:.2f} ({target_duration:.2f}s)"
                )
                render_archive_clip(
                    source_path=source_path,
                    output_path=segment_path,
                    start=start,
                    end=end,
                    role=role,
                    audio_mode=ffmpeg_audio_mode,
                    external_audio_path=None,
                    target_duration=target_duration,
                    lower_third=lower_third,
                )

            duration = get_media_duration(segment_path)
            if duration == 0.0:
                raise RuntimeError(
                    f"Rendered segment {idx} (speaker_clip, fragment={fragment_id!r}) "
                    f"has duration 0.0 at {segment_path}. Aborting export."
                )
            return idx, segment_path, {
                "index": idx,
                "path": str(segment_path),
                "timeline_item": item,
                "duration": duration,
                "subtitle_text": "",
                "rendered": True,
                "segment_type": seg_type,
                "fragment_id": item.get("fragment_id"),
                "muted": item.get("muted", False),
            }

        elif seg_type == "narration_broll" or item_audio_mode == "ai_voiceover":
            start = float(item.get("start") or 0)
            end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))
            if end <= start:
                end = start + DEFAULT_CLIP_DURATION
            ffmpeg_audio_mode = "external"
            external_audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
            if external_audio_path and not external_audio_path.exists():
                log(f"  [render] External audio not found: {external_audio_path}. Falling back to muted.")
                ffmpeg_audio_mode = "muted"
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
            target_duration = max(tts_duration, target_sec_per_segment)
            _vis_path_str = (
                item.get("render_visual_path")
                or item.get("visual_path")
                or item.get("asset_path")
            )
            if _vis_path_str:
                _vis_p = Path(_vis_path_str)
                if _vis_p.exists():
                    card_min_dur = max(target_duration, target_sec_per_segment)
                    log(f"  [render] narration_broll item {idx}: generated visual {_vis_p.name} + ai_voiceover (min_dur={card_min_dur:.1f}s)")
                    make_card_video_with_optional_voiceover(
                        image_path=_vis_p,
                        output_path=segment_path,
                        min_duration=card_min_dur,
                        audio_path=external_audio_path,
                    )
                    _seg_dur = get_media_duration(segment_path)
                    if _seg_dur == 0.0:
                        raise RuntimeError(
                            f"Rendered narration_broll visual segment {idx} (fragment={fragment_id!r}) "
                            f"has duration 0.0. Aborting."
                        )
                    return idx, segment_path, {
                        "index": idx,
                        "path": str(segment_path),
                        "timeline_item": item,
                        "duration": _seg_dur,
                        "subtitle_text": item.get("voiceover_text") or item.get("subtitle_text") or "",
                        "rendered": True,
                        "segment_type": "narration_broll",
                        "fragment_id": item.get("fragment_id"),
                        "muted": False,
                        "visual_mode": "generated_asset",
                    }
            if not source_path:
                log(f"  [render] narration_broll item {idx}: no generated visual and no source video. Skipping.")
                return idx, None, None
            if video_stream_dur > 0 and start >= (video_stream_dur - 0.5):
                fallback_start = max(0.0, video_stream_dur - target_duration - 1.0)
                fallback_start = max(0.0, fallback_start)
                log(
                    f"  [render] narration_broll item {idx}: source video ends at {video_stream_dur:.1f}s "
                    f"(requested ts={start:.1f}s). Using fallback visual at ts={fallback_start:.1f}s."
                )
                start = fallback_start
                end = fallback_start + target_duration
            lower_third = None

        elif seg_type == "archive_visual" or item_audio_mode == "silent" or item.get("muted") is True:
            start = float(item.get("start") or 0)
            end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))
            if end <= start:
                end = start + DEFAULT_CLIP_DURATION
            ffmpeg_audio_mode = "muted"
            lower_third = None

        elif seg_type == "speaker_visual":
            start = float(item.get("start") or 0)
            end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))
            if end <= start:
                end = start + DEFAULT_CLIP_DURATION
            ffmpeg_audio_mode = "source"

        else:
            start = float(item.get("start") or 0)
            end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))
            if end <= start:
                end = start + DEFAULT_CLIP_DURATION
            ffmpeg_audio_mode = "source"

        render_archive_clip(
            source_path=source_path,
            output_path=segment_path,
            start=start,
            end=end,
            role=role,
            audio_mode=ffmpeg_audio_mode,
            external_audio_path=external_audio_path,
            target_duration=target_duration,
            lower_third=lower_third,
        )

    else:
        log(f"  [render] Unknown timeline item type: {item_type!r} — skipping.")
        return idx, None, None

    duration = get_media_duration(segment_path)
    if duration == 0.0:
        seg_type_check = item.get("segment_type") or item.get("type") or "?"
        raise RuntimeError(
            f"Rendered segment {idx} ({seg_type_check}, fragment={item.get('fragment_id')!r}) "
            f"has duration 0.0 at {segment_path}. Aborting export."
        )
    return idx, segment_path, {
        "index": idx,
        "path": str(segment_path),
        "timeline_item": item,
        "duration": duration,
        "subtitle_text": "",
        "rendered": True,
        "segment_type": seg_type,
        "fragment_id": item.get("fragment_id"),
        "muted": item.get("muted", False),
    }


def _render_timeline_items(
    timeline: list[dict[str, Any]],
    work_dir: Path,
    target_sec_per_segment: float = 0.0,
) -> tuple[list[Path], list[dict[str, Any]], float]:
    """
    Render all timeline segments in parallel (up to 4 concurrent ffmpeg processes),
    then assemble results in original order with sequential timestamps.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    MAX_WORKERS = min(4, max(1, len(timeline)))
    raw_results: dict[int, tuple[Path | None, dict[str, Any] | None]] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_render_one_segment, idx, item, work_dir, target_sec_per_segment): idx
            for idx, item in enumerate(timeline, start=1)
        }
        for future in as_completed(futures):
            idx, path, info = future.result()  # re-raises any RuntimeError from render
            raw_results[idx] = (path, info)

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


def export_video(manifest_path: Path, output_name: str | None, burn_subs: bool) -> Path:
    manifest = load_manifest(manifest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_slug = slugify(manifest.get("title", "ai4ia_documentary"), 50)

    export_id = f"{timestamp}_{title_slug}"
    work_dir = TEMP_DIR / export_id
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

    segment_paths, segment_infos, current_time = _render_timeline_items(timeline, work_dir, _target_sec_per_segment)

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

    if not segment_paths:
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

    base_concat_path = work_dir / "base_concat.mp4"
    concat_segments(segment_paths, base_concat_path)

    if burn_subs:
        burn_subtitles(base_concat_path, srt_output_path, output_path)
    else:
        shutil.copyfile(base_concat_path, output_path)

    # Fix 6: A/V ending balance — if video stream is shorter than audio, pad last frame
    try:
        _vid_dur_check = float(run_cmd_capture([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ]).strip() or "0")
        _aud_dur_check = float(run_cmd_capture([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ]).strip() or "0")
        _av_delta = _aud_dur_check - _vid_dur_check
        if _av_delta > 1.0:
            log(f"  A/V balance: video={_vid_dur_check:.2f}s audio={_aud_dur_check:.2f}s delta={_av_delta:.2f}s — padding video stream")
            _padded = output_path.parent / (output_path.stem + "_avpad.mp4")
            run_cmd([
                "ffmpeg", "-y", "-i", str(output_path),
                "-vf", f"tpad=stop_mode=clone:stop_duration={_av_delta + 0.15:.3f}",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "copy",
                "-pix_fmt", "yuv420p",
                str(_padded),
            ])
            _padded.replace(output_path)
            log(f"  A/V balance: video stream padded — final duration ≈ {_aud_dur_check:.2f}s")
    except Exception as _av_exc:
        log(f"  A/V balance check failed (non-fatal): {_av_exc}")

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
        _speaker = (
            _ti.get("speaker") or _ti.get("speaker_name") or _ti.get("primary_speaker")
        ) if _stype == "speaker_clip" else None
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

    generate_final_audio_subtitles(output_path)

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
        if dur_delta > 1.0:
            log(f"WARNING: duration_delta ({dur_delta:.2f}s) exceeds 1 second — possible A/V sync issue.")
        export_manifest["duration_check"] = {
            "format_duration": fmt_dur,
            "video_stream_duration": vid_dur,
            "audio_stream_duration": aud_dur,
            "duration_delta": dur_delta,
        }
        export_manifest_path.write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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

    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()

    log("Starting Step 09: Export Refined Final Video")
    log(f"Manifest: {manifest_path}")

    output_path = export_video(
        manifest_path=manifest_path,
        output_name=args.output,
        burn_subs=args.burn_subtitles and not args.no_subtitles,
    )

    log(f"Done: {output_path}")


if __name__ == "__main__":
    main()
