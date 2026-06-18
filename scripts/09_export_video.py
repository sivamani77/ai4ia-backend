import argparse
import json
import re
import shutil
import subprocess
import sys
from math import ceil
from pathlib import Path
from datetime import datetime
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
GENERATED_ASSETS_DIR = PROJECT_ROOT / "data" / "generated_assets"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"
TEMP_DIR = PROJECT_ROOT / "temp" / "video_export"

LATEST_MANIFEST_PATH = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"
FINAL_SUBTITLE_SCRIPT = PROJECT_ROOT / "scripts" / "11_generate_final_subtitles.py"

EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
        log(result.stdout)
        log(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


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
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


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
    except Exception:
        return 0.0


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
        except Exception:
            pass

    try:
        start = float(clip.get("start") or 0)
        end = float(clip.get("end") or 0)
    except Exception:
        return DEFAULT_CLIP_DURATION

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

    total_frames = max(int(duration * FPS), 1)

    zoom_filter = (
        f"scale={OUTPUT_WIDTH + 180}:{OUTPUT_HEIGHT + 100},"
        f"zoompan=z='min(zoom+0.00085,1.08)':"
        f"d={total_frames}:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps={FPS},"
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

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ]

    run_cmd(cmd)


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
            chosen_clip["segment_type"] = chosen_clip_type
            chosen_clip["section_id"] = section_id
            chosen_clip["source_video"] = source_video
            # speaker_visual clips run muted: no subtitle from speaker quote
            chosen_clip["subtitle_text"] = chosen_clip.get("quote_summary", "") if chosen_clip_type == "speaker_clip" else ""
            # speaker_visual keeps its source audio (speaker is speaking); not muted by default
            chosen_clip["muted"] = False
            chosen_clip["selection_reason"] = selection_reason
            chosen_clip["fallback_reason"] = None
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
        return True
    if item_type == "card":
        return bool(segment.get("audio_path"))
    return False


def get_segment_subtitle_text(segment: dict[str, Any]) -> str:
    subtitle_type = classify_segment_subtitle_type(segment)

    if subtitle_type == "speaker":
        keys = [
            "text",
            "transcript_text",
            "quote_text",
            "subtitle_text",
            "quote_summary",
            "summary",
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


def export_video(manifest_path: Path, output_name: str | None, burn_subs: bool) -> Path:
    manifest = load_manifest(manifest_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_slug = slugify(manifest.get("title", "ai4ia_documentary"), 50)

    export_id = f"{timestamp}_{title_slug}"
    work_dir = TEMP_DIR / export_id
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"Work dir: {work_dir}")

    timeline, quality_checks = build_refined_timeline(manifest)
    segment_paths = []
    segment_infos = []
    current_time = 0.0

    for idx, item in enumerate(timeline, start=1):
        segment_label = slugify(item.get("label") or item.get("fragment_id") or f"segment_{idx}", 70)
        segment_path = work_dir / f"{idx:03d}_{segment_label}.mp4"

        if item["type"] == "card":
            image_path = Path(item["path"])
            audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
            min_duration = float(item.get("duration") or STATIC_CARD_DURATION)

            if not image_path.exists():
                log(f"Missing card image, skipping: {image_path}")
                continue

            make_card_video_with_optional_voiceover(
                image_path=image_path,
                output_path=segment_path,
                min_duration=min_duration,
                audio_path=audio_path,
            )

        elif item["type"] == "archive_clip":
            fragment_id = item.get("fragment_id")
            source_video = item.get("source_video")
            role = item.get("role", "archive_visual")
            segment_type = item.get("segment_type", "archive_visual")
            source_path = find_source_video(source_video, fragment_id)

            if not source_path:
                log(f"Source video not found for fragment {fragment_id}. Skipping.")
                continue

            start = float(item.get("start") or 0)
            end = float(item.get("end") or (start + DEFAULT_CLIP_DURATION))

            audio_mode = "source"
            external_audio_path = None
            target_duration = None
            lower_third = item.get("lower_third")

            if segment_type == "narration_broll":
                audio_mode = "external"
                external_audio_path = Path(item["audio_path"]) if item.get("audio_path") else None
                target_duration = float(item.get("duration") or get_clip_candidate_duration(item))
                lower_third = None
            elif segment_type == "speaker_visual" or item.get("muted") is True:
                # Speaker visual: render with source audio (the speaker is speaking),
                # but flag as speaker_visual so subtitles use narration text, not speaker quote
                audio_mode = "source"

            render_archive_clip(
                source_path=source_path,
                output_path=segment_path,
                start=start,
                end=end,
                role=role,
                audio_mode=audio_mode,
                external_audio_path=external_audio_path,
                target_duration=target_duration,
                lower_third=lower_third,
            )

        else:
            log(f"Unknown timeline item type: {item.get('type')}")
            continue

        duration = get_media_duration(segment_path)

        segment_paths.append(segment_path)
        segment_infos.append(
            {
                "index": idx,
                "path": str(segment_path),
                "timeline_item": item,
                "start_time": current_time,
                "end_time": current_time + duration,
                "duration": duration,
                "subtitle_text": "",
            }
        )

        current_time += duration

    if not segment_paths:
        raise RuntimeError("No video segments created.")

    if output_name:
        output_path = EXPORTS_DIR / output_name
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

    rhythm = {
        "enabled": True,
        "narration_broll_segments": sum(1 for item in timeline if item.get("segment_type") == "narration_broll"),
        "static_card_segments": sum(1 for item in timeline if item.get("type") == "card"),
        "speaker_lower_thirds": sum(1 for item in timeline if item.get("lower_third")),
        "fallback_to_card_segments": sum(1 for item in timeline if item.get("segment_type") == "fallback_narration"),
    }

    export_manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "work_dir": str(work_dir),
        "timeline": timeline,
        "segments": segment_infos,
        "burn_subtitles": burn_subs,
        "audio_strategy": "Narration b-roll uses muted archive visuals with external section voice-over. Speaker/archive clips keep original audio unless a narration track is attached.",
        "visual_strategy": "Title and closing remain cards. Section narration prefers documentary b-roll when a unique visual is available. Speaker clips can carry lower-thirds for the opening seconds.",
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
    }

    export_manifest_path = EXPORTS_DIR / f"{output_path.stem}_export_manifest.json"
    export_manifest_path.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    generate_final_audio_subtitles(output_path)

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
