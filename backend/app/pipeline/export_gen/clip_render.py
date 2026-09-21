from __future__ import annotations
from pathlib import Path
from typing import Any

from app.pipeline.media_probe import media_duration_or_raise as get_media_duration
from app.pipeline.export_gen.ffmpeg_utils import run_cmd, _vcodec_flags

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

# Fraction a card/generated-visual is zoomed in over its own length (Ken Burns).
# This filter used to be a plain scale+crop with a FIXED centre offset, which is
# the same pixels every frame — every narration segment was a literally frozen
# image for its whole 26-41s, and freezedetect confirmed 270s of a 277s export
# had zero motion. The crop filter only animates with eval=frame, which needs
# ffmpeg 5+; this runs on 4.4, so zoompan is the portable way to get real
# motion. zoompan steps in integer pixels, so the source is upscaled first
# (ZOOM_SUPERSAMPLE) or the pan visibly stair-steps.
KEN_BURNS_ZOOM = 0.08
ZOOM_SUPERSAMPLE = 2

ARCHIVE_AUDIO_VOLUME = 1.0
SECTION_VOICEOVER_VOLUME = 1.0

# Intermediate segments carry LOSSLESS audio; AAC is encoded exactly once, by
# concat_segments. AAC is a block codec: every encode adds priming samples at
# the head and padding at the tail, and the container records them in an edit
# list so a player can trim them. The concat demuxer ignores that edit list and
# decodes raw frames, so each per-segment AAC encode leaked ~7 ms of audio into
# the join while the video side stayed exact (setpts rebuilds it from frame
# count). Lip sync therefore drifted by roughly 7 ms x segment_count — 141 ms
# over 18 segments, past the point a viewer sees it against a talking face.
# Measured on real footage: 7.15 ms/segment before, 0.21 ms/segment after.
SEGMENT_AUDIO_CODEC = "pcm_s16le"
# MP4 is a poor PCM carrier; these files are throwaway inputs to concat.
SEGMENT_CONTAINER_SUFFIX = ".mkv"


def _first_present(*vals: Any, default: float = 0.0) -> float:
    """Like `a or b or default`, but treats 0.0/0 as present rather than falsy."""
    for v in vals:
        if v is not None and v != "":
            return float(v)
    return default


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
    """Escape text for embedding inside a single-quoted drawtext value
    (callers always wrap the result as text='{...}').

    A literal apostrophe can't be escaped as `\\'` inside an ffmpeg
    filtergraph single-quoted string — that backslash prints literally and
    the quote still closes right there, leaving everything after it (the
    rest of this value, every later option on the same filter, and any
    filter after it in the chain) to be parsed as raw unquoted syntax. That
    corrupted a downstream `enable=` expression and swallowed the next
    filter's name into it, aborting the whole ffmpeg command — reproduced
    live with a speaker name containing an apostrophe (e.g. "D'Souza").
    The correct escape is to close the quote, emit a literal quote via
    concatenation, and reopen: '\\''  (must run before the other
    replacements below, none of which touch a raw `'`.)
    """
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "'\\''")
        .replace("%", "\\%")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def build_lower_third(clip: dict[str, Any]) -> dict[str, str] | None:
    if clip.get("role") != "speaker_soundbite" and clip.get("segment_type") != "speaker_clip":
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

    # Slow centred zoom over the segment's full length. Centred (rather than a
    # diagonal pan) because these are text cards: panning drifts the headline
    # and eyebrow off the edge, while a centred zoom only ever loses
    # KEN_BURNS_ZOOM/2 from each side, which the card's margins absorb.
    zoom_frames = max(1, round(duration * FPS))
    zoom_filter = (
        f"scale={OUTPUT_WIDTH * ZOOM_SUPERSAMPLE}:{OUTPUT_HEIGHT * ZOOM_SUPERSAMPLE}:flags=lanczos,"
        f"zoompan="
        f"z='min(1+{KEN_BURNS_ZOOM}*on/{zoom_frames},{1 + KEN_BURNS_ZOOM})':"
        f"d=1:"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps={FPS},"
        f"setsar=1,"
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
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC,
            "-ar", "48000",
            "-ac", "2",
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
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC,
            "-ar", "48000",
            "-ac", "2",
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
        "tpad=stop_mode=clone:stop=-1"
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
    stream_loop: bool = False,
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
            *(["-stream_loop", "-1"] if stream_loop else []),
            "-ss", str(start),
            "-i", str(source_path),
            "-i", str(external_audio_path),
            "-t", str(duration),
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a", f"volume={SECTION_VOICEOVER_VOLUME},afade=t=in:st=0:d=0.2,afade=t=out:st={max(audio_duration - 0.25, 0)}:d=0.25,apad=pad_dur=0.4",
            "-shortest",
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC,
            "-ar", "48000",
            "-ac", "2",
            str(output_path),
        ]
    elif audio_mode == "muted":
        # A segment with NO audio stream at all (-an) breaks HLS/MSE playback
        # mid-stream: browsers require every segment in the stream to carry
        # an audio track with identical codec/sample-rate/channel-layout —
        # one segment missing audio entirely gets silently rejected along
        # with everything after it (confirmed by the frontend team on the
        # live HLS preview). Synthesize a silent AAC track instead, encoded
        # with the exact same settings as every other segment.
        cmd = [
            "ffmpeg",
            "-y",
            *(["-stream_loop", "-1"] if stream_loop else []),
            "-ss", str(start),
            "-i", str(source_path),
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", str(duration),
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC,
            "-ar", "48000",
            "-ac", "2",
            str(output_path),
        ]
    else:
        af = (
            f"volume={ARCHIVE_AUDIO_VOLUME},"
            f"afade=t=in:st=0:d=0.15,"
            f"afade=t=out:st={max(duration - 0.25, 0)}:d=0.25,"
            "apad"
        )

        cmd = [
            "ffmpeg",
            "-y",
            *(["-stream_loop", "-1"] if stream_loop else []),
            "-ss", str(start),
            "-i", str(source_path),
            "-t", str(duration),
            "-vf", vf,
            "-af", af,
            *_vcodec_flags(threads=2),
            "-c:a", SEGMENT_AUDIO_CODEC,
            "-ar", "48000",
            "-ac", "2",
            str(output_path),
        ]

    run_cmd(cmd)


def concat_segments(segment_paths: list[Path], output_path: Path, subtitle_path: Path | None = None) -> None:
    """Concat all rendered segments into one file, re-encoding once.

    Re-encode on concat to fix timestamp discontinuities from mixed segments.
    -c copy causes freeze/drift when segments have differing pts/dts origins —
    this was tested and reverted before, keep the re-encode.

    subtitle_path is optional: when given, the burn-in subtitles filter is
    folded into this same encode pass instead of running a second full-length
    ffmpeg pass over the output afterward (burn_subtitles() used to do that;
    subtitle timing is already known at this point since it comes from
    segment_infos computed before concat, so there's no ordering reason to
    keep it as a separate pass).
    """
    concat_file = output_path.parent / "concat_list.txt"

    with concat_file.open("w", encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{path.resolve()}'\n")

    vf_chain = "fps=30,setpts=N/(30*TB),format=yuv420p"
    if subtitle_path is not None:
        subtitle_filter_path = str(subtitle_path.resolve()).replace(":", "\\:")
        vf_chain += (
            f",subtitles='{subtitle_filter_path}':eof_action=pass:"
            "force_style='Fontsize=16,Outline=1,Shadow=1,MarginV=32,Alignment=2'"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+genpts",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-vf", vf_chain,
        "-af", "aresample=async=1:first_pts=0",
        *_vcodec_flags(threads=0),
        "-r", "30",
        "-vsync", "cfr",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(output_path),
    ]

    run_cmd(cmd)
