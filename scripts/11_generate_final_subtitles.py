import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path("/tmp/ai4ia_pipeline/subtitles_temp")


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4G: Generate final SRT/VTT subtitles from exported MP4 audio."
    )
    parser.add_argument("--video", required=True, help="Path to final exported video (.mp4/.mov/.mkv)")
    parser.add_argument("--output-srt", help="Optional output SRT path")
    parser.add_argument("--output-vtt", help="Optional output VTT path")
    parser.add_argument("--model", default="small", help="Whisper model: base|small|medium|large-v3")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Compute type: auto|int8|float16|int8_float16",
    )
    parser.add_argument("--language", default="en", help="Language code, default en")
    parser.add_argument("--update-manifest", action="store_true", help="Update export manifest JSON if present")
    parser.add_argument("--run-qa", action="store_true", help="Run export QA after subtitle generation")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp WAV if ffmpeg fallback is used")
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing output files (default: true)",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if output files already exist",
    )
    return parser.parse_args()


def validate_video_path(video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    if video_path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise ValueError("Video file must be one of: .mp4, .mov, .mkv")

    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")


def resolve_output_paths(
    video_path: Path,
    output_srt: str | None,
    output_vtt: str | None,
) -> tuple[Path, Path]:
    srt_path = Path(output_srt).resolve() if output_srt else video_path.with_suffix(".srt")
    vtt_path = Path(output_vtt).resolve() if output_vtt else video_path.with_suffix(".vtt")
    return srt_path, vtt_path


def ensure_writable(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists and overwrite is disabled: {path}"
        )

    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory, not a file: {path}")


def resolve_device_and_compute_type(device: str, compute_type: str) -> tuple[str, str]:
    resolved_device = device

    if resolved_device == "auto":
        try:
            import torch  # type: ignore

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"

    resolved_compute = compute_type
    if resolved_compute == "auto":
        resolved_compute = "int8_float16" if resolved_device == "cuda" else "int8"

    return resolved_device, resolved_compute


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def format_srt_timestamp(seconds: float) -> str:
    safe_seconds = max(0.0, float(seconds))
    total_milliseconds = int(round(safe_seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    return format_srt_timestamp(seconds).replace(",", ".")


def write_srt(segments: list[dict[str, float | str]], path: Path) -> None:
    lines: list[str] = []
    for idx, segment in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(
            f"{format_srt_timestamp(float(segment['start']))} --> {format_srt_timestamp(float(segment['end']))}"
        )
        lines.append(str(segment["text"]))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict[str, float | str]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.append(
            f"{format_vtt_timestamp(float(segment['start']))} --> {format_vtt_timestamp(float(segment['end']))}"
        )
        lines.append(str(segment["text"]))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def extract_audio_fallback(video_path: Path, keep_temp: bool) -> Path:
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="final_subtitles_", dir=TEMP_ROOT))
    wav_path = temp_dir / f"{video_path.stem}.subtitle_audio.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        stderr = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(
            "ffmpeg fallback failed. Ensure ffmpeg is installed and accessible. "
            f"Details: {stderr}"
        )

    if keep_temp:
        log(f"Kept temp audio: {wav_path}")

    return wav_path


def load_whisper_model(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install dependencies or run pip install faster-whisper."
        ) from exc

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def materialize_segments(segments_iter) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []

    for segment in segments_iter:
        text = clean_text(segment.text)
        if not text:
            continue

        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if end == start:
            end = start + 0.01

        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    return segments


def transcribe_media(model, media_path: Path, language: str) -> tuple[list[dict[str, float | str]], object]:
    segments_iter, info = model.transcribe(
        str(media_path),
        language=language,
        task="transcribe",
        vad_filter=True,
        beam_size=5,
    )
    segments = materialize_segments(segments_iter)
    return segments, info


def transcribe_video(
    video_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    keep_temp: bool,
) -> tuple[list[dict[str, float | str]], object, str, Path | None]:
    model = load_whisper_model(model_name, device, compute_type)

    try:
        segments, info = transcribe_media(model, video_path, language)
        return segments, info, "direct_mp4", None
    except Exception as direct_exc:
        log(f"Direct media transcription failed, falling back to ffmpeg WAV extraction: {direct_exc}")

        wav_path = extract_audio_fallback(video_path, keep_temp=keep_temp)
        try:
            segments, info = transcribe_media(model, wav_path, language)
            return segments, info, "ffmpeg_wav_fallback", wav_path
        except Exception:
            if wav_path.exists() and not keep_temp:
                wav_path.unlink()
                wav_path.parent.rmdir()
            raise


def update_manifest(
    video_path: Path,
    srt_path: Path,
    vtt_path: Path,
    model_name: str,
    language: str,
    segment_count: int,
    duration: float,
    warning: str | None = None,
) -> bool:
    manifest_path = video_path.with_name(f"{video_path.stem}_export_manifest.json")
    if not manifest_path.exists():
        log(f"Warning: manifest not found, skipping update: {manifest_path}")
        return False

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    payload["final_audio_subtitles"] = {
        "enabled": True,
        "method": "faster_whisper",
        "model": model_name,
        "language": language,
        "srt_path": str(srt_path),
        "vtt_path": str(vtt_path),
        "segment_count": segment_count,
        "duration": duration,
        "source": "final_mp4_audio",
    }
    if warning:
        payload["final_audio_subtitles"]["warning"] = warning

    subtitles = dict(payload.get("subtitles") or {})
    subtitles["srt_path"] = str(srt_path)
    subtitles["vtt_path"] = str(vtt_path)
    subtitles["entry_count"] = segment_count
    subtitles["source"] = "final_mp4_audio"
    if warning:
        subtitles["warning"] = warning
    payload["subtitles"] = subtitles

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def run_qa(video_path: Path) -> bool:
    manifest_path = video_path.with_name(f"{video_path.stem}_export_manifest.json")
    if not manifest_path.exists():
        log(f"Warning: manifest not found, skipping QA: {manifest_path}")
        return False

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "10_validate_video_export.py"),
        "--export-manifest",
        str(manifest_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown QA error"
        raise RuntimeError(f"QA validation failed: {stderr}")
    if result.stderr.strip():
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return True


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).resolve()
    srt_path, vtt_path = resolve_output_paths(video_path, args.output_srt, args.output_vtt)

    temp_audio_path: Path | None = None
    manifest_updated = False
    qa_ran = False
    subtitle_warning: str | None = None

    try:
        validate_video_path(video_path)
        ensure_writable(srt_path, overwrite=args.overwrite)
        ensure_writable(vtt_path, overwrite=args.overwrite)

        device, compute_type = resolve_device_and_compute_type(args.device, args.compute_type)

        log("Starting Step 11: Generate Final Subtitles from Final MP4 Audio")
        log(f"Python executable: {sys.executable}")
        log(f"Video: {video_path}")
        log(f"Model: {args.model}")
        log(f"Device: {device}")
        log(f"Compute type: {compute_type}")

        segments, info, transcription_source, temp_audio_path = transcribe_video(
            video_path=video_path,
            model_name=args.model,
            device=device,
            compute_type=compute_type,
            language=args.language,
            keep_temp=args.keep_temp,
        )

        if not segments:
            subtitle_warning = "Transcription produced no speech segments; output appears silent or non-verbal."
            log(f"Warning: {subtitle_warning}")

        write_srt(segments, srt_path)
        write_vtt(segments, vtt_path)

        detected_language = getattr(info, "language", args.language) or args.language
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        if args.update_manifest:
            manifest_updated = update_manifest(
                video_path=video_path,
                srt_path=srt_path,
                vtt_path=vtt_path,
                model_name=args.model,
                language=str(detected_language),
                segment_count=len(segments),
                duration=duration,
                warning=subtitle_warning,
            )

        if args.run_qa:
            qa_ran = run_qa(video_path)

        log("Final subtitle generation complete")
        log(f"Video: {video_path}")
        log(f"Model: {args.model}")
        log(f"Device: {device}")
        log(f"Segments: {len(segments)}")
        log(f"SRT: {srt_path}")
        log(f"VTT: {vtt_path}")
        log(f"Manifest updated: {'yes' if manifest_updated else 'no'}")
        log(f"QA run: {'yes' if qa_ran else 'no'}")
        log(f"Transcription source: {transcription_source}")

    except Exception as exc:
        log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    finally:
        if temp_audio_path and temp_audio_path.exists() and not args.keep_temp:
            try:
                temp_audio_path.unlink()
                temp_audio_path.parent.rmdir()
            except Exception:
                pass


if __name__ == "__main__":
    main()
