import csv
import json
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = Path(os.getenv("WHISPER_AUDIO_DIR", str(PROJECT_ROOT / "data" / "audio")))
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
SUBTITLE_DIR = PROJECT_ROOT / "data" / "subtitles"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "transcription_report.csv"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Overridable via env so the driver (run_transcribe_safe.py) can force CPU on a
# file that OOMs the 4GB GPU, without editing this file.
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8_float16")

# Transcribe a single named file (driver isolates each file in its own process).
ONLY_FILE = os.getenv("WHISPER_ONLY_FILE")

# Keep 1 for first test. Change to None after test works.
LIMIT = None


def log(message: str) -> None:
    print(message, flush=True)


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(segments: list[dict], output_path: Path) -> None:
    lines = []

    for idx, segment in enumerate(segments, start=1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        text = segment["text"].strip()

        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], output_path: Path) -> None:
    lines = ["WEBVTT", ""]

    for segment in segments:
        start = format_timestamp(segment["start"]).replace(",", ".")
        end = format_timestamp(segment["end"]).replace(",", ".")
        text = segment["text"].strip()

        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_audio(model: WhisperModel, audio_path: Path) -> dict:
    output_json = TRANSCRIPT_DIR / f"{audio_path.stem}.json"
    output_txt = TRANSCRIPT_DIR / f"{audio_path.stem}.txt"
    output_srt = SUBTITLE_DIR / f"{audio_path.stem}.srt"
    output_vtt = SUBTITLE_DIR / f"{audio_path.stem}.vtt"

    if output_json.exists() and output_json.stat().st_size > 0:
        log(f"SKIP existing transcript: {audio_path.name}")
        return {
            "audio_file": audio_path.name,
            "audio_path": str(audio_path),
            "transcript_json": str(output_json),
            "transcript_txt": str(output_txt),
            "subtitle_srt": str(output_srt),
            "subtitle_vtt": str(output_vtt),
            "status": "skipped_exists",
            "language": "",
            "duration_seconds": "",
            "segment_count": "",
            "error": "",
        }

    log(f"TRANSCRIBING: {audio_path.name}")

    try:
        segments_generator, info = model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = []

        for segment in segments_generator:
            segments.append(
                {
                    "id": segment.id,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "avg_logprob": float(segment.avg_logprob),
                    "no_speech_prob": float(segment.no_speech_prob),
                    "compression_ratio": float(segment.compression_ratio),
                }
            )

        full_text = "\n".join(segment["text"] for segment in segments).strip()

        payload = {
            "audio_file": audio_path.name,
            "audio_path": str(audio_path),
            "model": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "segment_count": len(segments),
            "transcribed_at": datetime.utcnow().isoformat(),
            "segments": segments,
            "text": full_text,
        }

        output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_txt.write_text(full_text, encoding="utf-8")
        write_srt(segments, output_srt)
        write_vtt(segments, output_vtt)

        log(f"DONE: {audio_path.name} -> {len(segments)} segments")

        return {
            "audio_file": audio_path.name,
            "audio_path": str(audio_path),
            "transcript_json": str(output_json),
            "transcript_txt": str(output_txt),
            "subtitle_srt": str(output_srt),
            "subtitle_vtt": str(output_vtt),
            "status": "transcribed",
            "language": info.language,
            "duration_seconds": info.duration,
            "segment_count": len(segments),
            "error": "",
        }

    except Exception as exc:
        log(f"FAILED: {audio_path.name}")
        log(str(exc))
        traceback.print_exc()

        return {
            "audio_file": audio_path.name,
            "audio_path": str(audio_path),
            "transcript_json": "",
            "transcript_txt": "",
            "subtitle_srt": "",
            "subtitle_vtt": "",
            "status": "failed",
            "language": "",
            "duration_seconds": "",
            "segment_count": "",
            "error": str(exc),
        }


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "audio_file",
        "audio_path",
        "transcript_json",
        "transcript_txt",
        "subtitle_srt",
        "subtitle_vtt",
        "status",
        "language",
        "duration_seconds",
        "segment_count",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Report written to: {REPORT_PATH}")


def main() -> None:
    try:
        log("Starting Step 03: Faster-Whisper Transcription")

        audio_files = sorted(AUDIO_DIR.glob("*.wav"))

        if ONLY_FILE:
            only = AUDIO_DIR / ONLY_FILE
            audio_files = [only] if only.exists() else []

        if LIMIT is not None:
            audio_files = audio_files[:LIMIT]

        log(f"Python executable: {sys.executable}")
        log(f"Project root: {PROJECT_ROOT}")
        log(f"Audio directory: {AUDIO_DIR}")
        log(f"Transcript directory: {TRANSCRIPT_DIR}")
        log(f"Subtitle directory: {SUBTITLE_DIR}")
        log(f"Selected WAV files: {len(audio_files)}")
        log(f"Model: {MODEL_SIZE}")
        log(f"Device: {DEVICE}")
        log(f"Compute type: {COMPUTE_TYPE}")

        if not audio_files:
            log("No WAV files found. Run scripts/02_extract_audio.py first.")
            return

        log(f"First audio file: {audio_files[0].name}")
        log("Loading Faster-Whisper model...")

        model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )

        log("Model loaded.")

        rows = []

        for audio_path in audio_files:
            row = transcribe_audio(model, audio_path)
            rows.append(row)
            write_report(rows)

        completed_count = sum(
            1 for row in rows
            if row["status"] in ["transcribed", "skipped_exists"]
        )
        failed_count = len(rows) - completed_count

        log("Summary")
        log(f"Total selected audio files: {len(rows)}")
        log(f"Transcripts ready: {completed_count}")
        log(f"Failed: {failed_count}")

    except Exception:
        log("FATAL ERROR in transcription script")
        traceback.print_exc()


if __name__ == "__main__":
    main()
