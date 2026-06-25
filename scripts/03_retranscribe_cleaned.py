"""
Step 03 (Retranscribe): Faster-Whisper on Cleaned Audio

Transcribes cleaned audio from data/audio_cleaned/ using Faster-Whisper.
Saves results to data/transcripts_cleaned/ and never touches data/transcripts/.

Optionally runs WhisperX word-level alignment if whisperx is installed.
Aligned output goes to data/transcripts_cleaned_aligned/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_AUDIO_DIR = PROJECT_ROOT / "data" / "audio_cleaned"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts_cleaned"
ALIGNED_DIR = PROJECT_ROOT / "data" / "transcripts_cleaned_aligned"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
TEMP_CHUNK_DIR = PROJECT_ROOT / "temp" / "transcription_chunks"

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = REPORT_DIR / "transcription_quality_report.csv"
AUDIT_JSON = REPORT_DIR / "audio_quality_report.json"

DEFAULT_MODEL = "small"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_MAX_DURATION_SECONDS = 3600
DEFAULT_CHUNK_SECONDS = 900

FLAGS_REQUIRING_RETRANSCRIBE = {"normalize", "denoise", "retranscribe"}


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "unknown"


def format_timestamp(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp(seg["start"]).replace(",", ".")
        end = format_timestamp(seg["end"]).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"].strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def compute_transcript_quality(segments: list[dict], duration_seconds: float) -> dict:
    if not segments:
        return {
            "segment_count": 0,
            "avg_confidence": 0.0,
            "avg_words_per_segment": 0.0,
            "empty_segment_ratio": 1.0,
            "repeated_phrase_ratio": 0.0,
            "low_confidence_ratio": 0.0,
            "recommended_action": "manual_review",
        }

    texts = [s.get("text", "").strip() for s in segments]
    empty_count = sum(1 for t in texts if not t)
    empty_ratio = round(empty_count / len(segments), 4)

    confidences = [
        1.0 + s.get("avg_logprob", -1.0)
        for s in segments
        if s.get("avg_logprob") is not None
    ]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < 0.6)
    low_conf_ratio = round(low_conf_count / len(confidences), 4) if confidences else 0.0

    word_counts = [len(re.findall(r"\b\w+\b", t)) for t in texts if t]
    avg_wps = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0.0

    phrase_counts = Counter(texts)
    repeated = sum(cnt - 1 for cnt in phrase_counts.values() if cnt > 1)
    repeated_ratio = round(repeated / len(texts), 4) if texts else 0.0

    action = "ok"
    if empty_ratio > 0.3 or avg_conf < 0.4:
        action = "manual_review"
    elif low_conf_ratio > 0.3 or repeated_ratio > 0.15:
        action = "rerun_large_model"
    elif avg_conf < 0.65:
        action = "rerun_large_model"

    return {
        "segment_count": len(segments),
        "avg_confidence": avg_conf,
        "avg_words_per_segment": avg_wps,
        "empty_segment_ratio": empty_ratio,
        "repeated_phrase_ratio": repeated_ratio,
        "low_confidence_ratio": low_conf_ratio,
        "recommended_action": action,
    }


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[:500]}")

    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError("ffprobe returned empty duration")

    try:
        duration = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid duration from ffprobe: {raw}") from exc

    if duration <= 0:
        raise RuntimeError(f"invalid non-positive duration: {duration}")
    return duration


def try_whisperx_align(audio_path: Path, segments: list[dict], language: str, device: str) -> list[dict] | None:
    try:
        import torch  # noqa: F401
        import whisperx  # type: ignore
    except ImportError:
        return None

    try:
        align_device = "cuda" if device == "cuda" else "cpu"
        model_a, metadata = whisperx.load_align_model(language_code=language, device=align_device)
        audio = whisperx.load_audio(str(audio_path))
        result = whisperx.align(
            [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments],
            model_a,
            metadata,
            audio,
            align_device,
            return_char_alignments=False,
        )
        return result.get("segments", [])
    except Exception as exc:
        log(f"  WhisperX alignment failed: {exc}")
        return None


def extract_segments(model, audio_path: Path) -> tuple[list[dict], object]:
    segments_gen, info = model.transcribe(
        str(audio_path),
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segments = []
    for seg in segments_gen:
        seg_dict = {
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "avg_logprob": float(seg.avg_logprob),
            "no_speech_prob": float(seg.no_speech_prob),
            "compression_ratio": float(seg.compression_ratio),
        }
        if hasattr(seg, "words") and seg.words:
            seg_dict["words"] = [
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(w.probability),
                }
                for w in seg.words
            ]
        segments.append(seg_dict)

    return segments, info


def shifted_segment(segment: dict, chunk_start: float) -> dict:
    adjusted = dict(segment)
    adjusted["start"] = float(segment["start"]) + chunk_start
    adjusted["end"] = float(segment["end"]) + chunk_start
    if segment.get("words"):
        adjusted["words"] = [
            {
                **word,
                "start": float(word["start"]) + chunk_start,
                "end": float(word["end"]) + chunk_start,
            }
            for word in segment["words"]
        ]
    return adjusted


def build_transcript_payload(
    *,
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    duration_seconds: float,
    segments: list[dict],
    language: str,
    language_probability: float | None,
    mode: str,
    chunk_seconds: int | None,
) -> dict:
    quality = compute_transcript_quality(segments, duration_seconds)
    full_text = "\n".join(s["text"] for s in segments).strip()
    payload = {
        "audio_file": audio_path.name,
        "source_audio_path": str(audio_path),
        "source_transcript_version": "cleaned_v2",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": language,
        "language_probability": language_probability,
        "duration_seconds": duration_seconds,
        "transcribed_at": datetime.utcnow().isoformat(),
        "transcription_mode": mode,
        "chunk_seconds": chunk_seconds,
        "quality": quality,
        "segments": segments,
        "text": full_text,
    }
    return payload


def write_transcript_outputs(base_path: Path, payload: dict) -> None:
    segments = payload["segments"]
    text = payload["text"]
    base_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    base_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    write_srt(segments, base_path.with_suffix(".srt"))
    write_vtt(segments, base_path.with_suffix(".vtt"))


def load_partial_payload(partial_path: Path) -> dict | None:
    if not partial_path.exists():
        return None
    try:
        return json.loads(partial_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_partial_payload(partial_path: Path, payload: dict) -> None:
    partial_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_chunk_file(audio_path: Path, chunk_path: Path, start_seconds: float, chunk_seconds: int) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(chunk_seconds),
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(chunk_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunking failed: {result.stderr.strip()[:500]}")


def maybe_align_transcript(audio_path: Path, payload: dict, align: bool, device: str, aligned_path: Path) -> None:
    if not align:
        return

    log("  Aligning with WhisperX...")
    aligned = try_whisperx_align(audio_path, payload["segments"], payload["language"], device)
    if aligned:
        aligned_payload = {
            "audio_file": audio_path.name,
            "source_transcript_version": "cleaned_v2_aligned",
            "language": payload["language"],
            "aligned_at": datetime.utcnow().isoformat(),
            "segments": aligned,
        }
        aligned_path.write_text(json.dumps(aligned_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Aligned transcript: {aligned_path.name}")
    else:
        log("  WhisperX not available or alignment failed — skipping alignment")


def result_row(
    *,
    audio_file: str,
    video_id: str,
    status: str,
    transcript_json: str = "",
    quality: dict | None = None,
    error: str = "",
) -> dict:
    quality = quality or {}
    return {
        "audio_file": audio_file,
        "video_id": video_id,
        "source": "cleaned_audio",
        "status": status,
        "transcript_json": transcript_json,
        "segment_count": quality.get("segment_count", ""),
        "avg_confidence": quality.get("avg_confidence", ""),
        "avg_words_per_segment": quality.get("avg_words_per_segment", ""),
        "empty_segment_ratio": quality.get("empty_segment_ratio", ""),
        "repeated_phrase_ratio": quality.get("repeated_phrase_ratio", ""),
        "low_confidence_ratio": quality.get("low_confidence_ratio", ""),
        "recommended_action": quality.get("recommended_action", ""),
        "error": error,
    }


def transcribe_normal(
    model,
    audio_path: Path,
    duration_seconds: float,
    args: argparse.Namespace,
) -> dict:
    stem = audio_path.stem.replace(".clean", "")
    out_json = TRANSCRIPT_DIR / f"{stem}.json"
    aligned_path = ALIGNED_DIR / f"{stem}.json"

    if not args.force and out_json.exists() and out_json.stat().st_size > 0:
        return result_row(
            audio_file=audio_path.name,
            video_id=stem,
            status="skipped_existing",
            transcript_json=str(out_json),
        )

    segments, info = extract_segments(model, audio_path)
    payload = build_transcript_payload(
        audio_path=audio_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        duration_seconds=duration_seconds,
        segments=segments,
        language=info.language,
        language_probability=getattr(info, "language_probability", None),
        mode="normal",
        chunk_seconds=None,
    )
    write_transcript_outputs(out_json, payload)
    maybe_align_transcript(audio_path, payload, not args.no_align, args.device, aligned_path)
    return result_row(
        audio_file=audio_path.name,
        video_id=stem,
        status="transcribed",
        transcript_json=str(out_json),
        quality=payload["quality"],
    )


def transcribe_chunked(
    model,
    audio_path: Path,
    duration_seconds: float,
    args: argparse.Namespace,
) -> dict:
    stem = audio_path.stem.replace(".clean", "")
    out_json = TRANSCRIPT_DIR / f"{stem}.json"
    partial_json = TRANSCRIPT_DIR / f"{stem}.partial.json"
    aligned_path = ALIGNED_DIR / f"{stem}.json"
    chunk_dir = TEMP_CHUNK_DIR / safe_slug(stem)

    if not args.force and out_json.exists() and out_json.stat().st_size > 0:
        return result_row(
            audio_file=audio_path.name,
            video_id=stem,
            status="skipped_existing",
            transcript_json=str(out_json),
        )

    total_chunks = int(math.ceil(duration_seconds / args.chunk_seconds))
    partial = load_partial_payload(partial_json) if args.resume else None
    completed_chunks = set()
    merged_segments: list[dict] = []
    language = "en"
    language_probability = None

    if partial:
        completed_chunks = {
            int(idx)
            for idx in partial.get("completed_chunks", [])
            if isinstance(idx, int) or str(idx).isdigit()
        }
        merged_segments = partial.get("segments", [])
        language = partial.get("language", "en")
        language_probability = partial.get("language_probability")
        log(f"  Resume enabled: {len(completed_chunks)}/{total_chunks} chunks already complete")
    elif partial_json.exists() and not args.resume:
        partial_json.unlink()

    chunk_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index in range(total_chunks):
        chunk_start = float(chunk_index * args.chunk_seconds)
        chunk_path = chunk_dir / f"chunk_{chunk_index:03d}.wav"
        if chunk_index in completed_chunks:
            log(f"  Chunk {chunk_index + 1}/{total_chunks}: already complete, resuming")
            continue

        log(f"  Chunk {chunk_index + 1}/{total_chunks}: start={chunk_start:.2f}s")
        create_chunk_file(audio_path, chunk_path, chunk_start, args.chunk_seconds)

        chunk_segments, info = extract_segments(model, chunk_path)
        language = getattr(info, "language", language)
        language_probability = getattr(info, "language_probability", language_probability)

        shifted = [shifted_segment(seg, chunk_start) for seg in chunk_segments]
        merged_segments.extend(shifted)
        merged_segments.sort(key=lambda seg: (seg["start"], seg["end"]))
        completed_chunks.add(chunk_index)

        partial_payload = build_transcript_payload(
            audio_path=audio_path,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            duration_seconds=duration_seconds,
            segments=merged_segments,
            language=language,
            language_probability=language_probability,
            mode="chunked_partial",
            chunk_seconds=args.chunk_seconds,
        )
        partial_payload["completed_chunks"] = sorted(completed_chunks)
        partial_payload["total_chunks"] = total_chunks
        save_partial_payload(partial_json, partial_payload)

        if not args.keep_chunks and chunk_path.exists():
            chunk_path.unlink()

    payload = build_transcript_payload(
        audio_path=audio_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        duration_seconds=duration_seconds,
        segments=merged_segments,
        language=language,
        language_probability=language_probability,
        mode="chunked",
        chunk_seconds=args.chunk_seconds,
    )
    payload["completed_chunks"] = sorted(completed_chunks)
    payload["total_chunks"] = total_chunks

    write_transcript_outputs(out_json, payload)
    maybe_align_transcript(audio_path, payload, not args.no_align, args.device, aligned_path)

    if partial_json.exists():
        partial_json.unlink()
    if not args.keep_chunks and chunk_dir.exists():
        shutil.rmtree(chunk_dir, ignore_errors=True)

    return result_row(
        audio_file=audio_path.name,
        video_id=stem,
        status="transcribed_chunked",
        transcript_json=str(out_json),
        quality=payload["quality"],
    )


def load_flagged_ids() -> set[str]:
    if not AUDIT_JSON.exists():
        return set()
    try:
        data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        return {
            f["video_id"]
            for f in data.get("files", [])
            if f.get("recommended_action") in FLAGS_REQUIRING_RETRANSCRIBE
        }
    except Exception:
        return set()


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "audio_file",
        "video_id",
        "source",
        "status",
        "transcript_json",
        "segment_count",
        "avg_confidence",
        "avg_words_per_segment",
        "empty_segment_ratio",
        "repeated_phrase_ratio",
        "low_confidence_ratio",
        "recommended_action",
        "error",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log(f"Report: {REPORT_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retranscribe cleaned audio with Faster-Whisper")
    parser.add_argument(
        "--only-flagged",
        action="store_true",
        help="Only process videos flagged in the audio quality report",
    )
    parser.add_argument("--video-id", help="Process only files matching this video ID substring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument(
        "--compute-type",
        choices=["int8", "float16", "float32"],
        default=DEFAULT_COMPUTE_TYPE,
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=int,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--chunk-long", action="store_true", default=True)
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--no-align", action="store_true", help="Skip WhisperX alignment step")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if output exists")

    args = parser.parse_args()
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be > 0")
    if args.max_duration_seconds <= 0:
        parser.error("--max-duration-seconds must be > 0")
    return args


def main() -> None:
    args = parse_args()

    log("Starting Step 03 (Retranscribe): Faster-Whisper on Cleaned Audio")
    log(f"Clean audio dir       : {CLEAN_AUDIO_DIR}")
    log(f"Transcript dir        : {TRANSCRIPT_DIR}")
    log(f"Aligned dir           : {ALIGNED_DIR}")
    log(f"Model                 : {args.model}")
    log(f"Device                : {args.device}")
    log(f"Compute type          : {args.compute_type}")
    log(f"Only flagged          : {args.only_flagged}")
    log(f"Align                 : {not args.no_align}")
    log(f"Max duration seconds  : {args.max_duration_seconds}")
    log(f"Skip long             : {args.skip_long}")
    log(f"Chunk long            : {args.chunk_long}")
    log(f"Chunk seconds         : {args.chunk_seconds}")
    log(f"Continue on error     : {args.continue_on_error}")
    log(f"Resume                : {args.resume}")
    log(f"Keep chunks           : {args.keep_chunks}")

    all_audio_files = sorted(CLEAN_AUDIO_DIR.glob("*.clean.wav"))
    log(f"Found cleaned audio files: {len(all_audio_files)}")

    audio_files = all_audio_files
    if args.video_id:
        audio_files = [f for f in audio_files if args.video_id in f.stem]

    if args.only_flagged:
        flagged = load_flagged_ids()
        if flagged:
            audio_files = [f for f in audio_files if any(fid in f.stem for fid in flagged)]
        else:
            log("No flagged IDs found. Run 02b_audit_audio_quality.py and 02c_clean_audio.py first.")

    if args.limit:
        audio_files = audio_files[:args.limit]

    log(f"After filtering: {len(audio_files)}")
    log(f"Python           : {sys.executable}")

    if not audio_files:
        log("No cleaned audio files found. Run scripts/02c_clean_audio.py first.")
        return

    log("Loading Faster-Whisper model...")
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    log("Model loaded.")

    rows: list[dict] = []
    total = len(audio_files)

    for index, audio_path in enumerate(audio_files, start=1):
        stem = audio_path.stem.replace(".clean", "")
        log(f"Processing {index}/{total}: {audio_path.name}")

        try:
            duration_seconds = probe_duration_seconds(audio_path)
            log(f"Duration: {duration_seconds:.2f} seconds")

            if duration_seconds > args.max_duration_seconds:
                if args.skip_long:
                    log("Mode: skipped_long")
                    row = result_row(
                        audio_file=audio_path.name,
                        video_id=stem,
                        status="skipped_long",
                        error=(
                            f"duration {duration_seconds:.2f}s exceeds "
                            f"max {args.max_duration_seconds}s"
                        ),
                    )
                elif args.chunk_long:
                    log("Mode: chunked")
                    row = transcribe_chunked(model, audio_path, duration_seconds, args)
                else:
                    log("Mode: normal")
                    row = transcribe_normal(model, audio_path, duration_seconds, args)
            else:
                log("Mode: normal")
                row = transcribe_normal(model, audio_path, duration_seconds, args)

            log(f"Status: {row['status']}")
        except Exception as exc:
            log(f"Status: failed")
            traceback.print_exc()
            row = result_row(
                audio_file=audio_path.name,
                video_id=stem,
                status="failed",
                error=str(exc),
            )
            if not args.continue_on_error:
                rows.append(row)
                write_report(rows)
                raise

        rows.append(row)
        write_report(rows)

    done = sum(1 for r in rows if r["status"] == "transcribed")
    chunked = sum(1 for r in rows if r["status"] == "transcribed_chunked")
    skipped_existing = sum(1 for r in rows if r["status"] == "skipped_existing")
    skipped_long = sum(1 for r in rows if r["status"] == "skipped_long")
    failed = sum(1 for r in rows if r["status"] == "failed")

    log("")
    log("Summary")
    log(f"  Transcribed         : {done}")
    log(f"  Transcribed chunked : {chunked}")
    log(f"  Skipped existing    : {skipped_existing}")
    log(f"  Skipped long        : {skipped_long}")
    log(f"  Failed              : {failed}")


if __name__ == "__main__":
    main()
