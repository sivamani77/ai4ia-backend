"""
Step 02c: Clean Audio

Takes source audio (WAV or MP4) and applies FFmpeg filters to produce a
cleaned WAV suitable for better Whisper transcription.

Two cleaning profiles:
    voice_norm_denoise  - highpass + lowpass + afftdn noise reduction + loudnorm
    loudnorm_only       - loudnorm only (lighter, safer fallback)

Input  : data/audio/*.wav  OR  data/raw/youtube/ai4ia_channel/*.mp4
Output : data/audio_cleaned/{stem}.clean.wav
Meta   : data/audio_cleaned/{stem}.clean.json

Never overwrites original audio files.

Usage:
    .venv/bin/python scripts/02c_clean_audio.py
    .venv/bin/python scripts/02c_clean_audio.py --only-flagged
    .venv/bin/python scripts/02c_clean_audio.py --video-id 0026 --profile loudnorm_only
    .venv/bin/python scripts/02c_clean_audio.py --limit 5 --dry-run
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
CLEAN_DIR = PROJECT_ROOT / "data" / "audio_cleaned"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

AUDIT_JSON = REPORT_DIR / "audio_quality_report.json"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = {
    "voice_norm_denoise": "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
    "loudnorm_only": "loudnorm=I=-16:TP=-1.5:LRA=11",
}

FLAGS_REQUIRING_RETRANSCRIBE = {"normalize", "denoise", "retranscribe"}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_flagged_video_ids() -> set[str]:
    if not AUDIT_JSON.exists():
        return set()
    try:
        data = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        return {
            f["video_id"]
            for f in data.get("files", [])
            if f.get("recommended_action") in FLAGS_REQUIRING_RETRANSCRIBE
        }
    except Exception as exc:
        log(f"WARNING: Could not load audit report: {exc}")
        return set()


def collect_source_files(video_id: str | None, only_flagged: bool) -> list[Path]:
    flagged_ids: set[str] = set()
    if only_flagged:
        flagged_ids = load_flagged_video_ids()
        if not flagged_ids:
            log("No flagged video IDs found in audit report. Run 02b_audit_audio_quality.py first.")

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if wav_files:
        sources = wav_files
    else:
        sources = sorted(VIDEO_DIR.glob("*.mp4"))

    if video_id:
        sources = [f for f in sources if video_id in f.stem]

    if only_flagged and flagged_ids:
        sources = [f for f in sources if any(fid in f.stem for fid in flagged_ids)]

    return sources


def already_cleaned(stem: str) -> bool:
    out = CLEAN_DIR / f"{stem}.clean.wav"
    return out.exists() and out.stat().st_size > 0


def clean_audio(source: Path, profile: str, dry_run: bool) -> dict:
    stem = source.stem
    out_wav = CLEAN_DIR / f"{stem}.clean.wav"
    out_meta = CLEAN_DIR / f"{stem}.clean.json"

    if already_cleaned(stem):
        log(f"SKIP (exists): {out_wav.name}")
        return {"video_id": stem, "status": "skipped_exists", "clean_audio": str(out_wav)}

    af_filter = PROFILES.get(profile, PROFILES["voice_norm_denoise"])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", af_filter,
        str(out_wav),
    ]

    cmd_str = " ".join(cmd)
    log(f"{'DRY-RUN' if dry_run else 'CLEANING'}: {source.name} -> {out_wav.name}")
    log(f"  Profile: {profile}")

    if dry_run:
        return {
            "video_id": stem,
            "status": "dry_run",
            "source_audio": str(source),
            "clean_audio": str(out_wav),
            "cleaning_profile": profile,
            "ffmpeg_command": cmd_str,
        }

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        log(f"FAILED: {source.name}")
        log(result.stderr[-500:])
        return {
            "video_id": stem,
            "status": "failed",
            "source_audio": str(source),
            "clean_audio": "",
            "cleaning_profile": profile,
            "ffmpeg_command": cmd_str,
            "error": result.stderr[-500:],
        }

    meta = {
        "video_id": stem,
        "source_audio": str(source),
        "clean_audio": str(out_wav),
        "cleaning_profile": profile,
        "af_filter": af_filter,
        "created_at": datetime.utcnow().isoformat(),
        "ffmpeg_command": cmd_str,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"DONE: {out_wav.name}")
    return {**meta, "status": "cleaned"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean audio for better transcription")
    parser.add_argument("--only-flagged", action="store_true",
                        help="Only process files flagged in audio quality report")
    parser.add_argument("--video-id", help="Process only files matching this video ID substring")
    parser.add_argument("--profile", choices=list(PROFILES.keys()),
                        default="voice_norm_denoise", help="FFmpeg filter profile")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files")
    parser.add_argument("--force", action="store_true", help="Re-clean even if output exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    log("Starting Step 02c: Clean Audio")
    log(f"Profile     : {args.profile}")
    log(f"Filter      : {PROFILES[args.profile]}")
    log(f"Only flagged: {args.only_flagged}")
    log(f"Dry run     : {args.dry_run}")

    sources = collect_source_files(args.video_id, args.only_flagged)

    if args.force:
        pass  # will overwrite via ffmpeg -y
    else:
        sources = [s for s in sources if not already_cleaned(s.stem)]

    if args.limit:
        sources = sources[:args.limit]

    log(f"Files to process: {len(sources)}")

    if not sources:
        log("Nothing to clean.")
        return

    results = []
    for source in sources:
        row = clean_audio(source, args.profile, args.dry_run)
        results.append(row)

    cleaned = sum(1 for r in results if r["status"] == "cleaned")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    failed = sum(1 for r in results if r["status"] == "failed")
    dry = sum(1 for r in results if r["status"] == "dry_run")

    log("")
    log("Summary")
    log(f"  Cleaned : {cleaned}")
    log(f"  Skipped : {skipped}")
    log(f"  Failed  : {failed}")
    if dry:
        log(f"  Dry-run : {dry}")

    report_path = REPORT_DIR / "audio_cleaning_report.json"
    report_path.write_text(
        json.dumps({
            "generated_at": datetime.utcnow().isoformat(),
            "profile": args.profile,
            "total": len(results),
            "cleaned": cleaned,
            "skipped": skipped,
            "failed": failed,
            "files": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Report: {report_path}")


if __name__ == "__main__":
    main()
