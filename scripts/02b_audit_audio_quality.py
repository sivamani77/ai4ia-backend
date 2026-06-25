"""
Step 02b: Audio Quality Audit

Scans source MP4 videos (or pre-extracted WAV files) and computes per-file
audio quality metrics using ffprobe / ffmpeg volumedetect.

Outputs:
    data/reports/audio_quality_report.csv
    data/reports/audio_quality_report.json

Recommended action values:
    ok              - audio looks fine
    normalize       - low mean volume, but speech present
    denoise         - high noise floor or clipping
    retranscribe    - poor audio warrants re-running Whisper on cleaned audio
    manual_review   - extreme issues (no audio, very short, etc.)
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_CSV = REPORT_DIR / "audio_quality_report.csv"
REPORT_JSON = REPORT_DIR / "audio_quality_report.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
LOW_MEAN_VOLUME_DB = -32.0
WEAK_MAX_VOLUME_DB = -12.0
HIGH_SILENCE_RATIO = 0.35
LOW_SPEECH_RATIO = 0.30


def log(msg: str) -> None:
    print(msg, flush=True)


def run_ffprobe_format(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {"valid": False, "error": result.stderr.strip()[:500]}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": str(exc)}

    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    duration = None
    try:
        duration = float(fmt.get("duration", 0) or 0)
    except (ValueError, TypeError):
        duration = None

    return {
        "valid": True,
        "has_audio": bool(audio_streams),
        "duration_seconds": duration,
        "audio_codec": audio_streams[0].get("codec_name", "") if audio_streams else "",
        "audio_sample_rate": audio_streams[0].get("sample_rate", "") if audio_streams else "",
        "audio_channels": audio_streams[0].get("channels", 0) if audio_streams else 0,
        "error": "",
    }


def run_volumedetect(path: Path) -> dict:
    """
    Uses ffmpeg -af volumedetect to get mean and max volume.
    Works on both MP4 and WAV.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "/dev/null",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = result.stderr

    mean_volume = None
    max_volume = None

    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
    if m:
        mean_volume = float(m.group(1))

    m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
    if m:
        max_volume = float(m.group(1))

    return {"mean_volume_db": mean_volume, "max_volume_db": max_volume}


def run_silencedetect(path: Path, duration_seconds: float) -> dict:
    """
    Uses ffmpeg silencedetect to estimate silence ratio.
    Silence = below -40dB for >= 0.5 s.
    """
    if not duration_seconds or duration_seconds <= 0:
        return {"silence_ratio": None, "silence_seconds": None}

    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af", "silencedetect=noise=-40dB:d=0.5",
        "-vn", "-sn", "-dn",
        "-f", "null", "/dev/null",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = result.stderr

    silence_durations = re.findall(r"silence_duration:\s*([\d.]+)", stderr)
    total_silence = sum(float(d) for d in silence_durations)
    silence_ratio = round(min(1.0, total_silence / duration_seconds), 4)

    return {
        "silence_ratio": silence_ratio,
        "silence_seconds": round(total_silence, 2),
    }


def estimate_speech_ratio(mean_volume_db: float | None, silence_ratio: float | None) -> float | None:
    """
    Rough heuristic: speech-like ratio = non-silence frames with volume above -40 dB.
    Approximated from silence_ratio and mean_volume.
    """
    if mean_volume_db is None or silence_ratio is None:
        return None
    non_silence = 1.0 - silence_ratio
    if mean_volume_db < -45:
        return round(non_silence * 0.3, 4)
    if mean_volume_db < -35:
        return round(non_silence * 0.6, 4)
    return round(non_silence * 0.85, 4)


def detect_clipping(max_volume_db: float | None) -> bool:
    if max_volume_db is None:
        return False
    return max_volume_db >= -0.5


def decide_action(
    has_audio: bool,
    duration_seconds: float | None,
    mean_volume_db: float | None,
    max_volume_db: float | None,
    silence_ratio: float | None,
    speech_like_ratio: float | None,
    clipping_detected: bool,
) -> str:
    if not has_audio:
        return "manual_review"
    if duration_seconds is not None and duration_seconds < 10:
        return "manual_review"
    if mean_volume_db is None:
        return "manual_review"

    if clipping_detected:
        return "denoise"

    if silence_ratio is not None and silence_ratio > HIGH_SILENCE_RATIO:
        return "manual_review"

    if mean_volume_db < LOW_MEAN_VOLUME_DB:
        if speech_like_ratio is not None and speech_like_ratio < LOW_SPEECH_RATIO:
            return "retranscribe"
        return "normalize"

    if max_volume_db is not None and max_volume_db < WEAK_MAX_VOLUME_DB:
        return "normalize"

    return "ok"


def audit_file(path: Path) -> dict:
    log(f"AUDITING: {path.name}")

    probe = run_ffprobe_format(path)
    if not probe["valid"]:
        return {
            "video_id": path.stem,
            "audio_path": str(path),
            "duration_seconds": None,
            "mean_volume_db": None,
            "max_volume_db": None,
            "silence_ratio": None,
            "silence_seconds": None,
            "low_volume_ratio": None,
            "speech_like_ratio": None,
            "clipping_detected": False,
            "has_audio": False,
            "audio_codec": "",
            "recommended_action": "manual_review",
            "audit_error": probe.get("error", "ffprobe_failed"),
            "audited_at": datetime.utcnow().isoformat(),
        }

    if not probe["has_audio"]:
        return {
            "video_id": path.stem,
            "audio_path": str(path),
            "duration_seconds": probe["duration_seconds"],
            "mean_volume_db": None,
            "max_volume_db": None,
            "silence_ratio": None,
            "silence_seconds": None,
            "low_volume_ratio": None,
            "speech_like_ratio": None,
            "clipping_detected": False,
            "has_audio": False,
            "audio_codec": "",
            "recommended_action": "manual_review",
            "audit_error": "no_audio_stream",
            "audited_at": datetime.utcnow().isoformat(),
        }

    vol = run_volumedetect(path)
    sil = run_silencedetect(path, probe["duration_seconds"] or 0)

    mean_db = vol["mean_volume_db"]
    max_db = vol["max_volume_db"]
    silence_ratio = sil["silence_ratio"]
    speech_like_ratio = estimate_speech_ratio(mean_db, silence_ratio)
    clipping = detect_clipping(max_db)

    low_volume_ratio = None
    if mean_db is not None:
        low_volume_ratio = round(max(0.0, min(1.0, (-mean_db - 16) / 30)), 4)

    action = decide_action(
        has_audio=probe["has_audio"],
        duration_seconds=probe["duration_seconds"],
        mean_volume_db=mean_db,
        max_volume_db=max_db,
        silence_ratio=silence_ratio,
        speech_like_ratio=speech_like_ratio,
        clipping_detected=clipping,
    )

    return {
        "video_id": path.stem,
        "audio_path": str(path),
        "duration_seconds": probe["duration_seconds"],
        "mean_volume_db": mean_db,
        "max_volume_db": max_db,
        "silence_ratio": silence_ratio,
        "silence_seconds": sil.get("silence_seconds"),
        "low_volume_ratio": low_volume_ratio,
        "speech_like_ratio": speech_like_ratio,
        "clipping_detected": clipping,
        "has_audio": probe["has_audio"],
        "audio_codec": probe.get("audio_codec", ""),
        "recommended_action": action,
        "audit_error": "",
        "audited_at": datetime.utcnow().isoformat(),
    }


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "video_id", "audio_path", "duration_seconds",
        "mean_volume_db", "max_volume_db", "silence_ratio", "silence_seconds",
        "low_volume_ratio", "speech_like_ratio", "clipping_detected",
        "has_audio", "audio_codec", "recommended_action", "audit_error", "audited_at",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log(f"CSV report: {REPORT_CSV}")


def write_json(rows: list[dict]) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_files": len(rows),
        "flagged": sum(1 for r in rows if r["recommended_action"] != "ok"),
        "action_counts": {},
        "files": rows,
    }
    actions = [r["recommended_action"] for r in rows]
    for a in set(actions):
        payload["action_counts"][a] = actions.count(a)
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"JSON report: {REPORT_JSON}")


def collect_source_files(video_id: str | None = None) -> list[Path]:
    # Prefer pre-extracted WAV; fall back to source MP4
    files: list[Path] = []

    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    if wav_files:
        if video_id:
            wav_files = [f for f in wav_files if video_id in f.stem]
        files = wav_files
    else:
        mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
        if video_id:
            mp4_files = [f for f in mp4_files if video_id in f.stem]
        files = mp4_files

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit audio quality for all source videos")
    parser.add_argument("--video-id", help="Process only files matching this video ID substring")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files")
    parser.add_argument("--only-flagged", action="store_true", help="(ignored at audit stage — always audits all)")
    args = parser.parse_args()

    log("Starting Step 02b: Audio Quality Audit")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Video dir: {VIDEO_DIR}")
    log(f"Audio dir: {AUDIO_DIR}")

    files = collect_source_files(args.video_id)

    if args.limit:
        files = files[:args.limit]

    log(f"Files to audit: {len(files)}")

    if not files:
        log("No source files found. Run scripts/01_download_videos.py or scripts/02_extract_audio.py first.")
        return

    rows = []
    for path in files:
        row = audit_file(path)
        rows.append(row)

    write_csv(rows)
    write_json(rows)

    actions = [r["recommended_action"] for r in rows]
    log("")
    log("Summary")
    log(f"  Total audited : {len(rows)}")
    for action in ("ok", "normalize", "denoise", "retranscribe", "manual_review"):
        count = actions.count(action)
        if count:
            log(f"  {action:16s}: {count}")
    flagged = sum(1 for a in actions if a != "ok")
    log(f"  Flagged total : {flagged}")


if __name__ == "__main__":
    main()
