import csv
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "audio_extraction_report.csv"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_ffprobe(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        return {
            "valid": False,
            "duration_seconds": "",
            "has_audio": False,
            "has_video": False,
            "error": result.stderr.strip(),
        }

    data = json.loads(result.stdout)

    audio_streams = [
        stream for stream in data.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]

    video_streams = [
        stream for stream in data.get("streams", [])
        if stream.get("codec_type") == "video"
    ]

    return {
        "valid": True,
        "duration_seconds": data.get("format", {}).get("duration", ""),
        "has_audio": bool(audio_streams),
        "has_video": bool(video_streams),
        "error": "",
    }


def extract_audio(video_path: Path) -> dict:
    output_audio = AUDIO_DIR / f"{video_path.stem}.wav"

    probe = run_ffprobe(video_path)

    if not probe["valid"]:
        print(f"INVALID VIDEO: {video_path.name}")
        return {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "audio_file": "",
            "audio_path": "",
            "status": "failed_probe",
            **probe,
        }

    if not probe["has_audio"]:
        print(f"NO AUDIO: {video_path.name}")
        return {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "audio_file": "",
            "audio_path": "",
            "status": "no_audio_stream",
            **probe,
        }

    if output_audio.exists() and output_audio.stat().st_size > 0:
        print(f"SKIP: {output_audio.name}")
        return {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "audio_file": output_audio.name,
            "audio_path": str(output_audio),
            "status": "skipped_exists",
            **probe,
        }

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio),
    ]

    print(f"EXTRACTING: {video_path.name}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(f"FAILED: {video_path.name}")
        return {
            "video_file": video_path.name,
            "video_path": str(video_path),
            "audio_file": output_audio.name,
            "audio_path": str(output_audio),
            "status": "failed_extract",
            **probe,
            "error": result.stderr[-1000:],
        }

    print(f"DONE: {output_audio.name}")

    return {
        "video_file": video_path.name,
        "video_path": str(video_path),
        "audio_file": output_audio.name,
        "audio_path": str(output_audio),
        "status": "extracted",
        **probe,
    }


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "video_file",
        "video_path",
        "audio_file",
        "audio_path",
        "status",
        "valid",
        "duration_seconds",
        "has_audio",
        "has_video",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport written to: {REPORT_PATH}")


def main() -> None:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Found MP4 files: {len(videos)}\n")

    if not videos:
        print("No MP4 files found. Check data/raw/youtube/ai4ia_channel")
        return

    rows = []

    for video_path in videos:
        row = extract_audio(video_path)
        rows.append(row)

    write_report(rows)

    extracted_count = sum(1 for row in rows if row["status"] in ["extracted", "skipped_exists"])
    failed_count = len(rows) - extracted_count

    print("\nSummary")
    print(f"Total videos: {len(rows)}")
    print(f"Audio ready: {extracted_count}")
    print(f"Failed / skipped without audio: {failed_count}")


if __name__ == "__main__":
    main()