"""Build/update the source-video duration index.

Stages 07 and 07b (storyline planning + validation) need to know each source
video's REAL video-stream duration to catch corrupted/truncated recordings
(footage that cuts out early while the audio track keeps going) before
committing to a plan. They don't download full video files — only stage 09
does that — so this script probes each video once, here, and writes the
result to a small sidecar file that 07/07b download instead.

Usage:
    python scripts/build_source_duration_index.py            # probe new/changed videos only
    python scripts/build_source_duration_index.py --full      # re-probe every video
    python scripts/build_source_duration_index.py --no-upload # skip the S3 push

Normally run automatically at the end of scripts/01_download_videos.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    RAW_VIDEO_DIR,
    SOURCE_DURATION_INDEX_S3_KEY,
)

INDEX_PATH = PROJECT_ROOT / "data" / "metadata" / "source_video_durations.json"


def probe_video_stream_duration(path: Path) -> float:
    """ffprobe the VIDEO stream duration (not container/audio) — same call used
    at plan- and render-time so the numbers agree everywhere."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def load_index() -> dict[str, dict]:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_index(index: dict[str, dict]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def build_index(full: bool) -> dict[str, dict]:
    index = {} if full else load_index()
    videos = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {RAW_VIDEO_DIR}")
        return index

    probed = skipped = 0
    for video_path in videos:
        key = video_path.stem
        stat = video_path.stat()
        cached = index.get(key)
        # Skip re-probing unchanged files unless --full was requested.
        if cached and cached.get("size_bytes") == stat.st_size and cached.get("video_stream_seconds", 0) > 0:
            skipped += 1
            continue

        duration = probe_video_stream_duration(video_path)
        index[key] = {
            "video_stream_seconds": round(duration, 3),
            "size_bytes": stat.st_size,
            "filename": video_path.name,
        }
        status = "OK" if duration > 0 else "FAILED TO PROBE"
        print(f"  [{status}] {key}: {duration:.1f}s")
        probed += 1

    print(f"\nProbed {probed} video(s), skipped {skipped} unchanged. Total indexed: {len(index)}")
    return index


def upload_index() -> bool:
    try:
        from app.pipeline.s3_helpers import use_s3, upload
    except ImportError:
        return False
    if not use_s3():
        print("S3 not configured (use_s3() is False) — skipping upload.")
        return False
    upload(INDEX_PATH, SOURCE_DURATION_INDEX_S3_KEY)
    print(f"Uploaded index to s3://.../{SOURCE_DURATION_INDEX_S3_KEY}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Re-probe every video, ignoring the existing index")
    parser.add_argument("--no-upload", action="store_true", help="Skip pushing the index to S3")
    args = parser.parse_args()

    print(f"Source video dir: {RAW_VIDEO_DIR}")
    print(f"Index file:       {INDEX_PATH}\n")

    index = build_index(full=args.full)
    save_index(index)
    print(f"Saved: {INDEX_PATH}")

    if not args.no_upload:
        upload_index()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
