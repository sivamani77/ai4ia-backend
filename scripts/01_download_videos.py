#!/usr/bin/env python3
"""01 — Download AI4IA conference videos from the YouTube channel.

Resumable: uses yt-dlp's --download-archive so already-downloaded videos are
skipped automatically. Re-running this picks up only NEW videos on the channel.

Output layout (matches the existing data/raw/youtube/ai4ia_channel/ exactly):
    {playlist_index:04d}_{video_id}_{title}.mp4
    ...same stem.info.json   (metadata)
    ...same stem.jpg         (thumbnail)
    ...same stem.en.vtt / .en-orig.vtt   (subtitles, when available)

The download archive lives at:
    data/metadata/youtube/download_archive.txt

Usage:
    python scripts/01_download_videos.py                 # download all new videos
    python scripts/01_download_videos.py --limit 5       # only next 5 new ones
    python scripts/01_download_videos.py --list           # list channel without downloading
    python scripts/01_download_videos.py --url <channel>  # override channel URL
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
META_DIR = PROJECT_ROOT / "data" / "metadata" / "youtube"
ARCHIVE_FILE = META_DIR / "download_archive.txt"

# Channel "Videos" tab — yt-dlp expands this to every public upload.
DEFAULT_CHANNEL = "https://www.youtube.com/@ai4iaconference861/videos"

# Filename template reproduces the existing 0002_<id>_<title>.<ext> convention.
OUTPUT_TEMPLATE = str(RAW_DIR / "%(playlist_index)04d_%(id)s_%(title)s.%(ext)s")


def build_command(channel_url: str, limit: int | None, list_only: bool) -> list[str]:
    if list_only:
        # Print channel entries without downloading anything.
        return ["yt-dlp", "--flat-playlist", "--print",
                "%(playlist_index)04d  %(id)s  %(title)s", channel_url]

    cmd = [
        "yt-dlp",
        # --- resume / dedupe ---
        "--download-archive", str(ARCHIVE_FILE),
        # --- format: best mp4 video+audio, merged ---
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        # --- companion files (match existing dataset) ---
        "--write-info-json",
        "--write-thumbnail",
        "--convert-thumbnails", "jpg",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        # --- robustness ---
        "--ignore-errors",          # skip a single failed video, keep going
        "--no-overwrites",
        "--continue",               # resume partial downloads
        "--retries", "10",
        "--output", OUTPUT_TEMPLATE,
    ]

    if limit:
        cmd += ["--max-downloads", str(limit)]

    cmd.append(channel_url)
    return cmd


def _archive_count() -> int:
    if not ARCHIVE_FILE.exists():
        return 0
    return sum(1 for line in ARCHIVE_FILE.read_text().splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_CHANNEL, help="channel/playlist URL")
    ap.add_argument("--limit", type=int, default=0, help="max NEW videos to fetch (0 = all)")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="list channel videos without downloading")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    already = _archive_count()
    print(f"Channel : {args.url}")
    print(f"Archive : {ARCHIVE_FILE} ({already} already downloaded)")
    print(f"Output  : {RAW_DIR}\n", flush=True)

    cmd = build_command(args.url, args.limit or None, args.list_only)
    print("Running:", " ".join(cmd), "\n", flush=True)

    # --max-downloads exits with code 101 once the cap is hit; treat that as success.
    result = subprocess.run(cmd)
    if result.returncode not in (0, 101):
        print(f"\nyt-dlp exited with code {result.returncode}", flush=True)
        return result.returncode

    if not args.list_only:
        mp4s = len(list(RAW_DIR.glob("*.mp4")))
        now = _archive_count()
        print(f"\nDone. {mp4s} mp4 files on disk; "
              f"{now} entries in archive ({now - already} new this run).", flush=True)

        if now > already:
            print("\nUpdating source-video duration index for the new videos...", flush=True)
            index_script = Path(__file__).resolve().parent / "build_source_duration_index.py"
            subprocess.run([sys.executable, str(index_script)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
