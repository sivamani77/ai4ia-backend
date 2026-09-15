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
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
META_DIR = PROJECT_ROOT / "data" / "metadata" / "youtube"
ARCHIVE_FILE = META_DIR / "download_archive.txt"

# yt-dlp needs a JS runtime >=20 to solve YouTube's JS challenge; without one,
# extraction silently reports many videos as "not available" even when they
# aren't. The system default `node` on PATH may be an older nvm version, so
# search nvm's installed versions for one new enough rather than assume PATH
# already points at it.
_MIN_NODE_MAJOR = 20


def _find_js_runtime_node() -> str | None:
    import shutil

    candidates: list[tuple[int, str]] = []

    path_node = shutil.which("node")
    if path_node:
        try:
            out = subprocess.run([path_node, "--version"], capture_output=True, text=True, timeout=10)
            match = re.match(r"v(\d+)", out.stdout.strip())
            if match:
                candidates.append((int(match.group(1)), path_node))
        except (OSError, subprocess.SubprocessError):
            pass

    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    if nvm_dir.is_dir():
        for version_dir in nvm_dir.iterdir():
            match = re.match(r"v(\d+)", version_dir.name)
            node_bin = version_dir / "bin" / "node"
            if match and node_bin.exists():
                candidates.append((int(match.group(1)), str(node_bin)))

    eligible = [c for c in candidates if c[0] >= _MIN_NODE_MAJOR]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c[0])[1]

# Channel "Videos" tab — yt-dlp expands this to every public upload.
DEFAULT_CHANNEL = "https://www.youtube.com/@ai4iaconference861/videos"

# Filename template reproduces the existing 0002_<id>_<title>.<ext> convention.
OUTPUT_TEMPLATE = str(RAW_DIR / "%(playlist_index)04d_%(id)s_%(title)s.%(ext)s")


def build_command(channel_url: str, limit: int | None, list_only: bool, cookies_file: str | None = None) -> list[str]:
    if list_only:
        # Print channel entries without downloading anything.
        cmd = ["yt-dlp"]
        if cookies_file:
            cmd += ["--cookies", cookies_file]
        cmd += ["--flat-playlist", "--print",
                "%(playlist_index)04d  %(id)s  %(title)s", channel_url]
        return cmd

    cmd = [
        "yt-dlp",
        # --- auth: without cookies, YouTube blocks cloud/datacenter IPs with
        # "Sign in to confirm you're not a bot" and every download fails ---
        *(["--cookies", cookies_file] if cookies_file else []),
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

    node_bin = _find_js_runtime_node()
    if node_bin:
        cmd += ["--js-runtimes", f"node:{node_bin}"]

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
    ap.add_argument("--cookies", default=os.environ.get("YOUTUBE_COOKIES_FILE"),
                    help="path to a Netscape-format cookies.txt for yt-dlp auth "
                         "(env: YOUTUBE_COOKIES_FILE). Required on cloud/datacenter "
                         "IPs to avoid YouTube's bot-check wall.")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    already = _archive_count()
    print(f"Channel : {args.url}")
    print(f"Archive : {ARCHIVE_FILE} ({already} already downloaded)")
    print(f"Output  : {RAW_DIR}")
    if args.cookies:
        if not Path(args.cookies).exists():
            print(f"WARNING: --cookies file not found: {args.cookies} — proceeding without it, "
                  f"downloads will likely fail with 'Sign in to confirm you're not a bot'.")
            args.cookies = None
        else:
            print(f"Cookies : {args.cookies}")
    print(flush=True)

    cmd = build_command(args.url, args.limit or None, args.list_only, args.cookies)
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
