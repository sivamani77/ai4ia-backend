#!/usr/bin/env python3
"""Find and repair source videos whose VIDEO track is shorter than their AUDIO track.

A truncated download produces an MP4 where the picture stops early while the
sound runs to the end. Nothing upstream notices: stage 02 extracts the audio,
03 transcribes it and 04 cuts fragments on transcript timestamps, so every
fragment sits on the AUDIO timeline. Fragments past the end of the video track
then have real audio and no picture, and stage 09 falls back to rendering a
black frame under the speaker's voice.

Found in the wild on 2026-09-21: `0082_5aoew3DYz7Y` had a 295.2s video track
and a 5249.3s audio track (YouTube's own metadata says the real video is
5249s), i.e. 5 minutes of picture for 87 minutes of sound. 1 bad file of 309.

Why it never self-healed: scripts/01_download_videos.py passes --download-archive
together with --no-overwrites, so once a video id is recorded in
download_archive.txt yt-dlp skips it forever. A bad download is permanent until
something explicitly re-fetches it. That is what this script does.

IMPORTANT — this script does NOT delete transcripts, fragments or embeddings.
They are derived from the audio track, which is intact, so they are correct and
become fully usable again the moment the video track is restored. Deleting them
would discard good data. Only if a re-download is ALSO short (i.e. the source
itself really is video-truncated) is there anything to prune, and this script
reports that case rather than acting on it.

Usage (read-only by default, safe to run anywhere):
    python scripts/repair_truncated_videos.py                  # scan + report
    python scripts/repair_truncated_videos.py --repair         # re-download bad ones
    python scripts/repair_truncated_videos.py --repair --push  # ... and upload to S3
    python scripts/repair_truncated_videos.py --only 5aoew3DYz7Y --repair --push

In a deployed pod the raw videos usually are not on local disk; pass --pull to
fetch each candidate from S3 first, or run with --only to repair one id.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.timeline_budget import (  # noqa: E402
    RAW_VIDEO_DIR,
    SOURCE_DURATION_INDEX_PATH,
    SOURCE_DURATION_INDEX_S3_KEY,
)

S3_RAW_PREFIX = "raw/youtube/ai4ia_channel/"

# A video track may legitimately end a beat before the audio (last frame vs last
# audio packet, container rounding). Only a gap larger than this is a real
# truncation rather than muxing noise. The observed real case was 4954s.
TRUNCATION_TOLERANCE_SECONDS = 5.0

# yt-dlp format string — must stay identical to scripts/01_download_videos.py,
# or a repaired file differs in codec/resolution from the rest of the archive.
YTDLP_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def _probe(path: Path, stream: str) -> float:
    """Duration of one stream ('v:0' / 'a:0') in seconds. 0.0 when unknown."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", stream,
                "-show_entries", "stream=duration",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return float(result.stdout.strip() or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def _video_id_of(path: Path) -> str:
    """`0082_5aoew3DYz7Y_Some Title` -> `5aoew3DYz7Y`. '' when unparseable."""
    parts = path.stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def scan(paths: list[Path]) -> list[dict]:
    """Probe every file; return one record per TRUNCATED video."""
    bad: list[dict] = []
    for i, path in enumerate(sorted(paths), 1):
        video = _probe(path, "v:0")
        audio = _probe(path, "a:0")
        gap = audio - video
        if video > 0 and audio > 0 and gap > TRUNCATION_TOLERANCE_SECONDS:
            bad.append({
                "path": path,
                "video_id": _video_id_of(path),
                "video_seconds": round(video, 1),
                "audio_seconds": round(audio, 1),
                "missing_seconds": round(gap, 1),
            })
        if i % 50 == 0:
            print(f"  ... probed {i}/{len(paths)}", flush=True)
    return bad


def redownload(video_id: str, dest_dir: Path, cookies: str | None) -> Path | None:
    """Fetch a fresh copy into dest_dir. Returns the new file, or None on failure.

    Downloads to a scratch name rather than over the existing file: the current
    copy, broken as it is, stays the only copy until the replacement has been
    probed and proven longer.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(dest_dir / f"repair_{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        *(["--cookies", cookies] if cookies else []),
        "--format", YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--retries", "10",
        "--force-overwrites",
        "--output", out_template,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    print(f"  [download] {' '.join(cmd[:1] + cmd[-1:])}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"  [download] FAILED for {video_id} (exit {result.returncode})")
        return None
    produced = sorted(dest_dir.glob(f"repair_{video_id}.*"))
    produced = [p for p in produced if p.suffix == ".mp4"]
    return produced[0] if produced else None


def refresh_index_entry(path: Path, index: dict) -> float:
    """Re-probe one file and write its entry into the duration index."""
    duration = _probe(path, "v:0")
    index[path.stem] = {
        "video_stream_seconds": round(duration, 3),
        "size_bytes": path.stat().st_size,
        "filename": path.name,
    }
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repair", action="store_true",
                        help="Re-download every truncated video (default: report only)")
    parser.add_argument("--push", action="store_true",
                        help="Upload repaired videos and the refreshed duration index to S3")
    parser.add_argument("--pull", action="store_true",
                        help="Download raw videos from S3 before scanning (for pods with no local data/)")
    parser.add_argument("--only", metavar="VIDEO_ID",
                        help="Restrict to a single YouTube id, e.g. 5aoew3DYz7Y")
    parser.add_argument("--cookies", default=None,
                        help="Netscape cookies.txt for yt-dlp (YouTube blocks datacenter IPs without it)")
    args = parser.parse_args()

    print(f"Raw video dir: {RAW_VIDEO_DIR}")
    print(f"Duration index: {SOURCE_DURATION_INDEX_PATH}\n")

    if args.pull:
        from app.pipeline import s3_helpers
        print(f"Pulling raw videos from s3://{S3_RAW_PREFIX} ...")
        s3_helpers.download_prefix(S3_RAW_PREFIX, RAW_VIDEO_DIR)

    paths = sorted(RAW_VIDEO_DIR.glob("*.mp4"))
    if args.only:
        paths = [p for p in paths if _video_id_of(p) == args.only]
        if not paths:
            print(f"No local mp4 matches video id {args.only!r} in {RAW_VIDEO_DIR}")
            return 1
    if not paths:
        print(f"No .mp4 files under {RAW_VIDEO_DIR} — pass --pull to fetch them from S3.")
        return 1

    print(f"Probing {len(paths)} video file(s) ...")
    bad = scan(paths)

    if not bad:
        print("\nNo truncated videos: every video track runs as long as its audio track.")
        return 0

    print(f"\n=== {len(bad)} truncated video(s) ===")
    for rec in bad:
        print(f"  {rec['path'].name}")
        print(f"     video {rec['video_seconds']}s / audio {rec['audio_seconds']}s "
              f"— missing {rec['missing_seconds']}s of picture")

    if not args.repair:
        print("\nReport only. Re-run with --repair to re-download these "
              "(add --push to upload the results to S3).")
        return 0

    # --- repair -----------------------------------------------------------
    index: dict = {}
    if SOURCE_DURATION_INDEX_PATH.exists():
        try:
            index = json.loads(SOURCE_DURATION_INDEX_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            index = {}

    scratch = RAW_VIDEO_DIR / "_repair_tmp"
    repaired: list[Path] = []
    still_bad: list[dict] = []

    for rec in bad:
        original: Path = rec["path"]
        video_id = rec["video_id"]
        if not video_id:
            print(f"\n[skip] {original.name}: cannot parse a video id from the filename")
            continue

        print(f"\n[repair] {video_id} ({original.name})")
        fresh = redownload(video_id, scratch, args.cookies)
        if fresh is None:
            continue

        new_video = _probe(fresh, "v:0")
        new_audio = _probe(fresh, "a:0")
        print(f"  [verify] fresh copy: video {new_video:.1f}s / audio {new_audio:.1f}s")

        if new_audio - new_video > TRUNCATION_TOLERANCE_SECONDS:
            # The source itself is short. Nothing to swap in; report it so a
            # human decides whether to prune the out-of-range fragments.
            print(f"  [verify] STILL TRUNCATED — the YouTube source really is "
                  f"video-short. Leaving the existing file alone.")
            still_bad.append({**rec, "refetched_video_seconds": round(new_video, 1)})
            fresh.unlink(missing_ok=True)
            continue

        if new_video <= rec["video_seconds"]:
            print(f"  [verify] fresh copy is no longer than the current one "
                  f"({new_video:.1f}s vs {rec['video_seconds']}s) — keeping the current file.")
            fresh.unlink(missing_ok=True)
            continue

        # Replace in place, keeping the archive's existing filename so every
        # fragment_id / source_video reference downstream still resolves.
        fresh.replace(original)
        print(f"  [replace] {original.name} now has {new_video:.1f}s of video")
        refresh_index_entry(original, index)
        repaired.append(original)

    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()

    if repaired:
        SOURCE_DURATION_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_DURATION_INDEX_PATH.write_text(
            json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nDuration index refreshed for {len(repaired)} video(s): {SOURCE_DURATION_INDEX_PATH}")

    if args.push and repaired:
        from app.pipeline import s3_helpers
        if not s3_helpers.use_s3():
            print("\n--push given but AI4IA_STORAGE_BACKEND is not 's3' — skipping upload.")
        else:
            for path in repaired:
                key = f"{S3_RAW_PREFIX}{path.name}"
                print(f"  [s3] uploading {path.name} -> s3://{key}")
                s3_helpers.upload(path, key)
            print(f"  [s3] uploading duration index -> s3://{SOURCE_DURATION_INDEX_S3_KEY}")
            s3_helpers.upload(SOURCE_DURATION_INDEX_PATH, SOURCE_DURATION_INDEX_S3_KEY)

    print(f"\nRepaired {len(repaired)} video(s).")
    if still_bad:
        print(f"{len(still_bad)} video(s) are short at the source and need a human decision:")
        for rec in still_bad:
            print(f"  {rec['path'].name}: source gives only "
                  f"{rec['refetched_video_seconds']}s of video for "
                  f"{rec['audio_seconds']}s of audio")
        print("  Fragments past the video-track end for these will keep rendering "
              "as black-with-audio. Prune them with the admin delete endpoint only "
              "after confirming the source really is video-short.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
