"""Single source of truth for ffprobe media measurement.

Shared by pipeline steps 08 / 09 (scripts import it as
``app.pipeline.media_probe``). Previously each script carried its own
``get_media_duration`` copy with divergent ffprobe flags and divergent
failure behaviour (one raised, one returned 0.0); a fix or timeout tweak in
one never reached the other. Both variants now live here:

- ``media_duration_or_raise`` — format duration; raises on non-numeric output.
  Step 09 depends on this: a 0-duration segment must abort the export.
- ``media_duration_or_zero`` — format duration; returns 0.0 on any failure.
  Step 08 uses this while planning, where a missing probe is non-fatal.
- ``video_stream_duration`` — the video stream's own duration (not
  format/audio), returns 0.0 on failure. Cached per path.

All results are cached per path behind a lock so the up-to-4 concurrent
render workers in step 09 never probe the same file twice or race the cache.
"""

import subprocess
import threading
from pathlib import Path

_FORMAT_DUR_CACHE: dict[str, float] = {}
_VIDEO_STREAM_DUR_CACHE: dict[str, float] = {}
_CACHE_LOCK = threading.Lock()


def _probe_format_duration(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    return result.stdout.strip()


def media_duration_or_raise(path: Path) -> float:
    """Format duration in seconds. Raises RuntimeError on failure/non-numeric output.

    Cached per path. Step 09 relies on the raise: a 0-duration or unprobeable
    segment is a hard error, never silently treated as empty.
    """
    key = str(path)
    with _CACHE_LOCK:
        if key in _FORMAT_DUR_CACHE:
            return _FORMAT_DUR_CACHE[key]

    output = _probe_format_duration(path)
    try:
        dur = float(output)
    except ValueError:
        raise RuntimeError(f"ffprobe returned non-numeric duration for {path}: {output!r}")

    with _CACHE_LOCK:
        _FORMAT_DUR_CACHE[key] = dur
    return dur


def media_duration_or_zero(path: Path) -> float:
    """Format duration in seconds. Returns 0.0 on any failure. Cached per path."""
    key = str(path)
    with _CACHE_LOCK:
        if key in _FORMAT_DUR_CACHE:
            return _FORMAT_DUR_CACHE[key]

    try:
        dur = float(_probe_format_duration(path) or 0.0)
    except (subprocess.SubprocessError, ValueError, OSError, RuntimeError):
        dur = 0.0

    with _CACHE_LOCK:
        _FORMAT_DUR_CACHE[key] = dur
    return dur


def video_stream_duration(source_path: Path) -> float:
    """Video stream duration (NOT format/audio duration). Returns 0.0 on failure. Cached per path."""
    key = str(source_path)
    with _CACHE_LOCK:
        if key in _VIDEO_STREAM_DUR_CACHE:
            return _VIDEO_STREAM_DUR_CACHE[key]

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(source_path),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        dur = float(result.stdout.strip() or "0")
    except (subprocess.SubprocessError, ValueError, OSError):
        dur = 0.0

    with _CACHE_LOCK:
        _VIDEO_STREAM_DUR_CACHE[key] = dur
    return dur
