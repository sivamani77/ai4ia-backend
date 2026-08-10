from __future__ import annotations
import subprocess
import time
from datetime import datetime


def log(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}", flush=True)


def run_cmd(cmd: list[str]) -> None:
    log("Running: " + " ".join(cmd))
    _t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _elapsed = time.time() - _t0

    if result.returncode != 0:
        if result.stdout.strip():
            log(result.stdout.rstrip())
        if result.stderr.strip():
            log(result.stderr.rstrip())
        detail = (result.stderr.strip() or result.stdout.strip() or "no output")[-3000:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{detail}")

    log(f"  [cmd-timing] {cmd[0]} finished in {_elapsed:.1f}s")


def run_cmd_capture(cmd: list[str]) -> str:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def _probe_nvenc() -> bool:
    """True only if h264_nvenc is both compiled in AND can actually encode a
    frame on this host (i.e. a working NVIDIA GPU/driver is present).
    ffmpeg -encoders lists h264_nvenc whenever the build includes NVENC
    support, regardless of whether a GPU/CUDA driver is available at
    runtime — so a codec-list check alone is a false positive on CPU-only
    hosts running a GPU-capable ffmpeg build.
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        if "h264_nvenc" not in r.stdout:
            return False

        probe = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
        )
        return probe.returncode == 0
    except Exception:
        return False


NVENC: bool = _probe_nvenc()


def _vcodec_flags(threads: int | None = 2) -> list[str]:
    """Return video codec flags for the current environment.

    On NVIDIA GPU hosts h264_nvenc is used (p4 preset, VBR quality 23).
    On CPU-only hosts libx264 is used with veryfast preset.
    threads is only applied for the CPU path; nvenc manages its own scheduling.
    """
    if NVENC:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-pix_fmt", "yuv420p"]
    flags = ["-c:v", "libx264"]
    if threads is not None:
        flags += ["-threads", str(threads)]
    return flags + ["-preset", "veryfast", "-pix_fmt", "yuv420p"]
