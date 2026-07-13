from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.core.paths import PROJECT_ROOT


def _newest_file_after(directory: Path, after_ts: float, pattern: str = "*.json") -> Path | None:
    """Return the newest file matching pattern created after after_ts (epoch seconds)."""
    candidates = [p for p in directory.glob(pattern) if p.stat().st_mtime >= after_ts - 2]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    stage: str
    status: Literal["completed", "failed", "skipped"]
    message: str
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Script runner
# ---------------------------------------------------------------------------

_STAGE_TIMEOUTS: dict[str, int] = {
    "07_storyline":           600,
    "07b_validate_storyline": 1800,
    "08_assets":             1200,
    "09_export":              900,
    "10_validate_export":     120,
    "11_subtitles":           600,
    "_default":               600,
}


def _kill_process_group(pid: int) -> None:
    """Send SIGTERM to the process group, then SIGKILL after 5 s if still alive."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return

    import threading

    def _force_kill():
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass

    t = threading.Timer(5.0, _force_kill)
    t.daemon = True
    t.start()


_LOG_FLUSH_INTERVAL = 2.0  # seconds between live log flushes to Supabase


def _stream_pump(pipe, stream_name: str, job_id: str, stage_name: str,
                  full_lines: list[str], pending: list[str], lock, append_event) -> None:
    """Read a subprocess pipe line-by-line, buffering for the periodic flush."""
    try:
        for line in iter(pipe.readline, ""):
            full_lines.append(line)
            with lock:
                pending.append(line.rstrip("\n"))
    finally:
        pipe.close()


def run_script(
    job_id: str,
    stage_name: str,
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess and emit stdout/stderr to Supabase events. Returns (exit_code, stdout, stderr).

    `timeout` overrides the per-stage default when provided — used by stage 09,
    whose render+transcribe time scales with the requested video duration and
    the host's encoder (GPU nvenc vs CPU libx264).

    Output is streamed live: reader threads pump stdout/stderr into buffers as
    the subprocess produces them, and a background thread flushes those
    buffers to Supabase as "script_log" events every few seconds, so pollers
    (e.g. load_events_after) see progress while the stage is still running —
    not just after it exits.
    """
    import threading
    import time

    from app.services.job_store import append_event, load_job, save_job

    env = {**os.environ, **(extra_env or {})} if extra_env else None
    if timeout is None:
        timeout = _STAGE_TIMEOUTS.get(stage_name, _STAGE_TIMEOUTS["_default"])

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=os.setsid,  # own process group so we can kill the whole tree
        )

        # Store PID so cancellation can kill the process group
        try:
            job = load_job(job_id)
            job["active_pid"] = process.pid
            save_job(job_id, job)
        except Exception:
            pass

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        pending: list[str] = []
        lock = threading.Lock()
        stop_flush = threading.Event()

        def _flush() -> None:
            with lock:
                if not pending:
                    return
                batch = "\n".join(pending)
                pending.clear()
            append_event(job_id, "script_log", stage_name, {"lines": batch})

        def _flush_loop() -> None:
            while not stop_flush.wait(_LOG_FLUSH_INTERVAL):
                _flush()

        readers = [
            threading.Thread(
                target=_stream_pump,
                args=(process.stdout, "stdout", job_id, stage_name, stdout_lines, pending, lock, append_event),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_pump,
                args=(process.stderr, "stderr", job_id, stage_name, stderr_lines, pending, lock, append_event),
                daemon=True,
            ),
        ]
        for t in readers:
            t.start()
        flusher = threading.Thread(target=_flush_loop, daemon=True)
        flusher.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process.pid)
            process.wait()  # reap after kill
            for t in readers:
                t.join(timeout=5)
            stop_flush.set()
            _flush()
            msg = f"Stage {stage_name} timed out after {timeout}s"
            append_event(job_id, "script_error", stage_name, {"error": msg})
            return -1, "", msg
        finally:
            # Clear PID once the process is no longer running
            try:
                job = load_job(job_id)
                job.pop("active_pid", None)
                save_job(job_id, job)
            except Exception:
                pass

        for t in readers:
            t.join(timeout=5)
        stop_flush.set()
        _flush()

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        append_event(job_id, "script_output", stage_name, {
            "exit_code": process.returncode,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        })
        return process.returncode, stdout, stderr
    except Exception as exc:
        append_event(job_id, "script_error", stage_name, {"error": str(exc)})
        return -1, "", str(exc)


# ---------------------------------------------------------------------------
# ScriptContract — documentation of what each pipeline script expects/produces
# ---------------------------------------------------------------------------

@dataclass
class ScriptContract:
    stage: str
    script_name: str
    required_args: list[str]
    output_description: str


CONTRACTS: dict[str, ScriptContract] = {
    "input_validation": ScriptContract(
        stage="input_validation",
        script_name="(internal)",
        required_args=["prompt"],
        output_description="No file output — validates prompt and required dirs/scripts exist",
    ),
    "07_storyline": ScriptContract(
        stage="07_storyline",
        script_name="07_generate_storyline.py",
        required_args=["--prompt", "--duration", "--style"],
        output_description="storyline_<timestamp>.json in data/storylines/",
    ),
    "07b_validate_storyline": ScriptContract(
        stage="07b_validate_storyline",
        script_name="07b_validate_storyline_with_llm.py",
        required_args=["--storyline", "--transcripts-dir", "--fallback-transcripts-dir",
                       "--output", "--max-iterations"],
        output_description="<storyline_stem>_validated.json alongside the input storyline",
    ),
    "08_assets": ScriptContract(
        stage="08_assets",
        script_name="08_generate_assets.py",
        required_args=["--storyline"],
        output_description="data/generated_assets/latest_asset_manifest.json plus voiceover and visual files",
    ),
    "09_export": ScriptContract(
        stage="09_export",
        script_name="09_export_video.py",
        required_args=["--manifest", "--output"],
        output_description="<job_id>[_title].mp4 plus _export_manifest.json, .srt, .vtt in data/exports/",
    ),
    "10_validate_export": ScriptContract(
        stage="10_validate_export",
        script_name="10_validate_video_export.py",
        required_args=["--export-manifest"],
        output_description="No file output — validates the export manifest; non-zero exit is non-fatal",
    ),
    "11_subtitles": ScriptContract(
        stage="11_subtitles",
        script_name="11_generate_final_subtitles.py",
        required_args=["--video", "--model", "--update-manifest"],
        output_description="<video_stem>.srt and <video_stem>.vtt alongside the exported video",
    ),
}


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    job_id: str
    request: dict[str, Any]
    storyline_path: str | None = None
    validated_storyline_path: str | None = None
    # In-memory copy of the parsed validated-storyline JSON. stage_07b_validate
    # deletes its own /tmp working dir (cleanup_tmp_dir) before returning in S3
    # mode, so anything that needs the content AFTER the stage returns (e.g.
    # the storyline cache write) can't re-read validated_storyline_path from
    # disk — it'll be gone. Set this instead, from the same in-memory dict the
    # stage already parsed before cleanup.
    validated_storyline_data: dict[str, Any] | None = None
    asset_manifest_path: str | None = None
    export_manifest_path: str | None = None
    video_path: str | None = None
    srt_path: str | None = None
    vtt_path: str | None = None
