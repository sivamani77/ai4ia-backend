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
    "07b_validate_storyline": 900,
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


def run_script(
    job_id: str,
    stage_name: str,
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess and emit stdout/stderr to Supabase events. Returns (exit_code, stdout, stderr)."""
    from app.services.job_store import append_event, load_job, save_job

    env = {**os.environ, **(extra_env or {})} if extra_env else None
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

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_group(process.pid)
            process.communicate()  # drain pipes to avoid deadlock
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

        stdout = stdout or ""
        stderr = stderr or ""
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
    asset_manifest_path: str | None = None
    export_manifest_path: str | None = None
    video_path: str | None = None
    srt_path: str | None = None
    vtt_path: str | None = None
