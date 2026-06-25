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
# Script runner (moved from video_job_runner._run_script)
# ---------------------------------------------------------------------------

def _logs_dir(job_id: str) -> Path:
    from app.core.paths import DATA_DIR
    return DATA_DIR / "jobs" / job_id / "logs"


def run_script(job_id: str, stage_name: str, cmd: list[str]) -> tuple[int, str, str]:
    """Run a subprocess, save stdout/stderr logs, return (exit_code, stdout, stderr)."""
    logs = _logs_dir(job_id)
    out_log = logs / f"{stage_name}.stdout.log"
    err_log = logs / f"{stage_name}.stderr.log"
    pid_file = logs / f"{stage_name}.pid"

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pid_file.write_text(str(process.pid))
        stdout, stderr = process.communicate()
        pid_file.unlink(missing_ok=True)
        out_log.write_text(stdout or "", encoding="utf-8")
        err_log.write_text(stderr or "", encoding="utf-8")
        return process.returncode, stdout or "", stderr or ""
    except Exception as exc:
        pid_file.unlink(missing_ok=True)
        err_log.write_text(str(exc), encoding="utf-8")
        return -1, "", str(exc)


def kill_running_stage(job_id: str, stage_name: str) -> bool:
    """Send SIGTERM to the process recorded in the stage's PID file.

    Returns True if a signal was delivered, False if no process was found
    (already finished, never started, or PID file missing).
    """
    pid_file = _logs_dir(job_id) / f"{stage_name}.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        pid_file.unlink(missing_ok=True)
        return True
    except (ProcessLookupError, ValueError):
        pid_file.unlink(missing_ok=True)
        return False


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
