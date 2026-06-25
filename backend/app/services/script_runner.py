import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from app.core.paths import PROJECT_ROOT
from app.services.job_store import load_job, save_job


def run_script_background(
    job_id: str,
    command: list[str],
    result_resolver=None,
) -> None:
    thread = threading.Thread(
        target=_run_script,
        args=(job_id, command, result_resolver),
        daemon=True,
    )
    thread.start()


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    job = load_job(job_id)
    job.update(updates)
    save_job(job_id, job)
    return job


def _run_script(job_id: str, command: list[str], result_resolver=None) -> None:
    job = _update_job(
        job_id,
        status="running",
        progress=10,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command: " + " ".join(command) + "\n\n")
            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout or []:
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

        if return_code != 0:
            _update_job(
                job_id,
                status="failed",
                progress=100,
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=f"Script failed with exit code {return_code}",
            )
            return

        result_path = None
        if result_resolver:
            try:
                resolved = result_resolver()
                result_path = str(resolved) if resolved else None
            except Exception:
                result_path = None

        _update_job(
            job_id,
            status="completed",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
            result_path=result_path,
            error=None,
        )

    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
