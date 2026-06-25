# DEPRECATED: This router uses the legacy 3-step script-runner flow
# (job_service.create_job + run_script_background calling 09_export_video.py).
# The GET endpoints (list, stream, artifact download) remain useful.
# POST /generate and /generate-latest are superseded by POST /api/video-jobs.
# Do not add new features to the POST endpoints here.

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.core.paths import EXPORTS_DIR, LATEST_ASSET_MANIFEST, SCRIPTS_DIR
from app.services.job_service import create_job
from app.services.script_runner import run_script_background

# Assets are written alongside exports; manifests live in the assets dir.
# We allow manifest_path from anywhere under data/ by checking the assets dir's parent.
_DATA_DIR = EXPORTS_DIR.resolve().parent


def _safe_manifest_path(raw: str) -> str:
    resolved = Path(raw).resolve()
    if not str(resolved).startswith(str(_DATA_DIR)):
        raise HTTPException(status_code=400, detail="manifest_path is outside the allowed directory")
    return str(resolved)

router = APIRouter(prefix="/exports", tags=["Exports"])


class GenerateExportRequest(BaseModel):
    output: str = "ai4ia_demo_mvp.mp4"
    manifest_path: str | None = None
    burn_subtitles: bool = True
    no_subtitles: bool = False


ARTIFACT_TYPES = {"download", "vtt", "srt", "manifest", "qa", "timeline"}


def latest_export_path() -> Path | None:
    if not EXPORTS_DIR.exists():
        return None

    files = sorted(EXPORTS_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def latest_asset_manifest_path() -> Path | None:
    return LATEST_ASSET_MANIFEST if LATEST_ASSET_MANIFEST.exists() else None


def normalized_manifest_path(manifest_path: str | None) -> str | None:
    if manifest_path is None:
        return None

    value = manifest_path.strip()
    if not value or value.lower() == "string":
        return None

    return value


def safe_export_path(filename: str) -> Path:
    exports_root = EXPORTS_DIR.resolve()
    path = (EXPORTS_DIR / filename).resolve()

    if not str(path).startswith(str(exports_root)):
        raise HTTPException(status_code=400, detail="Invalid export filename")

    if not path.exists():
        raise HTTPException(status_code=404, detail="Export artifact not found")

    return path


def export_artifact_path(filename: str, artifact_type: str) -> Path:
    if artifact_type not in ARTIFACT_TYPES:
        raise HTTPException(status_code=404, detail="Unsupported artifact type")

    mp4_path = safe_export_path(filename)

    if mp4_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Invalid export filename")

    stem = Path(filename).stem
    artifact_name = {
        "download": f"{stem}.mp4",
        "vtt": f"{stem}.vtt",
        "srt": f"{stem}.srt",
        "manifest": f"{stem}_export_manifest.json",
        "qa": f"{stem}_qa_report.json",
        "timeline": f"{stem}_timeline_metadata.json",
    }[artifact_type]

    return safe_export_path(artifact_name)


@router.get("")
def list_exports():
    items = []

    if not EXPORTS_DIR.exists():
        return items

    for path in sorted(EXPORTS_DIR.glob("*.mp4"), key=lambda export: export.stat().st_mtime, reverse=True):
        stem = path.stem
        srt_path = EXPORTS_DIR / f"{stem}.srt"
        vtt_path = EXPORTS_DIR / f"{stem}.vtt"
        manifest_path = EXPORTS_DIR / f"{stem}_export_manifest.json"
        qa_report_path = EXPORTS_DIR / f"{stem}_qa_report.json"
        timeline_path = EXPORTS_DIR / f"{stem}_timeline_metadata.json"
        size_bytes = path.stat().st_size

        items.append(
            {
                "filename": path.name,
                "stem": stem,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "stream_url": f"/api/exports/{path.name}/stream",
                "download_url": f"/api/exports/{path.name}/artifact/download",
                "vtt_url": f"/api/exports/{path.name}/artifact/vtt",
                "srt_url": f"/api/exports/{path.name}/artifact/srt",
                "manifest_url": f"/api/exports/{path.name}/artifact/manifest",
                "qa_report_url": f"/api/exports/{path.name}/artifact/qa",
                "timeline_url": f"/api/exports/{path.name}/artifact/timeline",
                "has_srt": srt_path.exists(),
                "has_vtt": vtt_path.exists(),
                "has_manifest": manifest_path.exists(),
                "has_qa_report": qa_report_path.exists(),
                "has_timeline": timeline_path.exists(),
            }
        )

    return items


@router.post("/generate")
def generate_export(request: GenerateExportRequest):
    job = create_job("generate_export", request.model_dump())

    command = [
        "python",
        str(SCRIPTS_DIR / "09_export_video.py"),
        "--output",
        request.output,
    ]

    manifest_path = normalized_manifest_path(request.manifest_path)
    if manifest_path:
        command.extend(["--manifest", _safe_manifest_path(manifest_path)])

    if request.no_subtitles or not request.burn_subtitles:
        command.append("--no-subtitles")

    run_script_background(
        job_id=job["job_id"],
        command=command,
        result_resolver=latest_export_path,
    )

    return {
        "job_id": job["job_id"],
        "status": "running",
        "message": "Video export started",
    }


@router.post("/generate-latest")
def generate_export_latest(request: GenerateExportRequest):
    latest_manifest = latest_asset_manifest_path()
    if not latest_manifest:
        raise HTTPException(status_code=404, detail="No latest asset manifest found")

    payload = request.model_dump()
    payload["manifest_path"] = str(latest_manifest)

    job = create_job("generate_export_latest", payload)

    command = [
        "python",
        str(SCRIPTS_DIR / "09_export_video.py"),
        "--output",
        request.output,
    ]

    if request.no_subtitles or not request.burn_subtitles:
        command.append("--no-subtitles")

    run_script_background(
        job_id=job["job_id"],
        command=command,
        result_resolver=latest_export_path,
    )

    return {
        "job_id": job["job_id"],
        "status": "running",
        "manifest_path": str(latest_manifest),
        "message": "Video export from latest assets started",
    }


@router.get("/{filename}/stream")
def stream_export(filename: str):
    path = safe_export_path(filename)

    if path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Invalid export filename")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
    )


@router.get("/{filename}/artifact/{artifact_type}")
def get_export_artifact(filename: str, artifact_type: str):
    path = export_artifact_path(filename, artifact_type)

    if artifact_type == "download":
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    if artifact_type == "vtt":
        return FileResponse(path, media_type="text/vtt", filename=path.name)

    if artifact_type == "srt":
        return FileResponse(path, media_type="application/x-subrip", filename=path.name)

    payload = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(content=payload, media_type="application/json")
