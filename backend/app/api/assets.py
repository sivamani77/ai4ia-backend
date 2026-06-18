import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.paths import LATEST_ASSET_MANIFEST, SCRIPTS_DIR, STORYLINES_DIR
from app.services.job_service import create_job
from app.services.script_runner import run_script_background

router = APIRouter(prefix="/assets", tags=["Assets"])


class GenerateAssetsRequest(BaseModel):
    storyline_path: str | None = None
    voice: str = "en-GB-RyanNeural"
    rate: str = "-6%"
    volume: str = "+0%"
    skip_tts: bool = False


def latest_asset_manifest_path() -> Path | None:
    return LATEST_ASSET_MANIFEST if LATEST_ASSET_MANIFEST.exists() else None


@router.get("/latest")
def get_latest_assets():
    if not LATEST_ASSET_MANIFEST.exists():
        raise HTTPException(status_code=404, detail="No latest asset manifest found")

    return json.loads(LATEST_ASSET_MANIFEST.read_text(encoding="utf-8"))


@router.post("/generate")
def generate_assets(request: GenerateAssetsRequest):
    job = create_job("generate_assets", request.model_dump())

    command = [
        "python",
        str(SCRIPTS_DIR / "08_generate_assets.py"),
        "--voice",
        request.voice,
        # Use --flag=value form: rate/volume start with '-'/'+' which argparse would
        # otherwise mistake for another flag (e.g. "--rate -6%" fails).
        f"--rate={request.rate}",
        f"--volume={request.volume}",
    ]

    if request.storyline_path:
        command.extend(["--storyline", request.storyline_path])

    if request.skip_tts:
        command.append("--skip-tts")

    run_script_background(
        job_id=job["job_id"],
        command=command,
        result_resolver=latest_asset_manifest_path,
    )

    return {
        "job_id": job["job_id"],
        "status": "running",
        "message": "Asset generation started",
    }


@router.post("/generate-latest")
def generate_assets_from_latest(request: GenerateAssetsRequest):
    if not STORYLINES_DIR.exists():
        raise HTTPException(status_code=404, detail="No storyline directory found")

    files = sorted(STORYLINES_DIR.glob("storyline_*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No storyline found")

    latest_storyline = files[-1]
    request.storyline_path = str(latest_storyline)

    job = create_job("generate_assets_latest", request.model_dump())

    command = [
        "python",
        str(SCRIPTS_DIR / "08_generate_assets.py"),
        "--storyline",
        str(latest_storyline),
        "--voice",
        request.voice,
        # --flag=value form so argparse doesn't read "-6%"/"+0%" as a new flag.
        f"--rate={request.rate}",
        f"--volume={request.volume}",
    ]

    if request.skip_tts:
        command.append("--skip-tts")

    run_script_background(
        job_id=job["job_id"],
        command=command,
        result_resolver=latest_asset_manifest_path,
    )

    return {
        "job_id": job["job_id"],
        "status": "running",
        "storyline_path": str(latest_storyline),
        "message": "Asset generation from latest storyline started",
    }
