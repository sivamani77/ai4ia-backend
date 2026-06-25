from fastapi import APIRouter, HTTPException

from app.services.job_store import load_job, list_jobs
from app.api.video_jobs import _job_response

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("")
def get_jobs():
    return [_job_response(j).model_dump() for j in list_jobs()]


@router.get("/{job_id}")
def get_job_status(job_id: str):
    try:
        job = load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job).model_dump()
