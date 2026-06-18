from fastapi import APIRouter, HTTPException

from app.services.job_service import get_job, list_jobs

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("")
def get_jobs():
    return list_jobs()


@router.get("/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job
