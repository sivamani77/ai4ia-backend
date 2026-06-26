import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger.json import JsonFormatter

from app.core.auth import require_api_key
from app.core.config import API_PREFIX
from app.core.paths import ensure_runtime_dirs
from app.core.settings import settings

from app.api.dashboard import router as dashboard_router
from app.api.jobs import router as jobs_router
from app.api.storylines import router as storylines_router
from app.api.assets import router as assets_router
from app.api.exports import router as exports_router
from app.api.archive import router as archive_router
from app.api.admin import router as admin_router
from app.api.insights import router as insights_router
from app.api.compare import router as compare_router
from app.api.video_jobs import router as video_jobs_router
from app.api.video_media import router as video_media_router
from app.api.concept_mindmap import router as concept_mindmap_router
from app.api.ingest import router as ingest_router


class _PipelineLogFilter(logging.Filter):
    """Inject default job_id/stage so all log records satisfy the formatter."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = getattr(record, "job_id", "-")
        record.stage = getattr(record, "stage", "-")
        return True


_pipeline_filter = _PipelineLogFilter()
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

if os.environ.get("LOG_FORMAT", "").lower() == "json":
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(job_id)s %(stage)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
    ))
    _handler.addFilter(_pipeline_filter)
    _root_logger.handlers = [_handler]
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(job_id)s | %(stage)s | %(message)s",
    )
    for _h in _root_logger.handlers:
        _h.addFilter(_pipeline_filter)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger("ai4ia.api")


def _reap_orphaned_ingest_jobs() -> None:
    """Mark any queued/running ingest jobs as failed on startup (thread-mode orphans only)."""
    try:
        from app.services.ingest_job_runner import _redis_reachable, reap_orphaned_ingest_jobs
        if _redis_reachable():
            logger.info("Redis is reachable; skipping ingest orphan reaping")
            return
        reap_orphaned_ingest_jobs()
    except Exception:
        logger.exception("Failed to reap orphaned ingest jobs on startup")


def _reap_orphaned_jobs() -> None:
    """Mark any jobs still in 'running'/'queued' state as failed.

    These are local-thread jobs whose owning API process was restarted.
    Celery-backed jobs are left alone because workers may still be running.
    """
    try:
        from app.services.video_job_runner import _redis_reachable

        if _redis_reachable():
            logger.info("Redis is reachable; skipping local-thread orphan job reaping")
            return

        from datetime import datetime, timezone
        from app.db.repositories.job_repository import JobRepository
        repo = JobRepository()
        all_jobs = repo.list_jobs()
        orphaned = [j for j in all_jobs if j.get("status") in ("running", "queued")]
        for job in orphaned:
            jid = job.get("id")
            repo.update_job(jid, {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": "Server restarted while job was in progress",
            })
            logger.info("Reaped orphaned job id=%s status=%s", jid, job.get("status"))
    except Exception:
        logger.exception("Failed to reap orphaned jobs on startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()

    for warning in settings.validate_deployment_env():
        logger.warning(warning)

    await asyncio.to_thread(_reap_orphaned_jobs)
    await asyncio.to_thread(_reap_orphaned_ingest_jobs)

    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API wrapper for AI4IA Demo Video Studio pipeline.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    logger.info(
        "[%s] START %s %s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start_time
        logger.exception(
            "[%s] ERROR %s %s %.2fs",
            request_id,
            request.method,
            request.url.path,
            duration,
        )
        raise

    duration = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{duration:.3f}"

    logger.info(
        "[%s] END %s %s %s %.2fs",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )

    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_supabase() -> dict:
    try:
        from app.db.repositories.job_repository import JobRepository
        JobRepository().list_jobs()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _check_chroma() -> dict:
    try:
        from app.services import embedding_index
        if not embedding_index.available():
            return {"ok": False, "error": "collection not loaded"}
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _check_ollama() -> dict:
    import urllib.request
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as r:
            return {"ok": r.status == 200}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


@app.get(f"{API_PREFIX}/health")
async def health():
    from app.services.video_job_runner import _redis_reachable

    redis_up = _redis_reachable()
    use_celery = bool(settings.redis_url) and redis_up

    supabase, chroma, ollama = await asyncio.gather(
        asyncio.to_thread(_check_supabase),
        asyncio.to_thread(_check_chroma),
        asyncio.to_thread(_check_ollama),
    )

    all_ok = redis_up and supabase["ok"] and chroma["ok"]

    return {
        "status": "ok" if all_ok else "degraded",
        "service": settings.app_name,
        "checks": {
            "redis": {"ok": redis_up},
            "supabase": supabase,
            "chroma": chroma,
            "ollama": ollama,
        },
        "worker_mode": "celery" if use_celery else "thread_fallback",
    }


_auth_dep = [Depends(require_api_key)]

app.include_router(dashboard_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(jobs_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(storylines_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(assets_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(exports_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(archive_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(admin_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(insights_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(compare_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(video_jobs_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(video_media_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(concept_mindmap_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(ingest_router, prefix=API_PREFIX, dependencies=_auth_dep)
