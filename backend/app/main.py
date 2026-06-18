import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_NAME, API_PREFIX
from app.core.paths import ensure_runtime_dirs

from app.api.dashboard import router as dashboard_router
from app.api.jobs import router as jobs_router
from app.api.storylines import router as storylines_router
from app.api.assets import router as assets_router
from app.api.exports import router as exports_router
from app.api.archive import router as archive_router
from app.api.admin import router as admin_router
from app.api.insights import router as insights_router
from app.api.compare import router as compare_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai4ia.api")


ensure_runtime_dirs()

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description="Backend API wrapper for AI4IA Demo Video Studio pipeline.",
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
    }


app.include_router(dashboard_router, prefix=API_PREFIX)
app.include_router(jobs_router, prefix=API_PREFIX)
app.include_router(storylines_router, prefix=API_PREFIX)
app.include_router(assets_router, prefix=API_PREFIX)
app.include_router(exports_router, prefix=API_PREFIX)
app.include_router(archive_router, prefix=API_PREFIX)
app.include_router(admin_router, prefix=API_PREFIX)
app.include_router(insights_router, prefix=API_PREFIX)
app.include_router(compare_router, prefix=API_PREFIX)