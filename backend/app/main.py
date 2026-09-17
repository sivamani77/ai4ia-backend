import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger.json import JsonFormatter

from app.core.auth import require_api_key, require_role
from app.core.config import API_PREFIX
from app.core import dependency_state
from app.core.otel import is_enabled as _otel_enabled, setup_tracing
from app.core.paths import ensure_runtime_dirs
from app.core.settings import settings
from app.core import metrics as _metrics
from app.db.schema import ensure_pipeline_schema
from app.services.auth_data_migration import run_once as run_auth_data_migration_once

# Must run before the httpx/requests/psycopg2 instrumentors below patch
# their clients, and before the app is constructed.
setup_tracing("ai4ia-api")

from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.archive import router as archive_router
from app.api.admin import router as admin_router
from app.api.admin_users import router as admin_users_router
from app.api.insights import router as insights_router
from app.api.compare import router as compare_router
from app.api.video_jobs import router as video_jobs_router
from app.api.video_media import router as video_media_router
from app.api.concept_mindmap import router as concept_mindmap_router
from app.api.ingest import router as ingest_router
from app.api.provenance import router as provenance_router


class _PipelineLogFilter(logging.Filter):
    """Inject default job_id/stage/otelTraceID/otelSpanID so all log records
    satisfy the formatter. otelTraceID/otelSpanID are normally set by
    LoggingInstrumentor (see app.core.otel) — the defaults here only kick in
    when tracing is disabled (no OTEL_EXPORTER_OTLP_ENDPOINT)."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = getattr(record, "job_id", "-")
        record.stage = getattr(record, "stage", "-")
        record.otelTraceID = getattr(record, "otelTraceID", "-")
        record.otelSpanID = getattr(record, "otelSpanID", "-")
        return True


_pipeline_filter = _PipelineLogFilter()
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)

if os.environ.get("LOG_FORMAT", "").lower() == "json":
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(job_id)s %(stage)s %(otelTraceID)s %(message)s",
        rename_fields={"asctime": "ts", "levelname": "level", "name": "logger", "otelTraceID": "trace_id"},
    ))
    _handler.addFilter(_pipeline_filter)
    _root_logger.handlers = [_handler]
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(job_id)s | %(stage)s | %(otelTraceID)s | %(message)s",
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
        if orphaned:
            _metrics.jobs_orphan_reaped_total.inc(len(orphaned))
    except Exception:
        logger.exception("Failed to reap orphaned jobs on startup")


_DEPENDENCY_CHECK_INTERVAL_SECONDS = 60


async def _dependency_metrics_loop() -> None:
    """Periodically refresh dependency_up/dependency_check_duration_seconds
    so they reflect current state even if nobody polls /api/health — reuses
    the same check functions /api/health calls, just on a timer instead of
    per-request."""
    while True:
        for name, check_fn in (
            ("postgres", _check_postgres),
            ("chroma", _check_chroma),
        ):
            start = time.perf_counter()
            try:
                result = await asyncio.to_thread(check_fn)
                ok = bool(result.get("ok"))
            except Exception:
                logger.exception("Dependency metrics check failed for %s", name)
                ok = False
            elapsed = time.perf_counter() - start
            _metrics.dependency_up.labels(dependency=name).set(1 if ok else 0)
            _metrics.dependency_check_duration_seconds.labels(dependency=name).observe(elapsed)
        await asyncio.sleep(_DEPENDENCY_CHECK_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()

    for warning in settings.validate_deployment_env():
        logger.warning(warning)

    try:
        await asyncio.to_thread(ensure_pipeline_schema)
    except Exception:
        logger.exception("Failed to ensure pipeline Postgres schema on startup")

    try:
        await asyncio.to_thread(run_auth_data_migration_once)
    except Exception:
        logger.exception("Failed to run one-time Supabase auth data migration on startup")

    await asyncio.to_thread(_reap_orphaned_jobs)
    await asyncio.to_thread(_reap_orphaned_ingest_jobs)

    dependency_metrics_task = asyncio.create_task(_dependency_metrics_loop())

    yield

    dependency_metrics_task.cancel()
    try:
        await dependency_metrics_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Backend API wrapper for AI4IA Demo Video Studio pipeline.",
    lifespan=lifespan,
)

# RED metrics (request rate/error/duration by method+path+status) served at
# /metrics for Prometheus scraping — unauthenticated, same as /api/health.
Instrumentator(excluded_handlers=["/metrics", f"{API_PREFIX}/health"]).instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)

# Traces — no-op unless JAEGER_URL is set (see app.core.otel).
if _otel_enabled():
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="/metrics," + f"{API_PREFIX}/health")
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    # skip_dep_check: the instrumentor's dependency check looks for a
    # "psycopg2" distribution specifically, but this project installs
    # "psycopg2-binary" (same `import psycopg2`, different package name) —
    # without this it silently no-ops instead of instrumenting.
    Psycopg2Instrumentor().instrument(skip_dep_check=True)


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


# Translates prose fields in JSON responses when the request carries
# ?lang=/&region= or Accept-Language (no-op unless TRANSLATION_ENABLED).
from app.middleware.translation import TranslationMiddleware  # noqa: E402

app.add_middleware(TranslationMiddleware)


def _openapi_with_translation_params():
    """Advertise the middleware-level lang/region query params on every
    operation so they are usable from Swagger UI (/docs)."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    from app.middleware.translation import EXCLUDED_PATH_SUBSTRINGS

    schema = get_openapi(
        title=app.title, version=app.version,
        description=app.description, routes=app.routes,
    )
    lang_param = {
        "name": "lang",
        "in": "query",
        "required": False,
        "description": "Translate prose fields of the response "
                       f"(supported: {', '.join(settings.translation_supported_languages)}). "
                       "Requires TRANSLATION_ENABLED on the server.",
        "schema": {"type": "string", "enum": list(settings.translation_supported_languages)},
    }
    region_param = {
        "name": "region",
        "in": "query",
        "required": False,
        "description": "Optional ISO region refining the dialect, e.g. lang=pt&region=BR.",
        "schema": {"type": "string"},
    }
    for path, operations in schema.get("paths", {}).items():
        if any(part in path for part in EXCLUDED_PATH_SUBSTRINGS):
            continue
        for op in operations.values():
            if isinstance(op, dict):
                op.setdefault("parameters", []).extend([lang_param, region_param])
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_translation_params

# allow_credentials=True is required for the refresh-token cookie
# (POST /api/auth/refresh, /login, /logout) to be sent/received by the
# browser. Browsers reject credentialed requests against a wildcard origin,
# so ALLOWED_ORIGINS must be a real, explicit origin list in any environment
# that uses cookie-based auth — "*" will silently break login for the
# frontend, not open up access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_postgres() -> dict:
    """Round-trips a real query against DATABASE_URL. Named for Supabase until
    the auth migration; the pipeline tables it touches have been on Postgres
    since well before that, so the old name described nothing."""
    try:
        from app.db.repositories.job_repository import JobRepository
        JobRepository().list_jobs()
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


def _check_chroma() -> dict:
    import urllib.request, json as _json
    host = settings.chroma_host
    port = settings.chroma_port
    url = f"http://{host}:{port}/api/v2/tenants/default_tenant/databases/default_database/collections"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as r:
            collections = _json.loads(r.read())
            names = [c.get("name") for c in collections]
            if settings.chroma_collection in names:
                return {"ok": True}
            return {"ok": False, "error": f"collection '{settings.chroma_collection}' not found (have: {names})"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}


@app.get(f"{API_PREFIX}/health")
async def health():
    from app.services.video_job_runner import _redis_reachable

    redis_up = _redis_reachable()
    use_celery = bool(settings.redis_url) and redis_up

    # snapshot() reads Redis, so it goes through to_thread like the other
    # two — it is a blocking client, and /api/health is hit by every probe
    # on every pod. Never let it stall the event loop.
    postgres, chroma, llm_hosts = await asyncio.gather(
        asyncio.to_thread(_check_postgres),
        asyncio.to_thread(_check_chroma),
        asyncio.to_thread(dependency_state.snapshot),
    )

    # Deliberately excludes the model hosts. They are scale-to-zero GPU
    # deployments, so "nobody has called it recently" is their normal idle
    # state — folding that into readiness would mark every API pod unready
    # whenever the GPU was correctly scaled down, and crash-loop the
    # deployment. llm_hosts below is reported for humans, never gated on.
    all_ok = redis_up and postgres["ok"] and chroma["ok"]

    return {
        "status": "ok" if all_ok else "degraded",
        "service": settings.app_name,
        "checks": {
            "redis": {"ok": redis_up},
            "postgres": postgres,
            "chroma": chroma,
        },
        # Last known state of vLLM/Ollama, recorded by whichever process
        # actually called them. Reading this costs nothing — no request is
        # made to the hosts here, which is the entire point: probing them is
        # what used to keep the GPU resident 24/7.
        "llm_hosts": llm_hosts,
        "worker_mode": "celery" if use_celery else "thread_fallback",
    }


@app.get(f"{API_PREFIX}/models/status", dependencies=[Depends(require_api_key)])
async def models_status():
    """Status variable for the model-gate warm-up (app/services/model_gate.py)
    — reads cached Redis state only, never probes the model hosts. Gating
    decisions never read this; they call model_gate.is_ready()/ensure_ready(),
    which probe directly. This is for the frontend to show "starting
    models..." after login."""
    from app.services import model_gate

    return await asyncio.to_thread(model_gate.all_states)


_auth_dep = [Depends(require_api_key)]
_admin_dep = [Depends(require_api_key), Depends(require_role("admin"))]

# No _auth_dep here: signup/login/etc. must be reachable without an API key
# or an existing session — that's the whole point of the auth endpoints.
app.include_router(auth_router, prefix=API_PREFIX)

app.include_router(dashboard_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(archive_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(admin_router, prefix=API_PREFIX, dependencies=_admin_dep)
app.include_router(admin_users_router, prefix=API_PREFIX, dependencies=_admin_dep)
app.include_router(insights_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(compare_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(video_jobs_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(video_media_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(concept_mindmap_router, prefix=API_PREFIX, dependencies=_auth_dep)
app.include_router(ingest_router, prefix=API_PREFIX, dependencies=_admin_dep)
app.include_router(provenance_router, prefix=API_PREFIX, dependencies=_admin_dep)
