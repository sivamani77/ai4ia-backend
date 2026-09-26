# AI4IA Video Studio — Backend

Backend API and job pipeline for the **AI4IA Demo Video Studio**: it ingests the AI4IA conference video archive from YouTube, indexes it into searchable, speaker-attributed transcript fragments, and generates fully edited documentary-style videos (storyline → assets → render → subtitles) from a natural-language prompt.

Built with **FastAPI** + **Celery**, backed by **Redis** (broker), **Postgres** (persistence + user auth), **ChromaDB** (vector search), **Ollama / vLLM** (LLM inference), and **S3** object storage.

---

## Architecture

```
                 ┌──────────────────────┐
  Frontend ────▶ │  FastAPI (app.main)  │────▶ Postgres — jobs, storylines, insights
                 │  REST + SSE + HLS    │────▶ Postgres — auth: users, OTP, sessions
                 └─────────┬────────────┘
                           │ enqueue (Celery)
                           ▼
                 ┌──────────────────────┐
                 │  Redis (broker +     │
                 │  result backend)     │
                 └─────────┬────────────┘
                           ▼
                 ┌──────────────────────┐      ┌────────────────┐
                 │  Celery worker       │────▶ │ ChromaDB       │ semantic search
                 │  video_jobs /        │────▶ │ Ollama / vLLM  │ storyline + LLM validation
                 │  ingest_jobs queues  │────▶ │ S3             │ shared job artifacts
                 └──────────────────────┘      └────────────────┘
```

- **API process** ([backend/app/main.py](backend/app/main.py)) — serves the REST API, validates and creates jobs, streams job events/logs, and serves rendered video (progressive download, signed URLs, HLS).
- **Worker process** ([backend/app/worker/](backend/app/worker/)) — executes the heavy pipelines. Two queues exist: `video_jobs` (video generation) and `ingest_jobs` (YouTube channel sync). Tasks use `acks_late` so a crashed worker re-queues its job. **See [Known gaps](#known-gaps) — the deployed entrypoint currently consumes only `video_jobs`.**
- **Thread fallback** — if Redis is unreachable, video jobs run in a background thread inside the API process, and orphaned jobs are reaped on startup. Fine for dev, not for production.
- **Storage backends** — `local` (single machine, shared `data/` directory) or `s3` (split API/worker deployments: the worker mirrors exports, HLS segments, and assets to S3 so the API can serve them from another machine). Production uses `s3`.

### Persistence

Everything lives in one Postgres database, reached via `DATABASE_URL`:

| Group | Tables |
|---|---|
| Pipeline | `jobs`, `job_events`, `fragments`, `storylines`, `insights`, `videos`, `speakers`, `ingest_*`, `llm_call_traces`, `agent_decision_records` |
| Auth | `app_users`, `user_sessions`, `otp_codes`, `refresh_tokens`, `auth_audit` |

Schema is applied idempotently at API startup by `ensure_pipeline_schema()` in [backend/app/db/schema.py](backend/app/db/schema.py), which is the source of truth; `db/migrations/` holds the reviewable copies.

Both groups were previously in Supabase. The pipeline tables moved first; the auth tables followed — see [db/migrations/README_auth_migration.md](db/migrations/README_auth_migration.md) for the migration and its one-time data copy. **The `supabase` package is no longer a dependency.** The Supabase projects are retained read-only as the backup of record.

## Pipelines

### Video generation (a "video job")

Triggered by `POST /api/video-jobs` with a prompt, target duration, and style. Orchestrated by [video_job_runner.py](backend/app/services/video_job_runner.py) as a sequence of stage modules in [backend/app/pipeline/](backend/app/pipeline/):

| Stage | Module | What it does |
|---|---|---|
| Input validation | `stage_input_validation` | Validates the request and environment |
| 07 — Storyline | `stage_07_storyline` | Retrieval over fragment embeddings + strong LLM (gpt-oss-120b via Ollama, or vLLM behind a flag) plans and writes a structured storyline |
| 07b — Validate | `stage_07b_validate` | LLM editorial validation of the storyline |
| 08 — Assets | `stage_08_assets` | Generates intro/outro/section cards, TTS narration, overlays |
| 09 — Export | `stage_09_export` | Cuts, concatenates, and renders the final video with ffmpeg (optional progressive HLS streaming while rendering, behind `AI4IA_ENABLE_HLS_STREAMING`) |
| 10 — Validate | `stage_10_validate` | Automated QA of the exported video |
| 11 — Subtitles | `stage_11_subtitles` | Final subtitle generation (SRT/VTT) |

Each stage runs as a subprocess under a per-stage timeout (`_STAGE_TIMEOUTS` in [pipeline/base.py](backend/app/pipeline/base.py)) with process-group kill on timeout or cancellation. Every stage emits structured job events (consumable via `GET /api/video-jobs/{id}/events`), progress, per-stage logs, and LLM trace rows for provenance. Jobs are cancellable mid-stage.

### Ingestion (a "sync job")

Triggered by `POST /api/ingest/sync` (or nightly by Celery beat when `ENABLE_AUTO_SYNC=true`). Runs the archive-building scripts `01`–`06` in [scripts/](scripts/): download videos → extract/clean audio → transcribe (faster-whisper) → create fragments → classify → embed into ChromaDB.

yt-dlp needs two things that are easy to miss: a **Node.js runtime** in the image (YouTube's JS challenge) and a **cookies file** (`YOUTUBE_COOKIES_FILE`) when running from a datacenter IP. Without either, downloads fail *silently* — every video is reported "not available" rather than raising.

## API surface

All routes live under `/api` (interactive docs at `/docs`). Protected routes require an `X-API-Key` header when `API_KEY` is set; user-facing auth is JWT access tokens + rotating refresh-token cookies.

| Area | Prefix | Highlights |
|---|---|---|
| Auth | `/api/auth` | signup + email OTP, login, refresh, logout, password reset, general→researcher upgrade |
| Video jobs | `/api/video-jobs` | create/list/cancel jobs, events, storyline, validation report, provenance, assets, result |
| Video media | `/api/video-jobs/{id}/video` | stream, download, signed URL, HLS playlist/segments, subtitles, thumbnail, timeline |
| Archive | `/api/archive` | semantic search, speakers, themes, fragments, source-video streaming |
| Insights | `/api/insights` | LLM-generated insights, sources, provenance, feed |
| Compare / Relay | `/api/compare`, `/api/synthesis`, `/api/relay` | cross-speaker comparison, theme synthesis, quote relays |
| Concept mindmap | `/api/concept-mindmap` | concept-graph generation |
| Ingest | `/api/ingest` (admin role) | channel sync jobs, per-video sync, job logs (incl. SSE log streaming) |
| Dashboard | `/api/dashboard` | aggregate stats |
| Admin | `/api/admin` (admin role) | storage management, reindexing, cleanup |
| Admin users | `/api/admin/users` (admin role) | full user lifecycle management |
| Provenance | `/api/provenance` (admin role) | cross-user decision/trace feed |

Any response can be machine-translated on the fly with `?lang=` / `?region=` (when `TRANSLATION_ENABLED`).

## Repository layout

```
backend/app/
  main.py          FastAPI app: routers, auth deps, health, /metrics, lifespan
  api/             One router module per feature area
  services/        Business logic (job runners, stores, auth, LLM, storage, mail)
  pipeline/        Video-job stage modules + S3 helpers
  worker/          Celery app, queues, tasks, beat schedule
  core/            settings.py (pydantic-settings), auth, paths, metrics, taxonomy
  db/              Postgres pool + repositories + idempotent schema
  schemas/         Pydantic request/response models
  middleware/      Response-translation middleware
scripts/           Numbered pipeline scripts (01 download → 11 subtitles) + migrations
config/            Prompts, taxonomy, themes, video list, branding assets
db/                init SQL + migrations (incl. the auth migration runbook)
data/              Runtime data (transcripts, fragments, embeddings, chroma, …) — NOT tracked
docker/            Dockerfile built by CI — the only one that ships
docs/              Process, design, security, runbooks and reference (see docs/README.md)
tests/             pytest suite (unit + api)
k6/                k6 load tests and reports
examples/local-dev/  Reference only — superseded k8s/compose manifests, NOT the deployment path
```

`data/` is deliberately untracked. S3 is the source of truth — use [scripts/sync_data_to_s3.py](scripts/sync_data_to_s3.py) and [scripts/download_data_from_s3.py](scripts/download_data_from_s3.py) to move it.

## Getting started (local development)

Prerequisites: Python 3.10+, `ffmpeg`, Node.js 20+ (for yt-dlp), and Docker for the local infrastructure.

```bash
# Postgres on :15432 and ChromaDB on :8001
docker compose -f docker-compose.infra.yml up -d

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# create a .env at the repo root — see Configuration below

./start.sh --local          # API on http://localhost:8077 + Celery worker
./start.sh --local lan      # same, exposed on your LAN
./start.sh --ngrok          # same, plus an ngrok tunnel
```

`start.sh` starts uvicorn (with auto-reload in dev mode) and a Celery worker side by side. Swagger UI: `http://localhost:8077/docs`.

Redis is optional locally — without it the API falls back to running video jobs in a thread.

Run the tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

Some tests currently require live infrastructure and some are stale — see [Known gaps](#known-gaps).

## Configuration

Settings are defined in [backend/app/core/settings.py](backend/app/core/settings.py) (pydantic-settings) and read from the environment / `.env`. The important ones:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string — all pipeline **and** auth tables |
| `API_KEY` | Enables `X-API-Key` enforcement on protected routes. Blank = open dev mode |
| `ALLOWED_ORIGINS` | CORS origins. Must be explicit (not `*`) wherever cookie auth is used |
| `JWT_SIGNING_KEY` | HS256 key for user access tokens |
| `REDIS_URL` | Celery broker/backend. Unset/unreachable → thread-fallback mode |
| `OLLAMA_BASE_URL` | LLM + embedding inference host |
| `STORYLINE_LLM_BACKEND` | `ollama` (default) or `vllm` for the storyline strong model |
| `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT` | `local` persistent client or `http` remote server |
| `AI4IA_STORAGE_BACKEND` | `local` or `s3` (required for split API/worker deployments) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AI4IA_OBJECT_STORAGE_BUCKET` | S3 credentials/bucket when backend is `s3` |
| `MAIL_SMTP_*` / `MAIL_FROM_ADDRESS` | SMTP for OTP emails |
| `YOUTUBE_COOKIES_FILE` | Netscape-format cookies.txt for yt-dlp — required on datacenter IPs |
| `TRANSLATION_ENABLED` | Enables the response-translation middleware |
| `ENABLE_AUTO_SYNC` / `AUTO_SYNC_CRON` | Nightly YouTube channel auto-sync via Celery beat |
| `AI4IA_ENABLE_HLS_STREAMING` | Progressive HLS streaming during export (stage 09) |
| `AUTH_SUPABASE_URL` / `AUTH_SUPABASE_SERVICE_KEY` | **Migration only.** Read-only access to the retained Supabase backup, used by `scripts/migrate_auth_data_from_supabase.py`. The application never reads these |

On startup the app logs a warning for every production-relevant variable that is missing (`validate_deployment_env`), so check the first lines of the log after any config change.

> **Note:** these are warnings, not hard failures — the app boots regardless. An unset `API_KEY` in particular disables the shared-key gate rather than refusing to start.

## Observability

- `GET /api/health` — dependency checks (Redis, Postgres, Chroma) + last-known LLM host state + current worker mode (`celery` vs `thread_fallback`).
- `GET /metrics` — Prometheus metrics: RED metrics per endpoint plus dependency-up gauges refreshed every 60 s. Unauthenticated.
- Structured logs — set `LOG_FORMAT=json` for JSON logs with `job_id`, `stage` and trace-id fields on every record.
- Tracing — OpenTelemetry is gated on **`JAEGER_URL`**, not `OTEL_EXPORTER_OTLP_ENDPOINT` (which is set in `.env` but read by nothing). `JAEGER_URL` is currently unset in the deployed environment, so **tracing is off in production**. See [Known gaps](#known-gaps).
- Per-job, per-stage log files and an event stream are exposed through the API for both video and ingest jobs.

## Deployment

Deployment is driven **only** by [.github/workflows/main.yaml](.github/workflows/main.yaml), which dispatches to a reusable workflow in the `gaiangroup/access` repository on a push to a recognised branch. That workflow builds **[docker/Dockerfile](docker/Dockerfile)** and deploys to the vCluster/Kubernetes tenant.

Consequences worth knowing:

- The workflow runs **no tests, no linting and no image scan** before deploying.
- The manifests and compose files under [examples/local-dev/](examples/local-dev/) deploy nothing. They describe a superseded VM topology and are kept for reference only.
- `docker-compose.infra.yml` at the repo root is local development infrastructure, not deployment.
- There is exactly one Dockerfile in play: **`docker/Dockerfile`**. The old repo-root one has been retired to [examples/local-dev/](examples/local-dev/) so there is no ambiguity about which image ships. It still holds one line worth porting — see [Known gaps](#known-gaps).

## Known gaps

Tracked in the release audit spreadsheet (kept outside git, in `local/reports/`). The ones most likely to surprise you:

| Gap | Effect |
|---|---|
| The deployed worker entrypoint consumes only `video_jobs` | Ingest jobs are accepted, enqueued, and never executed — silently, with no timeout or alert |
| `docker/Dockerfile` has no Whisper pre-download | Workers fetch the model at runtime on first transcription. The retired `examples/local-dev/Dockerfile` has the one-line fix |
| `require_api_key` returns successfully when `API_KEY` is unset | An unset env var disables the shared-key gate rather than failing closed |
| No LLM concurrency limit anywhere | Retries are fixed-interval and correlated; bursts fail together |
| Test suite is red and not run in CI | A bare `pytest` does not collect; some tests need live infrastructure |
| `opentelemetry-api`/`-sdk` are unpinned | A clean image build can resolve an incompatible version |
| No worker health endpoint or queue-depth metric | A worker consuming the wrong queue is indistinguishable from a healthy one |
| Tracing is gated on `JAEGER_URL`, which nothing sets | `OTEL_EXPORTER_OTLP_ENDPOINT` in `.env` is read by no code; traces are not being exported |

`user_sessions` had no schema definition in this repository until the auth migration added it in `db/migrations/0006_auth_tables_to_postgres.sql`; a fresh database could not previously be built from `db/` alone.
