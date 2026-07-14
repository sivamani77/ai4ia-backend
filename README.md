# AI4IA Video Studio — Backend

Backend API and job pipeline for the **AI4IA Demo Video Studio**: it ingests the AI4IA conference video archive from YouTube, indexes it into searchable, speaker-attributed transcript fragments, and generates fully edited documentary-style videos (storyline → assets → render → subtitles) from a natural-language prompt.

Built with **FastAPI** + **Celery**, backed by **Redis** (broker), **ChromaDB** (vector search), **Supabase** (persistence + user auth), **Ollama / vLLM** (LLM inference), and optional **S3** object storage.

---

## Architecture

```
                 ┌──────────────────────┐
  Frontend ────▶ │  FastAPI (app.main)  │────▶ Supabase (jobs, storylines, insights)
                 │  REST + SSE + HLS    │────▶ Supabase (auth: users, OTP, sessions)
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
                 │  ingest_jobs queues  │────▶ │ S3 (optional)  │ shared job artifacts
                 └──────────────────────┘      └────────────────┘
```

- **API process** ([backend/app/main.py](backend/app/main.py)) — serves the REST API, validates and creates jobs, streams job events/logs, and serves rendered video (progressive download, signed URLs, HLS).
- **Worker process** ([backend/app/worker/](backend/app/worker/)) — executes the heavy pipelines. Two queues: `video_jobs` (video generation) and `ingest_jobs` (YouTube channel sync). Tasks use `acks_late` so a crashed worker re-queues its job.
- **Thread fallback** — if Redis is unreachable, video jobs run in a background thread inside the API process, and orphaned jobs are reaped on startup. Fine for dev, not for production.
- **Storage backends** — `local` (single machine, shared `data/` directory) or `s3` (split API/worker deployments: the worker mirrors exports, HLS segments, and assets to S3 so the API can serve them from another machine).

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

Every stage emits structured job events (consumable via `GET /api/video-jobs/{id}/events`), progress, per-stage logs, and LLM trace rows for provenance. Jobs are cancellable mid-stage.

### Ingestion (a "sync job")

Triggered by `POST /api/ingest/sync` (or nightly by Celery beat when `ENABLE_AUTO_SYNC=true`). Runs the archive-building scripts `01`–`06` in [scripts/](scripts/): download videos → extract/clean audio → transcribe (faster-whisper) → create fragments → classify → embed into ChromaDB.

## API surface

All routes live under `/api` (interactive docs at `/docs`). Protected routes require an `X-API-Key` header when `API_KEY` is set; user-facing auth is JWT access tokens + rotating refresh-token cookies.

| Area | Prefix | Highlights |
|---|---|---|
| Auth | `/api/auth` | signup + email OTP, login, refresh, logout, password reset |
| Video jobs | `/api/video-jobs` | create/list/cancel jobs, events, storyline, validation report, provenance, assets, result |
| Video media | `/api/video-jobs/{id}/video` | stream, download, signed URL, HLS playlist/segments, subtitles, thumbnail, timeline |
| Archive | `/api/archive` | semantic search, speakers, themes, fragments, source-video streaming |
| Insights | `/api/insights` | LLM-generated insights, sources, provenance, feed |
| Compare / Relay | `/api/compare`, `/api/synthesis`, `/api/relay` | cross-speaker comparison, theme synthesis, quote relays |
| Concept mindmap | `/api/concept-mindmap` | concept-graph generation |
| Ingest | `/api/ingest` | channel sync jobs, per-video sync, job logs (incl. SSE log streaming) |
| Dashboard | `/api/dashboard` | aggregate stats |
| Admin | `/api/admin` (admin role) | storage management, reindexing, cleanup |
| Admin users | `/api/admin/users` (admin role) | full user lifecycle management |

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
  db/              Supabase client + repositories
  schemas/         Pydantic request/response models
  middleware/      Response-translation middleware
scripts/           Numbered pipeline scripts (01 download → 11 subtitles) + migrations
config/            Prompts, taxonomy, themes, video list, branding assets
data/              Runtime data (jobs, exports, embeddings, chroma, …) — not committed
tests/             pytest suite (unit + api)
test/              k6 load tests and reports
k8s/               Kubernetes manifests (alternative to the VM deployment)
```

## Getting started (local development)

Prerequisites: Python 3.10+, `ffmpeg`, and a running Redis (`docker run -p 6379:6379 redis:7-alpine` is enough).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# create a .env at the repo root — see Configuration below

./start.sh --local          # API on http://localhost:8077 + Celery worker
./start.sh --local lan      # same, exposed on your LAN
./start.sh --ngrok          # same, plus an ngrok tunnel
```

`start.sh` starts uvicorn (with auto-reload in dev mode) and a Celery worker on the `video_jobs` queue side by side. Swagger UI: `http://localhost:8077/docs`.

Run the tests:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Configuration

Settings are defined in [backend/app/core/settings.py](backend/app/core/settings.py) (pydantic-settings) and read from the environment / `.env`. The important ones:

| Variable | Purpose |
|---|---|
| `API_KEY` | Enables `X-API-Key` enforcement on protected routes. Blank = open dev mode |
| `ALLOWED_ORIGINS` | CORS origins. Must be explicit (not `*`) wherever cookie auth is used |
| `JWT_SIGNING_KEY` | HS256 key for user access tokens |
| `REDIS_URL` | Celery broker/backend. Unset/unreachable → thread-fallback mode |
| `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` | Pipeline data project (jobs, storylines, insights) |
| `AUTH_SUPABASE_URL` / `AUTH_SUPABASE_SERVICE_KEY` | Separate auth project (users, OTP, sessions) |
| `OLLAMA_BASE_URL` | LLM + embedding inference host |
| `STORYLINE_LLM_BACKEND` | `ollama` (default) or `vllm` for the storyline strong model |
| `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT` | `local` persistent client or `http` remote server |
| `AI4IA_STORAGE_BACKEND` | `local` or `s3` (required for split API/worker deployments) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AI4IA_OBJECT_STORAGE_BUCKET` | S3 credentials/bucket when backend is `s3` |
| `MAIL_SMTP_*` / `MAIL_FROM_ADDRESS` | SMTP for OTP emails |
| `TRANSLATION_ENABLED` | Enables the response-translation middleware |
| `ENABLE_AUTO_SYNC` / `AUTO_SYNC_CRON` | Nightly YouTube channel auto-sync via Celery beat |
| `AI4IA_ENABLE_HLS_STREAMING` | Progressive HLS streaming during export (stage 09) |

On startup the app logs a warning for every production-relevant variable that is missing (`validate_deployment_env`), so check the first lines of the log after any config change.

## Observability

- `GET /api/health` — dependency checks (Redis, Supabase, Chroma, vLLM) + current worker mode (`celery` vs `thread_fallback`).
- `GET /metrics` — Prometheus metrics: RED metrics per endpoint plus dependency-up gauges refreshed every 30 s.
- Structured logs — set `LOG_FORMAT=json` for JSON logs with `job_id` and `stage` fields on every record.
- Per-job, per-stage log files and an event stream are exposed through the API for both video and ingest jobs.

## Deployment

Production runs as a prebuilt Docker image across two VMs (API VM + worker VM) with `docker compose`. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full runbook, including the single-VM GPU variant and the Kubernetes manifests in [k8s/](k8s/).
