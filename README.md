# AI4IA Video Studio — Backend

Backend API and job pipeline for the **AI4IA Demo Video Studio**. It ingests the AI4IA
conference video archive from YouTube, indexes it into searchable, speaker-attributed
transcript fragments, and turns a natural-language prompt into a fully edited,
documentary-style video (storyline → assets → render → subtitles).

**Stack:** FastAPI · Celery · Redis · Postgres · ChromaDB · Ollama / vLLM · S3 · ffmpeg

---

## Contents

1. [Quick start](#quick-start)
2. [How it works](#how-it-works)
3. [Pipelines](#pipelines)
4. [Job lifecycle: queues, recovery, cancellation](#job-lifecycle-queues-recovery-cancellation)
5. [API surface](#api-surface)
6. [Repository layout](#repository-layout)
7. [Configuration](#configuration)
8. [Development guide](#development-guide)
9. [Delivery process and environments](#delivery-process-and-environments)
10. [Observability](#observability)
11. [Known gaps](#known-gaps)
12. [Further documentation](#further-documentation)

---

## Quick start

**Prerequisites:** Python 3.10+ (3.12 recommended locally), `ffmpeg`, Node.js 20+ (needed by
yt-dlp), Docker, and access to an Ollama host for LLM and embedding calls.

```bash
# 1. Local infrastructure: Postgres on :15432, ChromaDB on :8001
docker compose -f docker-compose.infra.yml up -d
#    Redis is also required to create video jobs (see "Job lifecycle").

# 2. Python environment (.venv is what start.sh activates)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"

# 3. Configuration
cp .env.example .env        # then fill in the values (see "Configuration")

# 4. Run the API (:8077) and a Celery worker side by side
./start.sh --local
```

Open **http://localhost:8077/docs** for the interactive API docs, and
**http://localhost:8077/api/health** for a dependency check.

Other ways to start:

```bash
./start.sh --local lan               # bind 0.0.0.0, reachable on your LAN
./start.sh --local prod              # 4 uvicorn workers, no auto-reload
CONCURRENCY=2 FORCE_PORT=1 ./start.sh --local
```

Pipeline working data (`data/`) is not in git. Pull what you need from S3 with
[scripts/download_data_from_s3.py](scripts/download_data_from_s3.py).

---

## How it works

```
                 ┌──────────────────────┐
  Frontend ────▶ │  FastAPI (app.main)  │────▶ Postgres: jobs, storylines, insights, auth
                 │  REST + SSE + HLS    │────▶ ChromaDB: archive search
                 └─────────┬────────────┘
                           │ enqueue (Celery)
                           ▼
                 ┌──────────────────────┐
                 │  Redis (broker)      │
                 └─────────┬────────────┘
                           ▼
                 ┌──────────────────────┐      ┌────────────────┐
                 │  Celery worker       │────▶ │ ChromaDB       │ semantic search
                 │  runs the pipeline   │────▶ │ Ollama / vLLM  │ LLM + embeddings
                 │  stage by stage      │────▶ │ S3             │ job artifacts, media
                 └──────────────────────┘      └────────────────┘
```

- **API process** ([backend/app/main.py](backend/app/main.py)): serves the REST API, validates
  and creates jobs, streams job events and logs, serves rendered video (download, signed URLs,
  HLS), and runs the job sweeper.
- **Worker process** ([backend/app/worker/](backend/app/worker/)): runs the long pipelines.
  Video jobs go to the queue named by `WORKER_QUEUE`, and ingest jobs go to `ingest_jobs`.
- **Storage**: `AI4IA_STORAGE_BACKEND=local` uses the shared `data/` directory on one machine.
  `s3` is for split API/worker deployments and is what every deployed environment uses.
- **Persistence**: one Postgres database (`DATABASE_URL`) holds both the pipeline tables and
  the auth tables. The schema is applied idempotently at API startup by
  [backend/app/db/schema.py](backend/app/db/schema.py), which is the source of truth.
  [db/migrations/](db/migrations/) holds reviewable copies.

---

## Pipelines

### Video generation (a "video job")

Triggered by `POST /api/video-jobs` with a prompt, target duration and style. The whole run is
`run_video_job` in [video_job_runner.py](backend/app/services/video_job_runner.py), which
calls the stage modules in [backend/app/pipeline/](backend/app/pipeline/) in this order:

| Stage | Module | What it does |
|---|---|---|
| Guardrail | `stage_guardrail` | Content-policy check on the prompt before any other work |
| Resolve | `stage_resolve` | Intent layer: classifies the prompt (speaker, theme, overview, panel…), checks that the archive covers it, and declines early when it can't |
| Input validation | `stage_input_validation` | Validates the request and environment |
| 07 Storyline | `stage_07_storyline` | Retrieval over fragment embeddings, then a strong LLM plans a structured storyline |
| 07b Validate | `stage_07b_validate` | LLM editorial validation and repair of the storyline |
| 08 Assets | `stage_08_assets` | Title and section cards, TTS narration, overlays |
| 09 Export | `stage_09_export` | Cuts and renders the final video with ffmpeg (optional progressive HLS) |
| 10 Validate | `stage_10_validate` | Automated QA of the exported video |
| 11 Subtitles | `stage_11_subtitles` | Final SRT/VTT subtitles |

- **Storyline cache.** Unless the request sets `force_refresh`, a cache hit skips 07 and 07b.
- **Model warm-up.** Before any stage runs, the worker waits (up to `MODEL_WARM_DEADLINE_SECONDS`)
  for the LLM and embedding hosts to respond.
- **Events and progress.** Each stage emits structured events (`GET /api/video-jobs/{id}/events`),
  progress, per-stage logs and LLM trace rows for provenance.

Design detail: [docs/design/intent-layer-design.md](docs/design/intent-layer-design.md).

### Ingestion (a "sync job")

Triggered by `POST /api/ingest/sync` (admin), or nightly by Celery beat when
`ENABLE_AUTO_SYNC=true`. It runs scripts `01`–`06` in [scripts/](scripts/): download → clean
audio → transcribe (faster-whisper) → create fragments → classify → embed into ChromaDB.
Failed videos can be retried without rediscovering the channel.

yt-dlp needs a **Node.js runtime** (for YouTube's JS challenge) and, from a datacenter IP, a
**cookies file** (`YOUTUBE_COOKIES_FILE`). Without either, every video is reported as
"not available" rather than raising an error.

Design detail: [docs/design/ingest-storage-design.md](docs/design/ingest-storage-design.md).

---

## Job lifecycle: queues, recovery, cancellation

| | Redis reachable | Redis unreachable |
|---|---|---|
| Create a video job | Enqueued to `WORKER_QUEUE` | **HTTP 503**, no fallback |
| Create an ingest job | Enqueued to `ingest_jobs` | Runs in a thread in the API process |

- **Queues are per environment.** Environments share one Redis, so each sets its own
  `WORKER_QUEUE` on both the API and the worker. `start.sh` starts a worker on `video_jobs`
  only. To run ingest jobs locally, start a second worker:
  ```bash
  cd backend && celery --app app.worker.celery_app:celery_app worker --queues ingest_jobs --concurrency 1
  ```
- **Leases and heartbeats.** A delivery may run only a queued job with a matching dispatch id.
  While the job runs, the worker stamps a heartbeat every 30 s.
- **Retry on worker loss.** If the worker dies (OOM kill, deploy, node drain), the job reruns
  from the start, up to `JOB_MAX_WORKER_LOSS_RETRIES` times (default 3; 0 turns retries off).
  The user sees an `interrupted` event. A second loss in the same stage, or running out of
  retries, fails the job as `worker_lost_repeatedly`. Stage errors, cancellation and the
  runtime cap are never retried.
- **Sweeper.** It runs in the API, with one leader per queue. It requeues or fails jobs with no
  heartbeat for 5 min, and fails jobs past the 35 min runtime cap and jobs queued for more than
  30 min. It also re-sends queued jobs whose message was lost.
- **Cancellation** is a database flag. `DELETE /api/video-jobs/{id}` sets `cancelled`, and the
  worker notices within seconds and kills the stage's process group.

Design detail: [docs/design/job-recovery-plan.md](docs/design/job-recovery-plan.md).

---

## API surface

All routes are under `/api` (interactive docs at `/docs`). User auth is short-lived JWT
access tokens plus rotating refresh-token cookies. Protected routes also need an `X-API-Key`
header when `API_KEY` is set.

| Area | Prefix | Highlights |
|---|---|---|
| Auth | `/api/auth` | Signup with email OTP, login, refresh, logout, password reset, persona upgrade |
| Video jobs | `/api/video-jobs` | Create, list and cancel jobs; events, storyline, validation report, provenance, result |
| Video media | `/api/video-jobs/{id}/video` | Stream, download, signed URL, HLS, subtitles, thumbnail, timeline |
| Archive | `/api/archive` | Semantic search, speakers, themes, fragments, source-video streaming |
| Insights | `/api/insights` | LLM-generated insights, sources, provenance, feed |
| Compare / Relay | `/api/compare`, `/api/synthesis`, `/api/relay` | Cross-speaker comparison, theme synthesis, quote relays |
| Concept mindmap | `/api/concept-mindmap` | Concept-graph generation |
| Feedback | `/api/feedback` | Submit feedback; admins list and resolve it |
| Notifications | `/api/notifications` | The signed-in user's notifications |
| Tours | `/api/tours` | Which product tours a user has seen |
| Dashboard | `/api/dashboard` | Aggregate stats |
| Ingest | `/api/ingest` (admin) | Channel sync jobs, per-video sync, logs (SSE) |
| Admin | `/api/admin`, `/api/admin/users`, `/api/admin/data` (admin) | Storage, reindexing, users, read-only data explorer |
| Provenance | `/api/provenance` (admin) | Cross-user decision and trace feed |

Responses can be machine-translated with `?lang=` / `?region=` when `TRANSLATION_ENABLED` is on.

---

## Repository layout

```
backend/app/
  main.py          FastAPI app: routers, auth dependencies, health, /metrics, startup
  api/             One router per feature area
  services/        Business logic: job runner, stores, auth, LLM, storage, mail, sweeper
  pipeline/        Video-job stage modules and S3 helpers
  worker/          Celery app, tasks, beat schedule
  core/            settings.py, auth, paths, metrics, tracing
  db/              Postgres pool, repositories, idempotent schema
  schemas/         Pydantic request/response models
  middleware/      Response translation
scripts/           Standalone pipeline scripts (01 download … 11 subtitles) and maintenance tools
config/            Prompts, taxonomy, themes, speaker registry, branding
db/                Init SQL and reviewable migration copies
docker/            Dockerfile, the only image that ships
docs/              Technical docs (see docs/README.md)
tests/             pytest suite: unit, api, integration
k6/                Load tests
examples/local-dev/  Superseded manifests, kept for reference only; deploys nothing
data/              Pipeline working data; not tracked (S3 is the source of truth)
local/             Your non-technical working files; gitignored
```

---

## Configuration

All settings are typed fields on `Settings` in
[backend/app/core/settings.py](backend/app/core/settings.py) (pydantic-settings). Each field
maps to an upper-case environment variable, and the values are read from the environment or
`.env`. The ones you are most likely to need:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres: pipeline and auth tables |
| `REDIS_URL` | Celery broker. Required to create video jobs |
| `WORKER_QUEUE` | Video-job queue name. Must be unique per environment |
| `API_KEY` | Enables the `X-API-Key` check. Blank = open dev mode |
| `JWT_SIGNING_KEY` | Signs user access tokens (at least 32 bytes) |
| `ALLOWED_ORIGINS` | CORS origins. Must be explicit wherever cookie auth is used |
| `OLLAMA_BASE_URL` | LLM and embedding host |
| `STORYLINE_LLM_BACKEND` | `vllm` (default) or `ollama` for the storyline model |
| `CHROMA_MODE` / `CHROMA_HOST` / `CHROMA_PORT` | Local persistent client, or a remote Chroma server |
| `AI4IA_STORAGE_BACKEND` | `local` or `s3` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AI4IA_OBJECT_STORAGE_BUCKET` | S3 access, when the backend is `s3` |
| `MAIL_SMTP_*` / `MAIL_FROM_ADDRESS` | SMTP for OTP emails |
| `YOUTUBE_COOKIES_FILE` | Cookies for yt-dlp on datacenter IPs |
| `ENABLE_AUTO_SYNC` / `AUTO_SYNC_CRON` | Nightly channel sync through Celery beat |
| `AI4IA_ENABLE_HLS_STREAMING` | Progressive HLS while stage 09 renders |
| `TRANSLATION_ENABLED` | Response translation |
| `INTENT_ROUTER_ENABLED` / `INTENT_ROUTER_MODEL` / `INTENT_ROUTER_TIMEOUT_SECONDS` | LLM intent router in the resolve stage (default on, the guardrail model, 20 s); it falls back to the rules on any failure |
| `AUTH_SUPABASE_URL` / `AUTH_SUPABASE_SERVICE_KEY` | Read-only access to the retired Supabase auth project, used only to copy auth data into Postgres |
| `AUTH_SUPABASE_CATCHUP` | Copies auth rows that changed in Supabase after the first copy, on every startup (v1.0.0 cutover) |

At startup, `validate_deployment_env()` logs a **warning** for each missing production setting.
The app still boots, so read the first lines of the log after any config change.

---

## Development guide

### Tests

```bash
.venv/bin/python -m pytest                                  # full suite, about 5 minutes
.venv/bin/python -m pytest tests/unit/test_settings.py      # one module
.venv/bin/python -m pytest -k "storyline"                   # by name
```

- The suite has about 2,400 tests, and all of them pass.
- Tests in `tests/integration/` that need Postgres use the local infra on `:15432`, each in a
  throwaway schema, and skip when it isn't running.
- CI runs no tests, so run the full suite before every PR and paste the result into it.
- `ruff` and `mypy` are installed but not configured. Use them on the files you touch, not as a gate.

### Conventions to know before editing

- **Scripts are standalone programs.** Everything in `scripts/` runs as a subprocess through
  `run_script()` in [pipeline/base.py](backend/app/pipeline/base.py) and never imports `app.*`.
  A script reports back through stdout or files, never the database.
- **Long-running work goes through `run_script`.** Otherwise it can't be timed out, cancelled
  or stopped when the job loses its lease.
- **Data access is layered:** `api/` → `services/*_store.py` → `db/repositories/` →
  `db/postgres.py`. Repositories use raw parameterized SQL.
- **Schema changes** go in `db/schema.py` first and must be safe to re-run (`if not exists`).
- **New config** is a typed `Settings` field with a default. Don't read `os.environ` at call sites.
- **Every file-writing stage** handles both storage backends (`local` and `s3`).
- **Auth:** `user_id` always comes from the verified token, never from the request body.

---

## Delivery process and environments

Every change follows [docs/process/README.md](docs/process/README.md):

```
GitHub Issue → feature/<issue#>-<slug> from dev → PR to dev → PR to uat → PR to release
```

- **Gates.** Each PR needs one approval. The uat and release PRs need lead or owner approval.
- **Releases.** Each release gets a SemVer tag on `release`, a [CHANGELOG.md](CHANGELOG.md)
  entry, and a GitHub Release.
- **Stage guides.** Step-by-step, repo-specific instructions for each stage are in
  [docs/runbooks/delivery/](docs/runbooks/delivery/).

| Branch | Deploys to | Notes |
|---|---|---|
| `dev` | Development | Integration branch |
| `uat` | Test service (`ai4ia-backend-test`) | **Shares Postgres, ChromaDB, Redis and S3 with production** |
| `release` | Production | Tagged `vX.Y.Z` |

Because uat and production share their data stores, any schema change, startup seeding or data
repair reaches production as soon as it deploys to uat. Label such work `data-change`.

Deployment is driven only by [.github/workflows/main.yaml](.github/workflows/main.yaml), which
calls a reusable workflow in `gaiangroup/access`. That workflow builds
[docker/Dockerfile](docker/Dockerfile) and deploys it. The worker command and the environment
values live in that external deploy repo.

---

## Observability

- **`GET /api/health`**: Redis, Postgres and Chroma checks, the last known LLM host state, and
  the worker mode.
- **`GET /metrics`**: Prometheus request metrics and dependency gauges. Unauthenticated.
- **Logs**: set `LOG_FORMAT=json` for structured logs carrying `job_id`, `stage` and trace ids.
- **Tracing**: OpenTelemetry is gated on `JAEGER_URL`, not `OTEL_EXPORTER_OTLP_ENDPOINT`.
  `JAEGER_URL` is not set in deployed environments, so tracing is off there.
- **Per job**: stage logs and an event stream, for both video and ingest jobs.

---

## Known gaps

| Gap | Effect |
|---|---|
| The deployed worker consumes only the video queue | Ingest jobs are accepted and enqueued but never run while Redis is up |
| `docker/Dockerfile` does not pre-download the Whisper model | Workers fetch it on first transcription |
| `require_api_key` passes when `API_KEY` is unset | An unset variable disables the shared-key check instead of failing closed |
| No global LLM concurrency limit | Bursts of retries fail together |
| No tests, lint or image scan in CI | Nothing enforces a green suite before deploy |
| `opentelemetry-api` / `-sdk` are unpinned | A clean build can resolve an incompatible version |
| No worker health endpoint or queue-depth metric | A worker on the wrong queue looks healthy |
| Tracing is gated on `JAEGER_URL`, which nothing sets | Traces are not exported |

---

## Further documentation

| Where | What |
|---|---|
| [docs/README.md](docs/README.md) | Index of all technical docs |
| [docs/process/](docs/process/) | Delivery process: issues, PRs, approvals, releases |
| [docs/runbooks/delivery/](docs/runbooks/delivery/) | Stage-by-stage delivery guides for this repo |
| [docs/design/](docs/design/) | Intent layer, ingest storage, job recovery |
| [docs/security/](docs/security/) | Security remediation and the frontend changes it required |
| [docs/reference/](docs/reference/) | Database schemas, video inventory, speaker registry |
| [db/migrations/README_auth_migration.md](db/migrations/README_auth_migration.md) | Supabase → Postgres auth migration |
| [CHANGELOG.md](CHANGELOG.md) | What shipped in each version |
