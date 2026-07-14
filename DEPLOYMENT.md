# Deployment — Two-VM Setup (API VM + Worker VM)

Production runs one shared Docker image on two VMs:

| VM | Compose file | Services | Exposed ports |
|---|---|---|---|
| **VM1 — API VM** | [docker-compose.api.yml](docker-compose.api.yml) | `api` (FastAPI/uvicorn), `redis` (Celery broker) | `8000` (API), `6379` (Redis, for the worker VM) |
| **VM2 — Worker VM** | [docker-compose.worker.yml](docker-compose.worker.yml) | `worker` (Celery, `video_jobs` queue), `chromadb` | `8001` (Chroma, for the API VM) |

How the two halves talk to each other:

- The **worker connects to Redis on the API VM** (`REDIS_URL=redis://<API_VM_IP>:6379/0`) to pick up jobs the API enqueues.
- The **API connects to ChromaDB on the worker VM** (`CHROMA_MODE=http`, `CHROMA_HOST=<WORKER_VM_IP>`, `CHROMA_PORT=8001`) for semantic search.
- Job artifacts (exports, HLS segments, generated assets) are **shared via S3** (`AI4IA_STORAGE_BACKEND=s3`): the worker renders and uploads, the API serves. The two VMs do **not** share a filesystem — S3 mode is required, not optional, in this topology.
- Everything else (Supabase, Ollama/vLLM, SMTP) is an external service both VMs reach directly.

---

## 1. Build and push the image

Both compose files pin the same image tag (currently `aswinmobiusdtaas/ai4ia-pipeline:v3.9`). To ship a new version, build from the repo root, push, then bump the tag in **both** compose files:

```bash
# on your build machine, from the repo root
docker build -t aswinmobiusdtaas/ai4ia-pipeline:v3.10 .
docker push aswinmobiusdtaas/ai4ia-pipeline:v3.10
```

Then update the `image:` line in `docker-compose.api.yml` and `docker-compose.worker.yml` to the new tag.

Notes on the image ([Dockerfile](Dockerfile)):

- Based on `python:3.10-slim` with `ffmpeg` installed.
- The Whisper `small` model is **baked into the image** at build time, so workers never download it at runtime — expect the build to take a few minutes and the image to be large.
- `PROJECT_ROOT=/app` is set inside the image; all `data/` paths resolve under `/app/data`, which the compose files bind-mount from the host.
- The default command runs the API (`uvicorn app.main:app`); the worker compose file overrides it with the Celery command.

## 2. Prepare each VM

On **both** VMs:

```bash
# Docker + compose plugin
curl -fsSL https://get.docker.com | sh

# Deployment directory — only the compose file(s), .env, and data/ are needed,
# but cloning the repo is the simplest way to keep them in sync:
git clone <repo-url> ai4ia-backend && cd ai4ia-backend
mkdir -p data/jobs data/exports data/generated_assets data/logs
```

Additionally on the **worker VM**:

```bash
mkdir -p data/embeddings data/embeddings_cleaned data/chroma
```

The worker needs the fragment embedding files present in `data/embeddings/` (or in S3 under `raw/embeddings/` — stage 07 downloads them automatically when `AI4IA_STORAGE_BACKEND=s3`), and the Chroma collection in `data/chroma/`. Seed these from your existing archive build (e.g. `scripts/download_data_from_s3.py`) before the first job.

## 3. Configure `.env` on each VM

Each VM gets its own `.env` at the deployment root (same directory as the compose file). The two files differ only in the cross-VM connection variables.

Shared values (both VMs — external services and secrets):

```dotenv
API_KEY=<strong-random-key>
JWT_SIGNING_KEY=<strong-random-key>
ALLOWED_ORIGINS=["https://your-frontend.example.com"]
REFRESH_COOKIE_SECURE=true

SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
AUTH_SUPABASE_URL=...
AUTH_SUPABASE_SERVICE_KEY=...

OLLAMA_BASE_URL=...

MAIL_SMTP_HOST=...
MAIL_SMTP_PORT=587
MAIL_SMTP_USER=...
MAIL_SMTP_PASSWORD=...
MAIL_FROM_ADDRESS=...

# Shared artifact storage — REQUIRED for the two-VM split
AI4IA_STORAGE_BACKEND=s3
AI4IA_OBJECT_STORAGE_BUCKET=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

VM1 (API VM) specifics:

```dotenv
# Redis runs in this VM's compose stack — use the service name
REDIS_URL=redis://redis:6379/0

# Chroma lives on the worker VM
CHROMA_MODE=http
CHROMA_HOST=<WORKER_VM_IP>
CHROMA_PORT=8001
CHROMA_COLLECTION=fragments
```

VM2 (Worker VM) specifics:

```dotenv
# Redis lives on the API VM
REDIS_URL=redis://<API_VM_IP>:6379/0

# Chroma runs in this VM's compose stack — host port 8001 maps to it
CHROMA_MODE=http
CHROMA_HOST=chromadb
CHROMA_PORT=8000
CHROMA_COLLECTION=fragments
```

> Inside the worker's compose network the Chroma service is reachable as `chromadb:8000`; the `8001` host mapping exists for the API VM (and for debugging).

## 4. Lock down the cross-VM ports

Redis and ChromaDB run **without authentication** — they must never be open to the internet. Restrict them to the peer VM (cloud security group or `ufw`):

| Port | On | Allow from |
|---|---|---|
| `6379` (Redis) | API VM | Worker VM IP only |
| `8001` (Chroma) | Worker VM | API VM IP only |
| `8000` (API) | API VM | Your frontend / reverse proxy (put TLS in front — the container serves plain HTTP) |

## 5. Start the stacks

Start the **API VM first** (the worker needs its Redis to register):

```bash
# VM1 — API VM
docker compose -f docker-compose.api.yml up -d
docker compose -f docker-compose.api.yml logs -f api   # watch for missing-env warnings
```

```bash
# VM2 — Worker VM
docker compose -f docker-compose.worker.yml up -d
docker compose -f docker-compose.worker.yml logs -f worker
```

The worker runs `celery -A app.worker.celery_app worker -Q video_jobs -c 2`. If you also want ingestion (YouTube channel sync) jobs to run, they use a separate `ingest_jobs` queue — either edit the `command:` to `-Q video_jobs,ingest_jobs`, or run a second worker service dedicated to that queue.

## 6. Verify

```bash
# From anywhere that can reach the API VM:
curl http://<API_VM_IP>:8000/api/health
```

A healthy split deployment returns:

```json
{
  "status": "ok",
  "checks": { "redis": {"ok": true}, "supabase": {"ok": true}, "chroma": {"ok": true}, "vllm": {"ok": true} },
  "worker_mode": "celery"
}
```

Watch for:

- `"worker_mode": "thread_fallback"` — the API cannot reach Redis; jobs would run inside the API container instead of on the worker VM. Fix `REDIS_URL` / firewall before serving traffic.
- `"chroma": {"ok": false}` — the API can't reach the worker VM's Chroma, or the `fragments` collection is missing (seed `data/chroma/`).
- On the worker VM, `celery inspect ping` is wired into the compose healthcheck — `docker compose ps` should show the worker as `healthy`.

End-to-end smoke test — submit a small job and follow its events:

```bash
curl -X POST http://<API_VM_IP>:8000/api/video-jobs \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "AI governance and safety", "target_duration": 60, "style": "documentary"}'

curl -H "X-API-Key: $API_KEY" http://<API_VM_IP>:8000/api/video-jobs/<job_id>/events
```

## 7. Upgrades

```bash
# Build + push the new tag (step 1), bump both compose files, then on each VM:
git pull                       # or copy the updated compose file
docker compose -f docker-compose.<api|worker>.yml pull
docker compose -f docker-compose.<api|worker>.yml up -d
```

Upgrade the **worker VM first** while it is idle (no running jobs — check `GET /api/video-jobs?status=running`). Celery tasks use `acks_late`, so a job interrupted by a worker restart is redelivered and safely no-ops if it already reached a terminal state, but letting jobs drain is cleaner.

Redis uses a named volume with AOF persistence, so queued jobs survive an API-VM restart.

## Variants

- **Single GPU VM** — [docker-compose.gpu.yml](docker-compose.gpu.yml) runs api + redis + worker + chroma on one machine, with the worker using the [Dockerfile.gpu-worker](Dockerfile.gpu-worker) image (`:vX.Y-gpu`, CUDA + NVENC ffmpeg) for GPU transcription and encoding.
- **Kubernetes** — equivalent manifests live in [k8s/](k8s/) (namespace, configmap, secret template, redis, chroma, shared storage, api, worker).
- **Bare-metal dev** — `./start.sh --local` runs uvicorn + a Celery worker from a venv; see the [README](README.md).
