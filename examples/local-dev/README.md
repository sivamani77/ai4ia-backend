# Reference only — none of this deploys anything

**Nothing in this directory is part of how AI4IA is deployed.**

Deployment goes through `.github/workflows/main.yaml`, which dispatches to a
reusable workflow in the `gaiangroup/access` repository. That workflow builds
**`docker/Dockerfile`** and deploys to the vCluster/Kubernetes tenant. The
manifests and compose files here describe an older VM-and-compose topology that
is no longer used.

They were moved out of the repository root because they were actively
misleading. The clearest example: `k8s/11-worker.yaml` starts the Celery worker
with `-Q video_jobs,ingest_jobs`, which is the *correct* configuration — while
the entrypoint that actually runs (`start.sh`) consumes only `video_jobs`. Any
reviewer reading the manifest would reasonably conclude the queues were wired
correctly, which is precisely why the "ingest jobs are accepted and never
executed" bug went unnoticed. Stale infrastructure-as-code reads as ground
truth.

## What is here

| Path | Was | Status |
|---|---|---|
| `k8s/` | Kubernetes manifests (namespace, configmap, redis, chromadb, storage, api, worker) | Superseded by the `gaiangroup/access` workflow |
| `docker-compose.api.yml` | API VM | Obsolete topology |
| `docker-compose.worker.yml` | Worker VM — consumes only `video_jobs` | Obsolete, and wrong |
| `docker-compose.gpu.yml` | Single-VM GPU variant | Obsolete |
| `Dockerfile.gpu-worker` | GPU worker image (`aswinmobiusdtaas/ai4ia-pipeline:v2.9-gpu`) | Not built by CI |
| `DEPLOYMENT.md` | Two-VM runbook | Describes a deployment that no longer exists |

## What is NOT here

`docker-compose.infra.yml` stays in the repository root. It is the local
development stack (Postgres on `:15432`, ChromaDB on `:8001`), it is actively
used, and it is referenced by `db/migrations/README_auth_migration.md` and the
test instructions.

## Before reusing any of this

Treat every value as stale until checked — image tags, resource limits, env var
names, queue names, service DNS. Several are known wrong:

- `k8s/02-secret.yaml.example` contains a real Supabase project URL.
- `k8s/11-worker.yaml` pins `:v2.9-gpu`, an image CI does not build.
- `DEPLOYMENT.md` documents `-Q video_jobs` only, omitting `ingest_jobs`.
- The auth tables these reference have since moved from Supabase to Postgres
  (see `db/migrations/README_auth_migration.md`).

If you need the real deployment topology, read the reusable workflow in
`gaiangroup/access`, not this directory.
