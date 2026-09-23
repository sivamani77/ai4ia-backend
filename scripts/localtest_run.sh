#!/usr/bin/env bash
# Run a command against the local rehearsal stores only.
#
#   scripts/localtest_run.sh .venv/bin/python scripts/migrate_fragments_to_postgres.py
#
# Loads .env.localtest over .env (process env wins over the dotenv file in
# pydantic-settings), then refuses to run if any store resolves to a
# non-local host or if S3 keys are present.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
# shellcheck disable=SC1091
. "$ROOT/.env.localtest"
set +a
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

"$ROOT/.venv/bin/python" - <<'EOF'
import sys
from urllib.parse import urlparse
from app.core.settings import settings

local = {"localhost", "127.0.0.1"}
problems = []
if urlparse(settings.database_url).hostname not in local:
    problems.append(f"DATABASE_URL host {urlparse(settings.database_url).hostname!r}")
if settings.chroma_mode == "http" and settings.chroma_host not in local:
    problems.append(f"CHROMA_HOST {settings.chroma_host!r}")
if settings.ai4ia_storage_backend != "local":
    problems.append(f"AI4IA_STORAGE_BACKEND {settings.ai4ia_storage_backend!r}")
if settings.ai4ia_object_storage_access_key:
    problems.append("S3 access key is set")
if problems:
    sys.exit("localtest guard: refusing to run: " + "; ".join(problems))
print(f"localtest guard: ok (pg={urlparse(settings.database_url).netloc.split('@')[-1]}, "
      f"chroma={settings.chroma_host}:{settings.chroma_port}, storage=local)", file=sys.stderr)
EOF

cd "$ROOT"
exec "$@"
