#!/usr/bin/env bash
# Start the AI4IA backend + Celery worker.
#
# Usage:
#   ./start.sh --local           # dev mode, localhost only
#   ./start.sh --local lan       # dev mode, exposed on LAN
#   ./start.sh --local prod      # prod mode, all interfaces
#   ./start.sh --ngrok           # dev mode + ngrok tunnel
#   ./start.sh --ngrok prod      # prod mode + ngrok tunnel
#
#   CONCURRENCY=2 ./start.sh --local
#   FORCE_PORT=1  ./start.sh --local lan
#
# The Celery worker is always started alongside the backend.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT/backend"

source "$PROJECT_ROOT/.venv/bin/activate"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama-mobius-sales.mobiusdtaas.ai}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

PORT="${PORT:-8077}"
FORCE_PORT="${FORCE_PORT:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"
NGROK_ADDR="${NGROK_ADDR:-http://127.0.0.1:4040}"

UVICORN_PID=""
WORKER_PID=""
NGROK_PID=""

# ── argument parsing ──────────────────────────────────────────────────────────

if [ $# -lt 1 ] || { [ "$1" != "--local" ] && [ "$1" != "--ngrok" ]; }; then
  echo "Usage: $0 --local [dev|lan|prod]"
  echo "       $0 --ngrok [dev|prod]"
  exit 1
fi

FLAG="$1"; shift
MODE="${1:-dev}"
START_NGROK=0
[ "$FLAG" = "--ngrok" ] && START_NGROK=1

# ── helpers ───────────────────────────────────────────────────────────────────

if [ -f "$PROJECT_ROOT/.env" ]; then
  eval "$(python3 -c "
import shlex
from dotenv import dotenv_values
for k, v in dotenv_values('$PROJECT_ROOT/.env').items():
    if v is not None:
        print(f'export {k}={shlex.quote(v)}')
")"
fi

cleanup() {
  for pid_var in NGROK_PID WORKER_PID UVICORN_PID; do
    pid="${!pid_var:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

check_port_available() {
  if ! command -v lsof >/dev/null 2>&1; then return; fi

  local listeners
  listeners="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  [ -z "$listeners" ] && return

  if [ "$FORCE_PORT" = "1" ]; then
    local pids
    pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [ -n "$pids" ]; then
      echo "Port $PORT in use. Killing PID(s): $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 1
      listeners="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
      [ -z "$listeners" ] && return
    fi
  fi

  echo "Port $PORT is already in use:"
  echo "$listeners"
  echo "Run with a different port: PORT=8078 $0 $FLAG $MODE"
  echo "Or force-take it:          FORCE_PORT=1 $0 $FLAG $MODE"
  exit 1
}

ensure_uvicorn_started() {
  if kill -0 "$UVICORN_PID" 2>/dev/null; then return; fi
  wait "$UVICORN_PID" || true
  echo "Backend failed to start. Port $PORT may already be in use."
  exit 1
}

get_lan_ip() {
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -v '^127\.' | head -n 1 || echo "<your-lan-ip>"
}

print_ngrok_url() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "ngrok started. Install curl to auto-print the public URL."
    return
  fi
  for _ in $(seq 1 20); do
    public_url="$(
      curl -fsS "$NGROK_ADDR/api/tunnels" 2>/dev/null \
        | tr -d '\n' \
        | sed -n 's/.*"public_url":"\([^"]*\)".*/\1/p' \
        | head -n 1
    )" || true
    if [ -n "$public_url" ]; then
      echo "ngrok public URL: $public_url"
      echo "Swagger docs:     $public_url/docs"
      return
    fi
    sleep 1
  done
  echo "ngrok started, but the public URL is not available yet."
  echo "Check the ngrok UI at $NGROK_ADDR"
}

# ── backend ───────────────────────────────────────────────────────────────────

check_port_available

if [ "$MODE" = "prod" ]; then
  WORKERS="${WEB_CONCURRENCY:-4}"
  LAN_IP="$(get_lan_ip)"
  UVICORN_ARGS=(app.main:app --host "0.0.0.0" --port "$PORT" --workers "$WORKERS")
  echo "Starting AI4IA backend (prod): $WORKERS workers"
  echo "  Local:   http://localhost:$PORT/docs"
  echo "  Network: http://$LAN_IP:$PORT/docs"
elif [ "$MODE" = "lan" ]; then
  LAN_IP="$(get_lan_ip)"
  UVICORN_ARGS=(app.main:app --host "0.0.0.0" --port "$PORT" --reload)
  echo "Starting AI4IA backend (lan/dev): auto-reload enabled"
  echo "  Local:   http://localhost:$PORT/docs"
  echo "  Network: http://$LAN_IP:$PORT/docs"
else
  UVICORN_ARGS=(app.main:app --host "127.0.0.1" --port "$PORT" --reload)
  echo "Starting AI4IA backend (dev): http://localhost:$PORT/docs  (auto-reload)"
fi

uvicorn "${UVICORN_ARGS[@]}" &
UVICORN_PID=$!

sleep 2
ensure_uvicorn_started

# ── worker ────────────────────────────────────────────────────────────────────

echo ""
echo "Starting Celery worker"
echo "  Queue:       video_jobs"
echo "  Concurrency: $CONCURRENCY"
echo "  Broker:      $REDIS_URL"
echo ""

celery \
  --app app.worker.celery_app:celery_app \
  worker \
  --queues video_jobs \
  --concurrency "$CONCURRENCY" \
  --loglevel "$LOG_LEVEL" \
  --hostname "worker@%h" &
WORKER_PID=$!

# ── ngrok ─────────────────────────────────────────────────────────────────────

if [ "$START_NGROK" = "1" ]; then
  if command -v ngrok >/dev/null 2>&1; then
    echo "Starting ngrok tunnel for port $PORT"
    ngrok http "$PORT" --log=stdout >/dev/null 2>&1 &
    NGROK_PID=$!
    print_ngrok_url
  else
    echo "ngrok not installed — skipping tunnel."
  fi
fi

wait "$UVICORN_PID"
