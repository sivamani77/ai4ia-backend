#!/usr/bin/env bash
# Run the AI4IA backend locally.
#
#   ./run_local.sh           # dev mode, localhost only
#   ./run_local.sh lan       # dev mode, exposed on local network (same WiFi/LAN)
#   ./run_local.sh prod      # prod mode, all interfaces
#   FORCE_PORT=1 ./run_local.sh lan   # force-take the port, then start LAN mode
#
# Then open http://localhost:8077/docs  (or the LAN IP printed by lan/prod mode)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT/backend"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/.venv/bin/activate"

# Use the working remote Ollama by default; override by exporting OLLAMA_BASE_URL.
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama-mobius-sales.mobiusdtaas.ai}"

MODE="${1:-dev}"
PORT="${PORT:-8077}"
START_NGROK="${START_NGROK:-0}"
FORCE_PORT="${FORCE_PORT:-0}"
NGROK_ADDR="${NGROK_ADDR:-http://127.0.0.1:4040}"

UVICORN_PID=""
NGROK_PID=""

cleanup() {
  if [ -n "${NGROK_PID:-}" ] && kill -0 "$NGROK_PID" 2>/dev/null; then
    kill "$NGROK_PID" 2>/dev/null || true
  fi

  if [ -n "${UVICORN_PID:-}" ] && kill -0 "$UVICORN_PID" 2>/dev/null; then
    kill "$UVICORN_PID" 2>/dev/null || true
  fi
}

check_port_available() {
  if ! command -v lsof >/dev/null 2>&1; then
    return
  fi

  local listeners
  listeners="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$listeners" ]; then
    return
  fi

  if [ "$FORCE_PORT" = "1" ]; then
    local pids
    pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
    if [ -n "$pids" ]; then
      echo "Port $PORT is in use. Stopping listener PID(s): $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 1
      listeners="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
      if [ -z "$listeners" ]; then
        return
      fi
    fi
  fi

  echo "Port $PORT is already in use:"
  echo "$listeners"
  echo "Stop the existing process or run with a different port, for example: PORT=8078 $0 $MODE"
  echo "To force this script to take over the port, run: FORCE_PORT=1 $0 $MODE"
  exit 1
}

ensure_uvicorn_started() {
  if kill -0 "$UVICORN_PID" 2>/dev/null; then
    return
  fi

  wait "$UVICORN_PID" || true
  echo "Backend failed to start."
  echo "Port $PORT is likely already in use. Stop the existing process or run with a different port, for example: PORT=8078 $0 $MODE"
  exit 1
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
      echo "Swagger docs: $public_url/docs"
      return
    fi

    sleep 1
  done

  echo "ngrok started, but the public URL was not available yet."
  echo "Check the ngrok local UI at $NGROK_ADDR"
}

trap cleanup EXIT INT TERM
check_port_available

get_lan_ip() {
  # Pick the first non-loopback IPv4 address
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -v '^127\.' | head -n 1 || echo "<your-lan-ip>"
}

if [ "$MODE" = "prod" ]; then
  # Multiple workers so a 60-90s LLM request doesn't block all other API calls.
  WORKERS="${WEB_CONCURRENCY:-4}"
  HOST="0.0.0.0"
  LAN_IP="$(get_lan_ip)"
  UVICORN_ARGS=(app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS")
  echo "Starting AI4IA backend (prod): $WORKERS workers"
  echo "  Local:   http://localhost:$PORT/docs"
  echo "  Network: http://$LAN_IP:$PORT/docs"
elif [ "$MODE" = "lan" ]; then
  HOST="0.0.0.0"
  LAN_IP="$(get_lan_ip)"
  UVICORN_ARGS=(app.main:app --host "$HOST" --port "$PORT" --reload)
  echo "Starting AI4IA backend (lan/dev): auto-reload enabled"
  echo "  Local:   http://localhost:$PORT/docs"
  echo "  Network: http://$LAN_IP:$PORT/docs"
else
  HOST="127.0.0.1"
  UVICORN_ARGS=(app.main:app --host "$HOST" --port "$PORT" --reload)
  echo "Starting AI4IA backend (dev): http://localhost:$PORT/docs  (auto-reload)"
fi

uvicorn "${UVICORN_ARGS[@]}" &
UVICORN_PID=$!

sleep 2
ensure_uvicorn_started

if [ "$START_NGROK" = "1" ]; then
  if command -v ngrok >/dev/null 2>&1; then
    echo "Starting ngrok tunnel for port $PORT"
    ngrok http "$PORT" --log=stdout >"$PROJECT_ROOT/.ngrok.log" 2>&1 &
    NGROK_PID=$!
    print_ngrok_url
  else
    echo "ngrok is not installed, so only the backend was started."
  fi
else
  echo "START_NGROK=0, skipping ngrok tunnel."
fi

wait "$UVICORN_PID"
