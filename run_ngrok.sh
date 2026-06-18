#!/usr/bin/env bash
# Run the AI4IA backend locally and expose it via ngrok.
#
#   ./run_ngrok.sh           # dev mode + ngrok tunnel
#   ./run_ngrok.sh prod      # prod mode + ngrok tunnel
#
# Then open the printed ngrok URL or http://localhost:8077/docs
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_NGROK=1 "$PROJECT_ROOT/run_local.sh" "$@"
