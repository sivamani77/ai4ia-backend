#!/usr/bin/env bash
# Install the backend as a systemd USER service so it starts on login and
# restarts on crash — no root needed.
set -euo pipefail

UNIT_DIR="$HOME/.config/systemd/user"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ai4ia-backend.service"

mkdir -p "$UNIT_DIR"
cp "$SRC" "$UNIT_DIR/ai4ia-backend.service"

systemctl --user daemon-reload
systemctl --user enable --now ai4ia-backend.service

# Keep the service running even when you're logged out (optional but recommended).
loginctl enable-linger "$(whoami)" 2>/dev/null || \
  echo "(could not enable linger; service still runs while you're logged in)"

echo
echo "Installed. Useful commands:"
echo "  systemctl --user status ai4ia-backend       # check it's running"
echo "  journalctl --user -u ai4ia-backend -f        # live logs"
echo "  systemctl --user restart ai4ia-backend       # after code changes"
echo "  systemctl --user stop ai4ia-backend          # stop"
echo
echo "API:  http://localhost:8077/docs"
