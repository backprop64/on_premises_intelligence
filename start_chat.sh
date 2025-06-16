#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start_chat.sh – convenience launcher for OPI demo
#
# • Spawns the FastAPI backend (uvicorn) on port 8000
# • Serves the static frontend via Python's built-in http.server (port 5173)
# • Opens the default browser at http://localhost:5173
#
# Stop the script with Ctrl-C to terminate both background processes.
# ---------------------------------------------------------------------------
set -euo pipefail

# Always resolve to repository root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Backend – FastAPI
# ---------------------------------------------------------------------------
UVICORN_CMD="uvicorn api.opi_api:app --host 0.0.0.0 --port 8000 --log-level info"
echo "[start_chat] Launching API → $UVICORN_CMD"
$UVICORN_CMD &
API_PID=$!

# ---------------------------------------------------------------------------
# Frontend – simple static file server
# ---------------------------------------------------------------------------
FRONTEND_DIR="$ROOT_DIR/frontend"
cd "$FRONTEND_DIR"
HTTP_PORT=5173
SERVER_CMD="python3 -m http.server $HTTP_PORT"
echo "[start_chat] Launching static server → $SERVER_CMD (cwd=$FRONTEND_DIR)"
$SERVER_CMD &
FE_PID=$!

# Give the servers a moment to initialise
sleep 2

# Open UI in default browser (macOS = open, Linux = xdg-open)
URL="http://localhost:$HTTP_PORT/chat.html"
if command -v open &>/dev/null; then
  open "$URL"
elif command -v xdg-open &>/dev/null; then
  xdg-open "$URL" &>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Wait / teardown
# ---------------------------------------------------------------------------
trap "echo '\n[start_chat] Caught signal – shutting down.'; kill $API_PID $FE_PID 2>/dev/null || true" INT TERM

# Block until background jobs exit (Ctrl-C)
wait 