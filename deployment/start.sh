#!/usr/bin/env bash
set -euo pipefail

export BACKEND_SEARCH_URL="${BACKEND_SEARCH_URL:-http://127.0.0.1:8781/api/search}"
export LANGCHAIN_TRACING_V2="${LANGCHAIN_TRACING_V2:-false}"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8781}"
CHAINLIT_HOST="${CHAINLIT_HOST:-0.0.0.0}"
CHAINLIT_PORT="${CHAINLIT_PORT:-7860}"

echo "[start] launching FastAPI backend on ${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

cleanup() {
  if kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[start] waiting for backend health (text index ready)..."
for attempt in $(seq 1 45); do
  if python - <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8781/api/health", timeout=3) as resp:
        if resp.status != 200:
            sys.exit(1)
        payload = json.loads(resp.read().decode("utf-8"))
        checks = payload.get("checks") or {}
        sys.exit(0 if checks.get("text_index") else 1)
except Exception:
    sys.exit(1)
PY
  then
    echo "[start] backend is healthy"
    break
  fi
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[start] backend process exited unexpectedly"
    exit 1
  fi
  sleep 2
  if [ "${attempt}" -eq 45 ]; then
    echo "[start] backend health check timed out"
    exit 1
  fi
done

echo "[start] launching Chainlit on ${CHAINLIT_HOST}:${CHAINLIT_PORT}"
exec chainlit run frontend/app.py --host "${CHAINLIT_HOST}" --port "${CHAINLIT_PORT}"
