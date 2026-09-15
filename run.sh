#!/usr/bin/env bash
# Start the API and the UI together.
#   ./run.sh          both (API on :8000, UI on :8501)
#   ./run.sh api      API only
#   ./run.sh ui       UI only
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env found; copying .env.example (add your GROQ_API_KEY)"; cp .env.example .env; }

start_api() { uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"; }
start_ui()  { streamlit run frontend/streamlit_app.py --server.port 8501; }

case "${1:-both}" in
  api) start_api --reload ;;
  ui)  start_ui ;;
  both)
    start_api &
    API_PID=$!
    trap 'kill $API_PID 2>/dev/null || true' EXIT
    # Wait for /health before launching the UI so the first render is populated.
    for _ in {1..30}; do
      curl -sf http://localhost:8000/health >/dev/null && break
      sleep 1
    done
    start_ui
    ;;
  *) echo "usage: $0 [api|ui|both]"; exit 1 ;;
esac
