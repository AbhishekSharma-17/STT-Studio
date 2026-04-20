#!/usr/bin/env bash
# Bring up the full stack (qwen + whisper + backend) via docker compose.
#
# Usage:
#   bash scripts/start.sh            # production-ish mode
#   bash scripts/start.sh --dev      # hot-reload backend, DEBUG logs
#   bash scripts/start.sh --follow   # tail logs after start
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMPOSE_FILES=(-f compose.yml)
FOLLOW=0
for arg in "$@"; do
    case "${arg}" in
        --dev) COMPOSE_FILES+=(-f compose.dev.yml) ;;
        --follow|-f) FOLLOW=1 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    esac
done

if [[ ! -f .env ]]; then
    echo "No .env found; copying from template."
    cp .env.example .env
fi

echo "Building + starting services (this takes 3–5 minutes on a cold pull)..."
docker compose "${COMPOSE_FILES[@]}" up -d --build

echo
echo "Services:"
docker compose "${COMPOSE_FILES[@]}" ps

cat <<EOF

  Backend UI : http://localhost:3000
  Backend API: http://localhost:3000/docs
  Qwen vLLM  : http://localhost:8000/v1/models
  Whisper    : http://localhost:8001/v1/models

  Wait for vLLM to finish loading weights (~1–3 min). Check status with:
    bash scripts/status.sh
  Or tail logs:
    bash scripts/start.sh --follow   (or: docker compose logs -f)
EOF

if (( FOLLOW )); then
    exec docker compose "${COMPOSE_FILES[@]}" logs -f --tail=100
fi
