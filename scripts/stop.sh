#!/usr/bin/env bash
# Stop all services (qwen + whisper + backend). Containers are removed but
# the HF cache on disk and the compose network stay put.
#
# Usage:
#   bash scripts/stop.sh          # stop + remove containers
#   bash scripts/stop.sh --all    # also remove volumes + networks
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

case "${1-}" in
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    --all) docker compose down -v ;;
    "")    docker compose down ;;
    *)     echo "Unknown flag: $1" >&2; exit 2 ;;
esac
