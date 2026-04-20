#!/usr/bin/env bash
# Tail docker compose logs. Optionally scope to a service: qwen | whisper | backend.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -gt 0 ]]; then
    exec docker compose logs -f --tail=200 "$@"
else
    exec docker compose logs -f --tail=200
fi
