#!/usr/bin/env bash
# One-shot first-time setup:
#   1. Ensure `.env` exists (copied from template)
#   2. Verify docker + NVIDIA Container Toolkit
#   3. Install backend Python deps with `uv`
#   4. Pre-pull model weights into ./hf_cache
#
# Usage:
#   bash scripts/setup.sh            # full setup
#   bash scripts/setup.sh --no-models  # skip model download
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SKIP_MODELS=0
for arg in "$@"; do
    case "${arg}" in
        --no-models) SKIP_MODELS=1 ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0 ;;
    esac
done

section() { echo; echo "== $* =="; }

# --- 1. .env --------------------------------------------------------
section "Environment"
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from template."
else
    echo ".env already present — leaving untouched."
fi

# --- 2. docker ------------------------------------------------------
section "Docker"
if ! command -v docker >/dev/null; then
    echo "ERROR: docker not found. Install Docker Engine + NVIDIA Container Toolkit." >&2
    exit 1
fi
docker --version
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable. Start it or add your user to 'docker' group." >&2
    exit 1
fi
# GPU check (non-fatal if absent — useful to know early)
if docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "GPU passthrough OK."
else
    echo "WARNING: GPU passthrough not working. Install NVIDIA Container Toolkit (nvidia-ctk)."
fi

# --- 3. uv + backend deps -------------------------------------------
section "Python (uv)"
if ! command -v uv >/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env" 2>/dev/null || true
fi
uv --version
uv sync --extra dev

# --- 4. Models ------------------------------------------------------
if (( SKIP_MODELS == 0 )); then
    section "Model weights"
    bash scripts/download_models.sh
else
    echo
    echo "(Skipping model download — run 'bash scripts/download_models.sh' when ready.)"
fi

echo
echo "Setup complete. Next: bash scripts/start.sh"
