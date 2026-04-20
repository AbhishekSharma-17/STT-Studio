#!/usr/bin/env bash
# WER benchmark across both models using data/samples/<id>.wav + <id>.txt pairs.
# Requires services up (bash scripts/start.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SAMPLES_DIR="${SAMPLES_DIR:-data/samples}"
OUT_DIR="${OUT_DIR:-results}"
LANGUAGE="${LANGUAGE:-ar}"
BACKEND_URL="${BACKEND_URL:-http://localhost:3000}"

if ! curl -fsS "${BACKEND_URL}/readyz" >/dev/null; then
    echo "Backend not ready at ${BACKEND_URL}. Run: bash scripts/start.sh" >&2
    exit 1
fi

count=$(find "${SAMPLES_DIR}" -maxdepth 1 -name '*.wav' | wc -l)
if (( count == 0 )); then
    echo "No *.wav pairs in ${SAMPLES_DIR}/. Drop <id>.wav + <id>.txt files in there first." >&2
    exit 1
fi

echo "Benchmarking ${count} samples in ${SAMPLES_DIR} (language=${LANGUAGE}) ..."
exec uv run python scripts/bench_wer.py \
    --backend "${BACKEND_URL}" \
    --samples "${SAMPLES_DIR}" \
    --out "${OUT_DIR}" \
    --language "${LANGUAGE}"
