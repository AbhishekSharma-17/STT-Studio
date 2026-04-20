#!/usr/bin/env bash
# End-to-end smoke: both vLLMs up, backend reachable, REST transcribe works.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

QWEN_URL="${QWEN_URL:-http://localhost:8000}"
WHISPER_URL="${WHISPER_URL:-http://localhost:8001}"
BACKEND_URL="${BACKEND_URL:-http://localhost:3000}"

pass() { echo -e "\033[32m PASS\033[0m  $*"; }
fail() { echo -e "\033[31m FAIL\033[0m  $*"; exit 1; }

echo "== Smoke tests =="

curl -fsS "${QWEN_URL}/v1/models" > /dev/null     && pass "Qwen /v1/models"    || fail "Qwen unreachable at ${QWEN_URL}"
curl -fsS "${WHISPER_URL}/v1/models" > /dev/null  && pass "Whisper /v1/models" || fail "Whisper unreachable at ${WHISPER_URL}"
curl -fsS "${BACKEND_URL}/healthz" > /dev/null    && pass "Backend /healthz"   || fail "Backend unreachable at ${BACKEND_URL}"
curl -fsS "${BACKEND_URL}/readyz" > /dev/null     && pass "Backend /readyz"    || fail "Backend not ready (vLLMs unreachable from backend?)"

# Optional: REST transcribe if a sample is present
SAMPLE="${1:-data/samples/sample_ar.wav}"
if [[ -f "${SAMPLE}" ]]; then
    for MODEL in qwen3-asr whisper; do
        RESP=$(curl -fsS -X POST "${BACKEND_URL}/transcribe" \
            -F "file=@${SAMPLE}" \
            -F "model=${MODEL}" \
            -F "language=ar")
        echo "  ${MODEL}: ${RESP}" | head -c 200
        echo
        pass "Backend /transcribe via ${MODEL}"
    done
else
    echo "  (skip /transcribe — no sample at ${SAMPLE})"
fi

echo "== All smoke checks passed =="
