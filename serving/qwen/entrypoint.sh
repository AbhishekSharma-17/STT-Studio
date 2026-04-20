#!/usr/bin/env bash
set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen3-ASR-1.7B}"
REVISION="${QWEN_REVISION:-main}"
PORT="${QWEN_PORT:-8000}"
GPU_MEM="${QWEN_GPU_MEMORY_UTILIZATION:-0.35}"
MAX_SEQS="${QWEN_MAX_NUM_SEQS:-16}"

echo "[qwen] starting vLLM serve: model=${MODEL} revision=${REVISION} port=${PORT} gpu_mem=${GPU_MEM}"

exec vllm serve "${MODEL}" \
    --revision "${REVISION}" \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-num-seqs "${MAX_SEQS}" \
    --host 0.0.0.0 \
    --port "${PORT}"
