#!/usr/bin/env bash
set -euo pipefail

MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
REVISION="${WHISPER_REVISION:-main}"
PORT="${WHISPER_PORT:-8001}"
GPU_MEM="${WHISPER_GPU_MEMORY_UTILIZATION:-0.35}"
MAX_SEQS="${WHISPER_MAX_NUM_SEQS:-16}"

# Force FLASH_ATTN: on Blackwell (sm_121) Triton's V1 implementation raises
# NotImplementedError for Whisper's encoder self-attn + encoder/decoder
# cross-attn. FLASH_ATTN supports both on this architecture. Older research
# suggesting TRITON_ATTN applied to T4/V100 only.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "[whisper] starting vLLM serve: model=${MODEL} revision=${REVISION} port=${PORT} gpu_mem=${GPU_MEM}"

# NOTE: --task transcription was removed in vLLM 0.17 (NGC 26.03). vLLM now
# auto-detects Whisper's pipeline as transcription from its architecture.
exec vllm serve "${MODEL}" \
    --revision "${REVISION}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-num-seqs "${MAX_SEQS}" \
    --host 0.0.0.0 \
    --port "${PORT}"
