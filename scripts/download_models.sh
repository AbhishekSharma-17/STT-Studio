#!/usr/bin/env bash
# Pre-pull model weights into the shared HF cache so first `make start` doesn't hang.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Load .env if present
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a && source .env && set +a
fi

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-ASR-1.7B}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
HF_CACHE_DIR="${HF_CACHE_DIR:-./hf_cache}"

mkdir -p "${HF_CACHE_DIR}"

echo "[download] HF cache: ${HF_CACHE_DIR}"
echo "[download] models:   ${QWEN_MODEL}, ${WHISPER_MODEL}"

# Pull with huggingface-cli inside the vLLM container so we don't depend on host Python.
docker run --rm \
    -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME=/cache \
    -v "$(realpath "${HF_CACHE_DIR}")":/cache \
    nvcr.io/nvidia/vllm:26.03-py3 \
    bash -c "
        set -e
        # Only pull safetensors + configs. Whisper's HF repo also ships FP32 .bin
        # and PyTorch .bin duplicates that waste ~20 GB if you pull everything.
        INCLUDE=('*.safetensors' '*.json' '*.txt' '*.model' 'tokenizer*' 'preprocessor*' 'generation_config*')
        huggingface-cli download '${QWEN_MODEL}' --revision '${QWEN_REVISION:-main}' \
            \${INCLUDE[@]/#/--include }
        huggingface-cli download '${WHISPER_MODEL}' --revision '${WHISPER_REVISION:-main}' \
            \${INCLUDE[@]/#/--include }
    "

echo "[download] done."
