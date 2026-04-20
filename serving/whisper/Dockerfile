# Whisper-Large-v3 server on vLLM — DGX Spark (sm_121 Blackwell, aarch64) compatible.
# Using 26.03-py3 to match the Qwen container.
FROM nvcr.io/nvidia/vllm:26.03-py3

# Whisper requires librosa for audio preprocessing in vLLM's transcription path
RUN pip install --no-cache-dir librosa soundfile

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD curl -fsS http://127.0.0.1:${WHISPER_PORT:-8001}/v1/models || exit 1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
