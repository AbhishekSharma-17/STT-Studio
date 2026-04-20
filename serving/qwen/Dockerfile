# Qwen3-ASR server on vLLM — DGX Spark (sm_121 Blackwell, aarch64) compatible.
#
# NGC 26.03-py3 has native `Qwen3ASRForConditionalGeneration` in its model
# registry — the first NVIDIA tag that does.
#
# BUT: vLLM's `transformers_utils/processors/__init__.py` eagerly imports
# FunASRProcessor, which pulls torchaudio. The NVIDIA container doesn't ship
# torchaudio, and PyPI torchaudio is built against a different torch ABI
# (undefined symbol `torch_dtype_float4_e2m1fn_x2`, an FP4 dtype NVIDIA added
# to their custom torch 2.11.0a0 build). Installing torchaudio fails at import.
#
# We don't use FunASR. So we patch the eager import to a lazy/optional one.
# This keeps us on the NVIDIA-shipped torch stack (CUDA 13, sm_121) and avoids
# any PyPI torch/torchaudio wheels that would break.
FROM nvcr.io/nvidia/vllm:26.03-py3

RUN python - <<'PY'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processors/__init__.py")
src = p.read_text()
# Guard the FunASR (torchaudio) and FireRedASR2 imports behind try/except so
# they don't hard-fail if their optional audio deps are missing.
replacement = '''from vllm.transformers_utils.processors.bagel import BagelProcessor
from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
try:
    from vllm.transformers_utils.processors.fireredasr2_processor import (
        FireRedASR2Processor,
    )
except Exception:
    FireRedASR2Processor = None  # torchaudio not installed
try:
    from vllm.transformers_utils.processors.funasr_processor import FunASRProcessor
except Exception:
    FunASRProcessor = None  # torchaudio not installed
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import HunYuanVLImageProcessor
from vllm.transformers_utils.processors.ovis import OvisProcessor
from vllm.transformers_utils.processors.ovis2_5 import Ovis2_5Processor
'''
old = '''from vllm.transformers_utils.processors.bagel import BagelProcessor
from vllm.transformers_utils.processors.deepseek_vl2 import DeepseekVLV2Processor
from vllm.transformers_utils.processors.fireredasr2_processor import (
    FireRedASR2Processor,
)
from vllm.transformers_utils.processors.funasr_processor import FunASRProcessor
from vllm.transformers_utils.processors.hunyuan_vl import HunYuanVLProcessor
from vllm.transformers_utils.processors.hunyuan_vl_image import HunYuanVLImageProcessor
from vllm.transformers_utils.processors.ovis import OvisProcessor
from vllm.transformers_utils.processors.ovis2_5 import Ovis2_5Processor
'''
assert old in src, "processors/__init__.py layout changed — update the patch"
p.write_text(src.replace(old, replacement))
print("patched:", p)
PY

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD curl -fsS http://127.0.0.1:${QWEN_PORT:-8000}/v1/models || exit 1

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
