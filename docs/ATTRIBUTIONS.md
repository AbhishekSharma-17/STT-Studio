# Attributions — Third-Party Components

STT Studio bundles, depends on, or is built on top of the open-source
projects, models, and fonts listed below. The project itself is released
under the **Apache License 2.0** (see [`LICENSE`](../LICENSE)). Everything
else below retains its original licence.

If you redistribute or deploy a derivative of STT Studio, keep this file (or
an equivalent) alongside the binary/image and preserve the licence notices
referenced.

---

## Machine-learning models

### Qwen3-ASR-1.7B
- Repository: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>
- Authors: Alibaba Cloud — Qwen team
- Licence: **Apache License 2.0**
- Role in STT Studio: primary multilingual ASR model (30 languages incl. Arabic).
- Full licence text: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/LICENSE>
  or <https://www.apache.org/licenses/LICENSE-2.0>

### OpenAI Whisper Large-v3
- Repository: <https://huggingface.co/openai/whisper-large-v3>
- Authors: OpenAI
- Licence: **MIT License** (model weights and code)
- Role in STT Studio: secondary multilingual ASR model (99 languages) used for
  A/B comparison and as a fallback.
- Full licence text: <https://github.com/openai/whisper/blob/main/LICENSE>

### Silero VAD (v5)
- Repository: <https://github.com/snakers4/silero-vad>
- Authors: Silero Team
- Licence: **MIT License**
- Role in STT Studio: server-side voice-activity detection on the live
  WebSocket path — chunks the audio stream on natural pauses before each
  utterance is sent to vLLM.
- Full licence text: <https://github.com/snakers4/silero-vad/blob/master/LICENSE>

---

## Inference / serving

### vLLM
- Repository: <https://github.com/vllm-project/vllm>
- Licence: **Apache License 2.0**
- Role in STT Studio: hosts both ASR models and exposes the OpenAI-compatible
  `/v1/audio/transcriptions` HTTP endpoint that the backend calls.
- Notice/licence: see NOTICE file in their repo.

### NVIDIA vLLM Container (`nvcr.io/nvidia/vllm:26.03-py3`)
- Source: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm>
- Licence: **NVIDIA Deep Learning Container License**
  <https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license>
- Role in STT Studio: the runtime image the Qwen and Whisper services are
  built from. Includes CUDA 13, a sm_121 (Blackwell)-compatible vLLM build,
  PyTorch, Triton, and other NVIDIA software components.
- Use of this image is subject to NVIDIA's licence terms; we do not
  redistribute the image itself, only its tag name in Dockerfiles.

### CUDA / cuDNN / TensorRT / etc.
- Publisher: NVIDIA Corporation
- Licence: **NVIDIA CUDA Toolkit EULA** and related per-library licences
  (shipped inside the NVIDIA container). See the licence notices in
  `/usr/share/doc/` inside the image.

---

## Python dependencies (backend)

All installed from PyPI. Versions are pinned in [`pyproject.toml`](../pyproject.toml).

| Package | Licence | Home |
|---|---|---|
| FastAPI | MIT | https://github.com/fastapi/fastapi |
| Starlette | BSD-3-Clause | https://github.com/encode/starlette |
| Uvicorn | BSD-3-Clause | https://github.com/encode/uvicorn |
| Pydantic (v2) | MIT | https://github.com/pydantic/pydantic |
| pydantic-settings | MIT | https://github.com/pydantic/pydantic-settings |
| httpx | BSD-3-Clause | https://github.com/encode/httpx |
| httpcore | BSD-3-Clause | https://github.com/encode/httpcore |
| structlog | Apache-2.0 / MIT dual | https://github.com/hynek/structlog |
| tenacity | Apache-2.0 | https://github.com/jd/tenacity |
| python-multipart | Apache-2.0 | https://github.com/andrew-d/python-multipart |
| websockets | BSD-3-Clause | https://github.com/python-websockets/websockets |
| prometheus-fastapi-instrumentator | ISC | https://github.com/trallnag/prometheus-fastapi-instrumentator |
| prometheus-client | Apache-2.0 | https://github.com/prometheus/client_python |
| silero-vad (PyPI wrapper) | MIT | https://github.com/snakers4/silero-vad |
| PyTorch (`torch`) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| torchaudio | BSD-2-Clause | https://github.com/pytorch/audio |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| PyAV (`av`) | BSD-3-Clause | https://github.com/PyAV-Org/PyAV |

Dev-only (not shipped in the runtime image):

| Package | Licence |
|---|---|
| pytest | MIT |
| pytest-asyncio | Apache-2.0 |
| pytest-cov | MIT |
| respx | BSD-3-Clause |
| ruff | MIT |
| mypy | MIT |
| uv | Apache-2.0 OR MIT |

---

## System / container deps

The backend Docker image (`backend/Dockerfile`) builds on
`python:3.12-slim-bookworm` and installs a few apt packages:

| Package | Licence |
|---|---|
| `ca-certificates` | MPL-2.0 |
| `curl` | MIT-style (curl licence) |
| `ffmpeg` | LGPL-2.1+ (link-time), with some GPL-licensed optional components disabled in the Debian build |

Debian package sources can be obtained from
<https://sources.debian.org/>.

---

## Frontend assets

### Fonts
- **Inter** — Rasmus Andersson — SIL Open Font License 1.1
  (<https://rsms.me/inter/>)
- **JetBrains Mono** — JetBrains — SIL Open Font License 1.1
  (<https://www.jetbrains.com/lp/mono/>)

Both fonts are loaded from Google Fonts at runtime via the `<link>` tag in
`frontend/index.html`. No font files are redistributed inside this
repository.

### JavaScript

The frontend is vanilla JS — no third-party runtime dependencies, no
bundler, no framework. The only remote script resources are the Google
Fonts CSS above.

### Brand assets (proprietary)

- `frontend/assets/genaiprotos-icon.svg`
- `frontend/assets/genaiprotos-wordmark.svg`
- `frontend/assets/genaiprotos-logo.png`
- `frontend/assets/favicon.png`
- Contents of `GenAI Protos Logo/`

These are **trademarks and branding of GenAI Protos**. They are included
in the repo so this deployment can display them. They are **not covered
by the Apache 2.0 licence** of the rest of this project — do not reuse
them in derivative works or unrelated projects without written permission
from GenAI Protos. If you fork STT Studio for your own brand, replace
these files with your own assets.

### Reference audio

`data/samples/jfk.wav` is derived from a short clip of President
John F. Kennedy's 1961 inaugural address, which is in the **public domain**
in the United States as a work of the U.S. Federal Government. It is
included solely for smoke-testing the transcription pipeline.

---

## Trademarks

"GenAI Protos" and associated logos/wordmarks are trademarks of GenAI
Protos. "Qwen" is a trademark of Alibaba Cloud. "Whisper" and "OpenAI" are
trademarks of OpenAI, Inc. "NVIDIA", "DGX Spark", "Grace Blackwell",
"Triton", "CUDA", "cuDNN", and "TensorRT" are trademarks of NVIDIA
Corporation. All other trademarks are the property of their respective
owners.

---

## Updating this file

When adding a new runtime dependency:

1. Append it to the table above with its name, licence, and upstream URL.
2. If the dependency's licence requires it, mirror the licence text into
   `docs/licenses/<package>.txt` and link to that file here.
3. For copyleft-adjacent licences (LGPL, MPL, GPL) double-check what
   "distribution" of the dependency means in your deployment — static
   linking, bundled container images, etc.

For questions, email <legal@genaiprotos.com> or open an issue on
GitHub.
