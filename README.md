# STT Inference — Arabic + Multilingual Speech-to-Text on NVIDIA DGX Spark

A self-hosted, production-minded Speech-to-Text service that runs **two open-source
ASR models side-by-side** on a single NVIDIA DGX Spark, behind one friendly HTTP
API and a browser UI.

- 🎙️ **Live mic transcription** over WebSocket (raw PCM16 → VAD → transcription)
- 📁 **File upload** transcription over REST (OpenAI-compatible `/v1/audio/transcriptions` under the hood)
- 🌍 **Multilingual**, with extra attention to **Modern Standard Arabic**
- 🔒 Runs **entirely local** — no data leaves the box

---

## What this repo gives you

| Piece | What it does |
|---|---|
| **Qwen3-ASR-1.7B** (Alibaba, Apache 2.0) | Primary model. 30 languages incl. Arabic. SOTA open-source ASR. Streaming-capable. |
| **Whisper-Large-v3** (OpenAI, MIT) | Secondary / comparison model. 99 languages. Mature ecosystem. |
| **vLLM × 2** | Each model gets its own vLLM server exposing an OpenAI-compatible API. |
| **FastAPI backend** | Routes browser traffic to the right model, adds server-side VAD, gives you REST + WebSocket + metrics. |
| **Vanilla web UI** | Mic button, model/language pickers, live transcript, file upload. |
| **Docker compose** | One command brings everything up. |

---

## High-level architecture

```
                  ┌────────────────────────────────────────┐
                  │          Browser (HTML/JS)             │
                  │  - getUserMedia → AudioWorklet → PCM16 │
                  │  - WebSocket client + file uploader    │
                  └─────────────────┬──────────────────────┘
                                    │
                     (:3000)  HTTP + WebSocket
                                    │
                  ┌─────────────────▼──────────────────────┐
                  │        FastAPI Backend                 │
                  │  ─ POST /transcribe  (file upload)     │
                  │  ─ WS   /ws/transcribe (live mic)      │
                  │  ─ GET  /healthz  /readyz  /metrics    │
                  │  ─ Silero VAD · structured JSON logs   │
                  └───────┬───────────────────────┬────────┘
                          │                       │
                 http://qwen:8000         http://whisper:8001
                          │                       │
            ┌─────────────▼──────┐    ┌───────────▼─────────┐
            │   vLLM server #1   │    │   vLLM server #2    │
            │   Qwen3-ASR-1.7B   │    │ Whisper-Large-v3    │
            │   /v1/audio/       │    │ /v1/audio/          │
            │    transcriptions  │    │   transcriptions    │
            └────────────────────┘    └─────────────────────┘
                          │                       │
                          └──────┬────────────────┘
                                 ▼
                    NVIDIA DGX Spark (GB10 Blackwell)
                    128 GB unified memory · sm_121 · aarch64
```

**Why two models**: different strengths — you can compare live, benchmark WER,
and fall back if one misbehaves on a given language or accent.

**Why VAD in the middle**: neither vLLM endpoint is truly "continuous
streaming" for Arabic-grade quality. We chunk on natural silence boundaries
(Silero VAD) and send each utterance as a fresh transcription call. Feels like
real-time (<1.5 s end-to-end), avoids the known
[Qwen3-ASR realtime-endpoint quality bug](https://github.com/vllm-project/vllm/issues/35767).

---

## How streaming actually works (short version)

1. Browser captures the mic at 16 kHz mono via **AudioWorklet**.
2. Worklet converts Float32 → Int16 and posts raw PCM to the main thread.
3. Main thread streams PCM bytes over WebSocket to the backend.
4. Backend feeds bytes into **Silero VAD**.
5. When VAD sees `500ms` of silence *after* at least `250ms` of speech, or when
   the segment passes `15s`, it emits that utterance.
6. Backend POSTs the utterance (as a WAV) to the selected vLLM model.
7. The transcription is pushed back to the browser as a `transcription.segment`
   JSON frame.

That's it. No proprietary streaming protocol, no WebRTC, no custom ML streaming
code — one HTTP round-trip per sentence.

---

## Requirements

- **Hardware**: NVIDIA DGX Spark (GB10, Blackwell, aarch64) — or any Linux host
  with an NVIDIA GPU, CUDA ≥ 13, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **Software**: `docker` (≥ 25), `docker compose` v2, `make` (optional), `uv`
  (optional, for running tests locally).
- **Disk**: ~20 GB for container images, ~7 GB for model weights.
- **Network**: internet access on first run (pulls Docker image + model weights).

---

## Quick start

```bash
# 1. First-time setup: creates .env, installs uv deps, pulls models
bash scripts/setup.sh

# 2. Bring up the stack (Qwen + Whisper + backend)
bash scripts/start.sh

# 3. Wait 1–3 minutes, then verify
curl http://localhost:3000/readyz

# 4. Open the web UI
xdg-open http://localhost:3000    # Linux
open     http://localhost:3000    # macOS
```

After that, click **Start recording**, speak Arabic MSA (or any supported
language), pause, and watch the transcript appear.

For full operational detail — every endpoint, every flag, curl examples,
troubleshooting — see **[docs/GUIDE.md](docs/GUIDE.md)**.

---

## Use the vLLM servers on their own

Each vLLM container is a full **OpenAI-compatible HTTP server** — you don't
need this repo's backend to use them. Point any OpenAI audio client at Qwen
or Whisper directly:

```bash
# Run just Qwen3-ASR (stops the other services if they're up)
docker compose up -d qwen

# Transcribe with curl — OpenAI-compatible endpoint
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@sample_ar.wav \
  -F language=ar | jq
```

Python OpenAI SDK:

```python
from openai import OpenAI
qwen = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
with open("sample_ar.wav", "rb") as f:
    print(qwen.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B", file=f, language="ar",
    ).text)
```

Same idea for Whisper at `http://localhost:8001/v1` with model
`openai/whisper-large-v3`. Full section (when to go direct vs. through the
backend, Node.js SDK example, parameter support matrix, auth/CORS, running
only one container): **[docs/GUIDE.md → Using vLLM servers standalone](docs/GUIDE.md#using-vllm-servers-standalone-openai-compatible)**.

---

## Everyday commands

All common tasks have a corresponding shell script. The `Makefile` is a thin
wrapper if you prefer `make`.

```bash
bash scripts/setup.sh              # first-time setup
bash scripts/start.sh              # start all services
bash scripts/start.sh --dev        # start with hot-reload backend
bash scripts/start.sh --follow     # start + tail logs
bash scripts/status.sh             # stack status (containers + endpoints + GPU)
bash scripts/stop.sh               # stop services
bash scripts/stop.sh --all         # stop + wipe volumes
bash scripts/logs.sh               # tail all logs
bash scripts/logs.sh backend       # tail only the backend
bash scripts/smoke_test.sh         # end-to-end smoke check
bash scripts/test.sh               # ruff + mypy + pytest
bash scripts/test.sh unit          # pytest only
bash scripts/bench.sh              # WER benchmark
bash scripts/download_models.sh    # pull model weights
```

---

## Project layout

```
STT_Inference/
├── README.md                 ← you are here (high-level overview)
├── PLAN.md                   ← architectural plan + decisions
├── docs/
│   └── GUIDE.md              ← operator guide (endpoints, params, examples)
├── compose.yml               ← docker compose: qwen + whisper + backend
├── compose.dev.yml           ← dev overrides (hot reload, DEBUG logs)
├── .env.example              ← config template — copy to .env
├── Makefile                  ← convenience wrapper over scripts/
├── pyproject.toml            ← backend deps (FastAPI, Silero, etc.)
├── backend/
│   ├── Dockerfile            ← multi-stage; prod + dev targets
│   └── src/stt_backend/
│       ├── main.py           ← FastAPI app factory + lifespan
│       ├── config.py         ← env-driven settings (Pydantic)
│       ├── logging.py        ← structlog JSON logging
│       ├── routes/
│       │   ├── health.py     ← /healthz /readyz
│       │   ├── transcribe.py ← POST /transcribe (file upload)
│       │   └── ws.py         ← WS /ws/transcribe (live mic)
│       ├── services/
│       │   ├── vllm_client.py ← async httpx client to both vLLM servers
│       │   ├── vad.py         ← Silero VAD streaming chunker
│       │   └── audio.py       ← WAV wrapper, ms↔samples helpers
│       └── schemas/
│           ├── transcription.py ← request/response models
│           └── ws_messages.py   ← WebSocket wire-protocol types
├── serving/
│   ├── qwen/                 ← vLLM Dockerfile + entrypoint for Qwen3-ASR
│   └── whisper/              ← vLLM Dockerfile + entrypoint for Whisper
├── frontend/
│   ├── index.html            ← mic + upload UI
│   ├── style.css             ← dark theme, RTL-aware
│   ├── app.js                ← WebSocket + mic capture
│   └── worklet.js            ← AudioWorkletProcessor (Float32 → Int16)
├── scripts/
│   ├── setup.sh              ← first-time setup
│   ├── start.sh / stop.sh    ← compose lifecycle
│   ├── status.sh             ← report stack health + endpoints + GPU
│   ├── logs.sh               ← tail logs
│   ├── test.sh               ← ruff + mypy + pytest
│   ├── smoke_test.sh         ← end-to-end sanity
│   ├── bench.sh              ← WER benchmark
│   ├── bench_wer.py          ← WER calc over samples/
│   └── download_models.sh    ← pull HF weights into ./hf_cache
└── data/samples/             ← drop <id>.wav + <id>.txt here for benching
```

---

## Notes on DGX Spark (Blackwell / sm_121 / aarch64)

A few things that matter on this hardware and are already baked into the
Dockerfiles — listed here so you know what to look out for when upgrading:

- Uses **`nvcr.io/nvidia/vllm:26.03-py3`** — the first NGC tag with native
  `Qwen3ASRForConditionalGeneration`. Older tags (`25.11`, `26.01`) don't have it.
- **Never** `pip install vllm` or `qwen-asr[vllm]` on top of the container —
  PyPI vLLM is CUDA-12, breaks with `libcudart.so.12 not found` on sm_121.
- **Never** `pip install torchaudio` — PyPI wheels are ABI-incompatible with
  NVIDIA's custom FP4-enabled torch. We patch vLLM's processor imports to
  avoid needing it.
- Whisper on Blackwell needs `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, not Triton
  (Triton lacks encoder-decoder cross-attention on sm_121).
- `CUDA_VISIBLE_DEVICES=""` (empty string) hides all GPUs — leave it unset or
  commented out in `.env`.

Full rationale, error signatures, and fix commands: see
**[docs/GUIDE.md → DGX Spark deployment gotchas](docs/GUIDE.md#dgx-spark-deployment-gotchas-blackwell--sm_121--aarch64)**.

Qwen3-ASR-1.7B + Whisper-Large-v3 together use ~8 GB VRAM; the 128 GB unified
memory is plenty. First `docker compose up` takes 1–3 min per container while
vLLM loads weights and compiles CUDA graphs; subsequent starts use the cached
graphs (~30 s).

---

## Security note

Out of the box this is a **developer-grade local deployment** — no auth, no
rate limiting, CORS limited to `localhost:3000`. Don't expose port 3000 to the
public internet without adding an auth layer. See **[docs/GUIDE.md → Security](docs/GUIDE.md#security-and-hardening)**
for the recommended hardening checklist before production use.

---

## Licensing & attribution

**STT Studio itself** — this repo's backend, frontend, Dockerfiles, scripts,
and docs — is released under the **Apache License 2.0**. Full text in
[`LICENSE`](LICENSE), attribution text in [`NOTICE`](NOTICE).

Copyright © 2026 GenAI Protos.

### Bundled third-party components

| Component | Role | License |
|---|---|---|
| **Qwen3-ASR-1.7B** — [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Primary multilingual ASR | Apache 2.0 |
| **Whisper Large-v3** — [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) | Secondary ASR / benchmark | MIT |
| **vLLM** (runs both models) | Inference server | Apache 2.0 |
| **NVIDIA vLLM container** `nvcr.io/nvidia/vllm:26.03-py3` | Runtime image | NVIDIA Deep Learning Container License |
| **Silero VAD v5** | Server-side voice activity detection | MIT |
| FastAPI, Starlette, Pydantic, httpx, structlog, tenacity, etc. | Backend framework + libs | Mostly MIT / BSD / Apache 2.0 |
| Inter, JetBrains Mono (Google Fonts) | UI typography | SIL OFL 1.1 |
| `data/samples/jfk.wav` | Smoke-test audio | U.S. public domain (JFK 1961 inaugural) |

A **full list** of every third-party project with links and licence notices
is maintained in **[docs/ATTRIBUTIONS.md](docs/ATTRIBUTIONS.md)**. If you fork
or deploy this repo, keep that file alongside your distribution — it's how
you comply with the notice terms of the licences above.

### Brand assets (not Apache 2.0)

The files under `frontend/assets/genaiprotos-*.*` and `GenAI Protos Logo/`
are **trademarks and branding of GenAI Protos**. They are included so this
deployment can display them. **They are not covered by the Apache 2.0
licence** — if you fork STT Studio for your own brand, replace them with
your own.

### Trademarks

"Qwen" is a trademark of Alibaba Cloud. "Whisper" and "OpenAI" are
trademarks of OpenAI. "NVIDIA", "DGX Spark", "Grace Blackwell", "CUDA",
"cuDNN", and "Triton" are trademarks of NVIDIA Corporation. Use of these
names here is purely descriptive; it does not imply endorsement.

---

## Further reading

- **[docs/GUIDE.md](docs/GUIDE.md)** — detailed operator & API guide
- **[PLAN.md](PLAN.md)** — architectural plan, decisions, open questions
- [Qwen3-ASR on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Whisper-Large-v3 on HuggingFace](https://huggingface.co/openai/whisper-large-v3)
- [vLLM OpenAI-compatible server docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [NVIDIA DGX Spark docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
