# STT Inference — Implementation Plan (Production-minded, Early-stage)

## Goals

- Serve **Qwen3-ASR-1.7B** and **Whisper-Large-v3** via vLLM on DGX Spark.
- FastAPI backend that fronts both models behind a single WebSocket + REST API.
- Minimal HTML/CSS/JS frontend to test real-time mic transcription and file upload.
- Good engineering hygiene: typed, async, configurable, containerised, observable — without over-engineering.

## Non-goals (for now)

- Auth, multi-tenancy, rate limiting (wire hooks but don't implement).
- Kubernetes / horizontal scaling.
- Fine-tuning.
- Speaker diarization / timestamps (keep the schema extensible, defer implementation).

---

## Architecture

```
  ┌───────────────────────────────┐
  │  Browser (HTML/CSS/JS)        │
  │  - getUserMedia → AudioWorklet│
  │  - Sends PCM16 @ 16kHz via WS │
  │  - Model/language selector    │
  │  - Interim + final transcript │
  └──────────────┬────────────────┘
                 │ ws://backend:3000/ws/transcribe
  ┌──────────────▼────────────────┐
  │  FastAPI Backend (:3000)      │
  │  - /ws/transcribe  (live)     │
  │  - /transcribe     (file)     │
  │  - /healthz /readyz /metrics  │
  │  - Silero VAD chunker         │
  │  - Async httpx → vLLMs        │
  │  - Structured JSON logs       │
  └─────────┬──────────────┬──────┘
            │              │
    ┌───────▼─────┐ ┌──────▼──────────┐
    │ vLLM Qwen   │ │ vLLM Whisper    │
    │ :8000       │ │ :8001           │
    │ Qwen3-ASR   │ │ whisper-large-v3│
    │ -1.7B       │ │                 │
    └─────────────┘ └─────────────────┘
         (NVIDIA container nvcr.io/nvidia/vllm:25.11-py3, both on DGX Spark GPU)
```

**Why not direct browser → vLLM**: keeps vLLM internal (never exposed), centralises VAD + session state + routing + logging, gives us a single place to add auth/rate-limiting later.

---

## Project layout

```
STT_Inference/
├── README.md                          # Setup, run, test
├── PLAN.md                            # This file
├── Makefile                           # make start / stop / status / test / logs / bench
├── compose.yml                        # prod-ish compose
├── compose.dev.yml                    # dev overrides (hot reload, debug)
├── .env.example                       # template for env vars
├── .gitignore
├── pyproject.toml                     # backend deps + tooling (ruff/mypy/pytest)
│
├── backend/
│   ├── Dockerfile
│   └── src/stt_backend/
│       ├── __init__.py
│       ├── main.py                    # FastAPI app factory + lifespan
│       ├── config.py                  # Pydantic Settings (env-driven)
│       ├── logging.py                 # JSON structlog setup
│       ├── routes/
│       │   ├── health.py              # /healthz /readyz
│       │   ├── transcribe.py          # POST /transcribe (file upload)
│       │   └── ws.py                  # WebSocket /ws/transcribe
│       ├── services/
│       │   ├── vllm_client.py         # Async OpenAI-compatible clients
│       │   ├── vad.py                 # Silero VAD chunker
│       │   └── audio.py               # webm/opus → PCM16 16k mono
│       └── schemas/
│           └── ws_messages.py         # Pydantic schemas for WS protocol
│   └── tests/
│       ├── conftest.py
│       ├── test_config.py
│       ├── test_vad.py
│       └── test_vllm_client_mock.py
│
├── serving/
│   ├── qwen/
│   │   ├── Dockerfile                 # FROM nvcr.io/nvidia/vllm:25.11-py3
│   │   └── entrypoint.sh
│   └── whisper/
│       ├── Dockerfile                 # FROM nvcr.io/nvidia/vllm:25.11-py3
│       └── entrypoint.sh
│
├── frontend/
│   ├── index.html                     # served statically by backend
│   ├── style.css
│   ├── app.js                         # WS client, audio capture
│   └── worklet.js                     # AudioWorkletProcessor (raw PCM)
│
├── scripts/
│   ├── download_models.sh             # HF snapshot downloads to cache
│   ├── smoke_test.sh                  # curl probe both vLLMs + backend
│   └── bench_wer.py                   # WER on CV-ar test split
│
└── data/
    └── samples/                       # small hand-curated audio for tests
```

---

## Tech choices (and why)

| Concern | Choice | Reasoning |
|---|---|---|
| Python | 3.11+ | stable async, good typing |
| Web | **FastAPI + uvicorn** | native async + auto OpenAPI docs + WS support |
| HTTP client | **httpx** (async) | one client, reused, pool-aware |
| Validation | **Pydantic v2** | settings + request/response schemas |
| VAD | **silero-vad** | small (<2MB), accurate, ONNX, CPU-fast |
| Audio decode | **pyav** (FFmpeg bindings) | robust browser webm/opus → PCM16 |
| Logging | **structlog** (JSON) | grep-able, ships to any aggregator later |
| Testing | **pytest + pytest-asyncio** | standard |
| Lint/format | **ruff** (replaces black+isort+flake8) | one tool |
| Types | **mypy** (strict) | catches bugs pre-runtime |
| Deps | **uv** or **pip + pyproject.toml** | pinned in `pyproject.toml`, lockfile via `uv lock` |
| Container | NVIDIA `vllm:25.11-py3` for serving; slim Python for backend | known-good sm_121 on DGX Spark |
| Orchestration | **docker compose** | enough for single-host DGX Spark |
| Frontend | vanilla HTML/CSS/JS | zero build step, easy to hack |

---

## vLLM server commands

### Qwen3-ASR (port 8000)
```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
  --trust-remote-code \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --host 0.0.0.0 --port 8000
```

### Whisper Large-v3 (port 8001)
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
vllm serve openai/whisper-large-v3 \
  --task transcription \
  --gpu-memory-utilization 0.35 \
  --max-num-seqs 16 \
  --host 0.0.0.0 --port 8001
```

Both expose `/v1/models`, `/v1/audio/transcriptions` (OpenAI-compatible).

`gpu_memory_utilization=0.35` on each → ~70% total → room for KV cache growth + headroom on DGX Spark's unified memory.

---

## WebSocket protocol (versioned, typed)

Client → Server (JSON frames interleaved with binary audio frames):

```json
// First message
{"type": "session.start", "model": "qwen3-asr" | "whisper", "language": "ar" | "en" | null, "sample_rate": 16000}

// Audio (binary frames) — raw PCM16 little-endian, 16kHz, mono

// Finalize
{"type": "session.commit"}
```

Server → Client:

```json
{"type": "session.accepted"}
{"type": "transcription.partial", "text": "…", "segment_id": 3}
{"type": "transcription.final",   "text": "…", "segment_id": 3, "language": "ar", "duration_ms": 2310, "rtf": 0.18}
{"type": "error", "code": "vad_timeout" | "vllm_unavailable" | ..., "message": "…"}
```

Interim updates happen as VAD emits segments. Final is authoritative.

Schema lives in `schemas/ws_messages.py` as Pydantic models so both sides (server validate, we generate TypeScript types later if needed) stay in sync.

---

## Good-practice checklist (early but sturdy)

- [x] **Config**: all tunables via env (model names, ports, VAD thresholds, GPU memory, HF cache path). Pydantic Settings. `.env.example` committed, `.env` ignored.
- [x] **Logging**: structlog JSON to stdout; correlation ID per WS session; no PII (no full transcripts at INFO, only lengths; DEBUG for content).
- [x] **Health**: `/healthz` (always 200 if process up), `/readyz` (200 only if both vLLMs reachable). vLLM containers have `HEALTHCHECK` hitting `/v1/models`.
- [x] **Error handling**: httpx timeouts + retries (tenacity) on vLLM calls; WS: graceful close with error frame, never crash the session.
- [x] **Async**: FastAPI + httpx + asyncio throughout; no blocking calls in request path (VAD runs in `loop.run_in_executor` since it's torch CPU).
- [x] **Backpressure**: bounded `asyncio.Queue` between WS receive and VAD; drop with warning if queue full (log rate-limited).
- [x] **Tests**: pytest with asyncio; mock vLLM with respx; one real sample through VAD; CI-runnable without GPU.
- [x] **Types**: strict mypy on `backend/src/`; Pydantic v2 for boundaries.
- [x] **Lint**: ruff in pyproject.toml, pre-commit optional (not required to merge).
- [x] **Docker**: multi-stage backend Dockerfile; non-root user; pinned base image digests ideally.
- [x] **Reproducibility**: pin model revision (commit SHA) when downloading from HF; log model revision at startup.
- [x] **Makefile** for common dev tasks: `make start / stop / status / test / lint / logs / shell / bench`.
- [x] **Security posture** (pre-auth): CORS limited to configured origins, WS origin check, max upload size, max WS session duration, max audio per session. No secrets in logs.
- [x] **Observability hooks**: Prometheus `/metrics` endpoint via `prometheus-fastapi-instrumentator` (labels: model, language, endpoint, status).
- [x] **Graceful shutdown**: FastAPI lifespan drains inflight WS sessions on SIGTERM.

Deliberately deferred (stubs only):

- Auth (leave a single `Depends(get_current_user)` stub that's a no-op; swap later).
- Rate limiting (leave a middleware hook).
- Distributed tracing (structlog context keys are already correlation-friendly).

---

## Implementation order (8 phases, committable each)

1. **Scaffolding** — `pyproject.toml`, Makefile, `.env.example`, `.gitignore`, README skeleton.
2. **Serving containers** — `serving/qwen/Dockerfile` + `serving/whisper/Dockerfile` + entrypoints. Bring up via `compose.yml`. Smoke test with `curl` against `/v1/models` and a known-good WAV.
3. **Backend skeleton** — FastAPI app factory, config, logging, `/healthz /readyz`, Dockerfile, wired into compose.
4. **vLLM client** — async httpx wrapper for `/v1/audio/transcriptions` with tenacity retries and typed responses. Unit tests with respx.
5. **REST `/transcribe`** (file upload) — working end-to-end through both models; integration-smoke test via `make smoke`.
6. **VAD + audio decode** — Silero VAD wrapper; pyav webm/opus → PCM16. Unit tests on fixture audio.
7. **WebSocket `/ws/transcribe`** — session state, backpressure, routing by `model` field, interim + final frames, error paths.
8. **Frontend** — HTML/CSS/JS + AudioWorklet; mic capture; model + language dropdowns; transcript pane (interim faded, final bold); file upload card. Served at `/`.

Each phase gets its own commit with a runnable `make` target and a smoke test.

---

## Verification

**Per-service**:
- `curl http://localhost:8000/v1/models` → Qwen listed.
- `curl http://localhost:8001/v1/models` → Whisper listed.
- `curl http://localhost:3000/readyz` → 200 + JSON `{qwen: "ok", whisper: "ok"}`.

**End-to-end**:
- `make smoke` — posts a known Arabic WAV to both models via backend REST and asserts non-empty text, correct RTL characters, language detected `ar`.
- Browser: open `http://localhost:3000`, pick Qwen + Arabic, speak MSA sentence → transcript appears within ~1.5s of pause → swap to Whisper → same works.
- `make bench` — WER on a 50-utterance Arabic sample across both models; writes `results/bench_YYYY-MM-DD.csv`.

**Pass bar**:
- Qwen3-ASR Arabic MSA WER < 25%, Whisper < 30%, on our sample set (treat worse than these as a regression to investigate).
- p95 end-to-end latency for a 5s utterance under 1.5s.
- No backend OOM or vLLM crash over a 30-minute live session.

---

## Open decisions to confirm before coding

1. **Whisper variant**: Large-v3 (recommended, best Arabic accuracy) vs. Large-v3 Turbo (faster, lower Arabic accuracy). Default: **Large-v3**.
2. **Audio transport**: AudioWorklet (raw PCM16, what vLLM/pyav wants — cleaner) vs. MediaRecorder (webm/opus, needs transcoding — more compatible with older browsers). Default: **AudioWorklet** with MediaRecorder fallback.
3. **Package manager**: `uv` (fast, modern, has lockfile) vs. `pip` (ubiquitous). Default: **uv**, fall back to pip if it causes trouble on DGX Spark aarch64.
4. **Frontend framework**: vanilla (zero build) vs. bundled (Vite + TS). Default: **vanilla** for now — revisit only if the JS grows past ~300 LoC.

Once these four are confirmed (or default-accepted), phase 1 begins.
