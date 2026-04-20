# Operator & API Guide

> Everything you need to run, operate, and integrate with the STT Inference
> service. If you want the 60-second overview, see the top-level
> [README.md](../README.md).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [First-time setup](#first-time-setup)
3. [Day-to-day operations](#day-to-day-operations)
4. [Shell scripts reference](#shell-scripts-reference)
5. [Configuration (.env) reference](#configuration-env-reference)
6. [HTTP API reference](#http-api-reference)
    - [GET /healthz](#get-healthz)
    - [GET /readyz](#get-readyz)
    - [GET /metrics](#get-metrics)
    - [POST /transcribe](#post-transcribe)
    - [WebSocket /ws/transcribe](#websocket-wstranscribe)
7. [WebSocket protocol details](#websocket-protocol-details)
8. [Use case recipes](#use-case-recipes)
    - [Transcribe an Arabic WAV file from a script](#recipe-transcribe-an-arabic-wav-file)
    - [Live caption a meeting in the browser](#recipe-live-caption-a-meeting)
    - [Batch transcribe a folder of recordings](#recipe-batch-transcribe-a-folder)
    - [Compare Qwen vs Whisper WER on your data](#recipe-compare-qwen-vs-whisper)
    - [Call from Python using the OpenAI SDK](#recipe-call-via-the-openai-sdk)
9. [Using vLLM servers standalone (OpenAI-compatible)](#using-vllm-servers-standalone-openai-compatible)
10. [Observability](#observability)
11. [Benchmarking](#benchmarking)
12. [DGX Spark deployment gotchas](#dgx-spark-deployment-gotchas-blackwell--sm_121--aarch64)
13. [Troubleshooting](#troubleshooting)
14. [Security and hardening](#security-and-hardening)
15. [FAQ](#faq)

---

## Prerequisites

| Requirement | Why | How to check |
|---|---|---|
| Docker Engine ≥ 25 | runs the three containers | `docker --version` |
| Docker Compose v2 | orchestrates them | `docker compose version` |
| NVIDIA Container Toolkit | GPU passthrough | `docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi` |
| ~20 GB free disk | images + weights | `df -h` |
| Outbound internet (first run) | pulls image + model weights | — |

Optional (for local testing / running scripts without compose):

| Tool | Purpose |
|---|---|
| [`uv`](https://docs.astral.sh/uv/) | fast Python env + deps manager |
| `make` | thin wrapper over shell scripts |
| `curl`, `jq` | API poking |

---

## First-time setup

```bash
cd STT_Inference
bash scripts/setup.sh
```

What that does:

1. Copies `.env.example` → `.env` (only if `.env` doesn't already exist).
2. Verifies Docker is present and the daemon is running.
3. Smoke-tests GPU passthrough (`nvidia-smi` inside a container).
4. Installs `uv` if missing, then `uv sync --extra dev` (for tests/lint locally).
5. Runs `scripts/download_models.sh` to pull **Qwen3-ASR-1.7B** + **Whisper-Large-v3** into `./hf_cache/`.

If the GPU check fails, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To re-pull a specific model (e.g. after a version bump), edit `.env`:

```env
QWEN_REVISION=abc123        # commit SHA or tag
WHISPER_REVISION=main
```

Then re-run `bash scripts/download_models.sh`.

---

## Day-to-day operations

**Bring up the stack**

```bash
bash scripts/start.sh                # production-ish mode
bash scripts/start.sh --dev          # hot-reload backend + DEBUG logs
bash scripts/start.sh --follow       # start and tail logs
```

First start takes 1–3 minutes while vLLM warms up and compiles CUDA graphs.
You'll see `Application startup complete` in the vLLM logs when each is ready.

**Check status**

```bash
bash scripts/status.sh              # colored, human-friendly summary
bash scripts/status.sh --quiet      # exit 0/1 only (for CI / cron)
```

`status.sh` prints: container health, `/v1/models` reachability for Qwen and
Whisper, backend `/healthz` + `/readyz`, and GPU memory usage. Exit code 0 =
fully ready, 1 = something's off (see output for which thing).

Raw endpoint probes if you want to script directly:

```bash
curl -s localhost:3000/healthz | jq
curl -s localhost:3000/readyz  | jq
```

`/readyz` is the authoritative "stack is up and reachable" check — it probes
both vLLM servers from the backend.

**Tail logs**

```bash
bash scripts/logs.sh                # all services
bash scripts/logs.sh backend        # only the backend
bash scripts/logs.sh qwen whisper   # both vLLMs
```

**Stop**

```bash
bash scripts/stop.sh                # stops containers, keeps volumes
bash scripts/stop.sh --all          # + removes volumes (kills HF cache path in compose)
```

---

## Shell scripts reference

All live in `scripts/` and take `-h`/`--help` flags where sensible.

| Script | Purpose |
|---|---|
| `setup.sh` | First-time setup (env + deps + models). `--no-models` to skip download. |
| `start.sh` | `docker compose up -d --build` for all services. `--dev` for hot reload, `--follow` to tail logs after start. |
| `stop.sh` | `docker compose down`. `--all` to also wipe volumes + network. |
| `status.sh` | Reports containers + `/v1/models` + backend `/readyz` + GPU memory. Exits 0 if fully ready, 1 otherwise. `--quiet` for CI. |
| `logs.sh` | Tail compose logs. Pass service name(s) to scope. |
| `test.sh` | Runs ruff + mypy + pytest. Sub-modes: `unit`, `lint`, `type`, `fmt`, `cov`. |
| `smoke_test.sh` | End-to-end sanity: hits `/v1/models` on both vLLMs and `/readyz` on backend. Optionally transcribes a sample. |
| `bench.sh` | Wraps `bench_wer.py` over `data/samples/<id>.wav` + `<id>.txt` pairs. Env vars: `BACKEND_URL`, `SAMPLES_DIR`, `OUT_DIR`, `LANGUAGE`. |
| `bench_wer.py` | The actual WER benchmarker. Outputs CSV to `results/`. |
| `download_models.sh` | Pulls Qwen3-ASR + Whisper weights into `./hf_cache/` via the NVIDIA vLLM container. |

Everything is idempotent and safe to re-run.

---

## Configuration (.env) reference

All runtime tunables are environment variables. Full list in `.env.example`.
Summary of the ones you most likely want to change:

| Variable | Default | What it does |
|---|---|---|
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace ID to serve on port 8000. |
| `WHISPER_MODEL` | `openai/whisper-large-v3` | HuggingFace ID to serve on port 8001. |
| `QWEN_REVISION` / `WHISPER_REVISION` | `main` | Pin to a specific commit SHA for reproducibility. |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.35` | Fraction of GPU memory the Qwen server grabs. |
| `WHISPER_GPU_MEMORY_UTILIZATION` | `0.35` | Same for Whisper. They coexist; keep sum ≤ 0.8 on DGX Spark. |
| `BACKEND_PORT` | `3000` | Where the FastAPI server listens. |
| `QWEN_PORT` / `WHISPER_PORT` | `8000` / `8001` | Host port bindings. |
| `VAD_THRESHOLD` | `0.5` | Silero speech probability cutoff (0..1). Raise for noisy rooms. |
| `VAD_MIN_SILENCE_MS` | `500` | Silence duration that triggers a segment emission. |
| `VAD_MIN_SPEECH_MS` | `250` | Below this, a candidate segment is treated as noise and dropped. |
| `VAD_MAX_SEGMENT_MS` | `15000` | Hard cut. Long utterances get chopped at this length. |
| `WS_MAX_SESSION_SECONDS` | `1800` | WebSocket session hard cap (30 min). |
| `WS_MAX_AUDIO_QUEUE` | `200` | Bounded backpressure queue; oldest chunks dropped past this. |
| `VLLM_TIMEOUT_SECONDS` | `60` | Per-request HTTP timeout from backend → vLLM. |
| `VLLM_RETRIES` | `2` | Retries on transport-level failures (connection resets etc). |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS whitelist. Change to expose the UI on another host. |
| `LOG_LEVEL` | `INFO` | Standard Python levels. `DEBUG` is verbose. |
| `LOG_FORMAT` | `json` | `json` (for log aggregators) or `console` (human-friendly). |
| `LOG_TRANSCRIPT_CONTENT` | `false` | If `true`, DEBUG logs include the transcript text. **Privacy risk.** |
| `HF_CACHE_DIR` | `./hf_cache` | Shared HuggingFace cache mounted into both vLLM containers. |
| `HF_TOKEN` | *(empty)* | Only needed for gated models. Our two defaults are public. |
| `CUDA_VISIBLE_DEVICES` | *(empty)* | Pin GPUs, e.g. `0`. Empty = all visible. |

After any change, restart: `bash scripts/stop.sh && bash scripts/start.sh`.

---

## HTTP API reference

Base URL: `http://<host>:<BACKEND_PORT>` (default `http://localhost:3000`).

Live, interactive OpenAPI docs are at `http://localhost:3000/docs`.

---

### `GET /healthz`

Liveness probe. Always returns `200` if the Python process is up. Does **not**
check upstream vLLM reachability — use `/readyz` for that.

**Response `200`**

```json
{"status": "ok", "version": "0.1.0"}
```

Example:

```bash
curl -s http://localhost:3000/healthz | jq
```

---

### `GET /readyz`

Readiness probe. Makes live `/v1/models` calls to both vLLM servers and returns
their status. `200` only if both respond.

**Response `200`**

```json
{
  "ready": true,
  "upstreams": {
    "qwen":    {"ok": true,  "detail": "ok", "url": "http://qwen:8000/"},
    "whisper": {"ok": true,  "detail": "ok", "url": "http://whisper:8001/"}
  }
}
```

**Response `503`** (one upstream down)

```json
{
  "ready": false,
  "upstreams": {
    "qwen":    {"ok": false, "detail": "unreachable: ConnectError", "url": "http://qwen:8000/"},
    "whisper": {"ok": true,  "detail": "ok", "url": "http://whisper:8001/"}
  }
}
```

Use this in Kubernetes readiness probes / load-balancer health checks.

---

### `GET /metrics`

Prometheus-format metrics for the backend. Auto-generated via
`prometheus-fastapi-instrumentator` — includes request counts, latencies, and
in-flight request gauges, labelled by endpoint and status.

```bash
curl -s http://localhost:3000/metrics | head -40
```

If you need GPU metrics, scrape the `dcgm-exporter` or `nvidia-smi` separately
— we intentionally don't fan those into this endpoint.

---

### `POST /transcribe`

Upload an audio file, get back a transcription. **Use this for batch / CLI
scripts.** Don't use it for live mic — that's what the WebSocket is for.

**Request (multipart/form-data)**

| Field | Type | Required | Default | Notes |
|---|---|:---:|---|---|
| `file` | file | ✅ | — | Audio file. Supported: wav, mp3, flac, ogg, webm, m4a. Max **50 MB**. |
| `model` | string | ✅ | — | `qwen3-asr` or `whisper` |
| `language` | string | ❌ | *(auto)* | ISO 639-1, e.g. `ar`, `en`, `es`. |
| `prompt` | string | ❌ | — | Optional text prompt to bias the decoder (both models support this). |
| `temperature` | float | ❌ | `0.0` | Sampling temp. `0.0` = deterministic greedy. |

**Response `200`**

```json
{
  "text": "مرحبا بك في نظام التعرف على الكلام",
  "language": "ar",
  "model": "qwen3-asr",
  "duration_ms": 812
}
```

`duration_ms` is wall-clock round-trip to the vLLM server for *that*
transcription — useful for RTF-ish feel.

**Error responses**

| Status | Code | Meaning |
|---|---|---|
| 400 | `invalid_model` | `model` field wasn't `qwen3-asr` or `whisper`. |
| 400 | `empty_audio` | File was zero bytes. |
| 413 | `audio_too_large` | File > 50 MB. |
| 502 | `vllm_500` / `vllm_unreachable` / `vllm_bad_json` | Upstream vLLM failed — inspect its logs. |

**cURL examples**

Transcribe Arabic MSA via Qwen:

```bash
curl -s http://localhost:3000/transcribe \
  -F "file=@recording.wav" \
  -F "model=qwen3-asr" \
  -F "language=ar" \
| jq
```

Transcribe English via Whisper with a bias prompt:

```bash
curl -s http://localhost:3000/transcribe \
  -F "file=@meeting.mp3" \
  -F "model=whisper" \
  -F "language=en" \
  -F "prompt=Discussion of Q3 revenue and ARR forecasts." \
| jq
```

Auto-detect language (omit `language`):

```bash
curl -s http://localhost:3000/transcribe \
  -F "file=@mystery.wav" \
  -F "model=qwen3-asr" \
| jq
```

---

### WebSocket `/ws/transcribe`

Live mic transcription. Browser streams raw PCM, backend returns transcription
segments as VAD cuts them out.

**Endpoint**: `ws://<host>:<port>/ws/transcribe`
**Binary payload**: raw PCM16 **little-endian**, **mono**, **16 kHz**.
**Text payload**: JSON control frames (see protocol below).

See **[WebSocket protocol details](#websocket-protocol-details)** for the full
wire format.

Test from the command line with `websocat`:

```bash
# 1. Record 5s of mic to raw PCM16 @ 16 kHz mono
arecord -f S16_LE -c 1 -r 16000 -d 5 audio.raw

# 2. Open WS, send session.start, stream the audio, commit, read responses
websocat -E ws://localhost:3000/ws/transcribe <<EOF
{"type":"session.start","model":"qwen3-asr","language":"ar","sample_rate":16000}
EOF
# ... or better, use the web UI — scripting WS with stdin is fiddly
```

In practice: use the web UI at `http://localhost:3000`, or a Python client
(see [Use case recipes](#use-case-recipes)).

---

## WebSocket protocol details

**Framing**
- JSON **text frames** for control messages.
- Raw **binary frames** for audio (PCM16 LE mono @ 16 kHz).

### Client → Server messages

**`session.start`** (first frame, **required**)

```json
{
  "type": "session.start",
  "model": "qwen3-asr",
  "language": "ar",
  "prompt": "optional biasing text",
  "sample_rate": 16000
}
```

- `model`: `qwen3-asr` or `whisper`
- `language`: ISO 639-1, or `null` for auto-detect
- `prompt`: optional string
- `sample_rate`: **must be 16000** (server rejects otherwise)

**`session.commit`** (finalise the session)

```json
{ "type": "session.commit" }
```

Equivalent to closing the socket: the server flushes any pending VAD buffer
and sends `session.ended`.

**Binary frames** — any size of raw PCM16 bytes. Chunk size doesn't matter to
the server (it re-buffers into 512-sample VAD windows); aim for ~50–200 ms at
a time from the browser.

### Server → Client messages

**`session.accepted`** (after a valid `session.start`)

```json
{ "type": "session.accepted", "model": "qwen3-asr", "session_id": "a1b2c3d4" }
```

**`transcription.segment`** (one per VAD-emitted utterance)

```json
{
  "type": "transcription.segment",
  "segment_id": 1,
  "text": "مرحبا بك",
  "language": "ar",
  "model": "qwen3-asr",
  "audio_duration_ms": 1240,
  "upstream_duration_ms": 312,
  "reason": "silence"
}
```

- `segment_id` increments per session, starting at 1
- `reason`: `silence` (natural pause), `max_length` (15 s hard cut), `flush`
  (final flush on close), `error:<code>` (vLLM failed for this segment; `text`
  will be empty)
- `audio_duration_ms` = length of the emitted utterance
- `upstream_duration_ms` = vLLM response time for that utterance
- `RTF ≈ upstream_duration_ms / audio_duration_ms` (lower is better; <1.0 means
  faster than real time)

**`session.ended`** (clean close)

```json
{ "type": "session.ended", "total_segments": 7 }
```

**`error`** (fatal — session ends)

```json
{ "type": "error", "code": "unsupported_sample_rate", "message": "only 16000 Hz supported, got 48000" }
```

Error codes: `no_start`, `bad_start`, `unsupported_sample_rate`,
`session_timeout`.

---

## Use case recipes

### Recipe: transcribe an Arabic WAV file

From a shell script or cron job:

```bash
#!/usr/bin/env bash
set -euo pipefail

RESULT=$(curl -fsS http://localhost:3000/transcribe \
    -F "file=@$1" \
    -F "model=qwen3-asr" \
    -F "language=ar")

echo "$RESULT" | jq -r .text
```

### Recipe: live caption a meeting

Use the built-in web UI at `http://localhost:3000`. It's the happy path:

1. Pick **Qwen3-ASR-1.7B** or **Whisper Large-v3** from the Model dropdown.
2. Pick **Arabic (ar)** (or another language / Auto-detect) from the Language dropdown.
3. Click **Start recording**, allow microphone access.
4. Speak normally — a line appears per natural sentence/pause.
5. Click **Stop recording** when done.

If you need to embed this in another web app, copy `frontend/app.js` + `frontend/worklet.js`
— the code is vanilla and has no build step.

### Recipe: batch transcribe a folder

```bash
# Transcribe every .wav in ~/recordings via Qwen, write .txt next to each.
for f in ~/recordings/*.wav; do
    txt="${f%.wav}.txt"
    curl -fsS http://localhost:3000/transcribe \
        -F "file=@${f}" \
        -F "model=qwen3-asr" \
        -F "language=ar" \
    | jq -r .text > "${txt}"
    echo "  -> ${txt}"
done
```

### Recipe: compare Qwen vs Whisper

```bash
# Place matched pairs in data/samples/:
#   data/samples/s01.wav
#   data/samples/s01.txt   (reference transcript)
#   data/samples/s02.wav
#   data/samples/s02.txt
#   ...

bash scripts/bench.sh
```

Outputs `results/bench_<UTC_timestamp>.csv` with columns:

```
sample, model, language, wer, duration_ms, reference, hypothesis
```

Also prints a summary like:

```
== Summary ==
  qwen3-asr avg WER=0.187 over 10 samples
  whisper   avg WER=0.234 over 10 samples
```

You can change the language and directory:

```bash
LANGUAGE=en SAMPLES_DIR=data/eng_samples bash scripts/bench.sh
```

### Recipe: call via the OpenAI SDK

See the dedicated section [Using vLLM servers standalone (OpenAI-compatible)](#using-vllm-servers-standalone-openai-compatible)
below — covers Qwen and Whisper separately, with Python, JavaScript, and curl
examples, plus notes on which OpenAI parameters each model respects.

---

## Using vLLM servers standalone (OpenAI-compatible)

**Both vLLM servers are, on their own, fully-featured OpenAI-compatible HTTP
servers.** You do not need this repo's FastAPI backend to use them. Any tool
that speaks the [OpenAI audio transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription)
can point at `http://localhost:8000/v1` (Qwen) or `http://localhost:8001/v1`
(Whisper) and start working — a drop-in replacement for OpenAI, running on
your DGX Spark.

This section covers:

- When to use the backend vs. direct vLLM
- Running only one container (Qwen *or* Whisper)
- Full worked examples in Python (OpenAI SDK), Node.js, and curl
- What OpenAI parameters each model supports
- CORS + auth considerations if you expose the server to other apps

### When to use the backend vs. direct vLLM

| Use direct vLLM when... | Use the FastAPI backend when... |
|---|---|
| You're building a single-purpose app that only needs Arabic transcription, or only needs Whisper's multilingual transcription. | You want one URL that can dispatch to either model by name. |
| Your client app already speaks the OpenAI audio API. | You want live-microphone streaming with server-side VAD (`/ws/transcribe`). |
| You don't need multi-model A/B testing. | You want the 50 MB upload cap, error envelopes, structured logging, and Prometheus metrics. |
| You want the smallest possible deployment (one container). | You want a web UI served alongside. |

There's no wrong answer — you can do both: some apps hit vLLM directly, others
go through the backend, sharing the same GPU.

### Running only one model

All three services are defined in `compose.yml`, but nothing forces you to
run all three. Docker Compose will start dependencies you ask for and skip
the rest.

**Just Qwen3-ASR (port 8000):**

```bash
docker compose up -d qwen
# verify
curl -s http://localhost:8000/v1/models | jq
```

**Just Whisper-Large-v3 (port 8001):**

```bash
docker compose up -d whisper
curl -s http://localhost:8001/v1/models | jq
```

**Both vLLMs, no backend:**

```bash
docker compose up -d qwen whisper
```

To also skip the front-facing UI, that's already done — the backend service
is the only one that exposes a UI, and you're not starting it.

**Stop a single service:**

```bash
docker compose stop qwen     # leaves whisper + backend running
docker compose rm -f qwen    # also removes the container
```

**Useful: pointing another project at a running Qwen container.** If your
other app is also running in Docker on the same host, add it to the same
Docker network so it can reach `http://qwen:8000` by hostname:

```bash
docker network connect stt-inference_stt my-other-app-container
# Inside that container: base_url=http://qwen:8000/v1
```

Or, from a host-side process, use `http://localhost:8000/v1`.

### Python — OpenAI SDK

```bash
pip install "openai>=1.0"
```

**Qwen3-ASR for Arabic:**

```python
from openai import OpenAI

qwen = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

with open("sample_ar.wav", "rb") as f:
    result = qwen.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",     # must match the id in /v1/models
        file=f,
        language="ar",                    # ISO 639-1
        response_format="json",           # "json" | "text" | "verbose_json" | "srt" | "vtt"
        temperature=0.0,                  # 0 = deterministic greedy
        prompt="A news broadcast in formal Arabic.",  # optional biasing
    )

print(result.text)
```

**Whisper-Large-v3 for English with auto language detection:**

```python
from openai import OpenAI

whisper = OpenAI(base_url="http://localhost:8001/v1", api_key="not-used")

with open("meeting.mp3", "rb") as f:
    result = whisper.audio.transcriptions.create(
        model="openai/whisper-large-v3",
        file=f,
        # language omitted → Whisper auto-detects
        response_format="verbose_json",   # includes per-segment timestamps
    )

print(result.text)
for seg in result.segments:
    print(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")
```

**Both models behind a simple router in your app:**

```python
from openai import OpenAI

clients = {
    "qwen3-asr": OpenAI(base_url="http://localhost:8000/v1", api_key="x"),
    "whisper":   OpenAI(base_url="http://localhost:8001/v1", api_key="x"),
}
model_ids = {
    "qwen3-asr": "Qwen/Qwen3-ASR-1.7B",
    "whisper":   "openai/whisper-large-v3",
}

def transcribe(path: str, backend: str, language: str | None = None) -> str:
    with open(path, "rb") as f:
        r = clients[backend].audio.transcriptions.create(
            model=model_ids[backend], file=f, language=language,
        )
    return r.text
```

**Async version (for high-concurrency servers):**

```python
import asyncio
from openai import AsyncOpenAI

qwen = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")

async def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        r = await qwen.audio.transcriptions.create(
            model="Qwen/Qwen3-ASR-1.7B", file=f, language="ar",
        )
    return r.text

print(asyncio.run(transcribe("a.wav")))
```

### JavaScript / Node.js — OpenAI SDK

```bash
npm install openai
```

```javascript
import { OpenAI } from "openai";
import fs from "fs";

const qwen = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "not-used",   // the SDK requires one; vLLM ignores it unless you
                        // set --api-key on the server
});

const result = await qwen.audio.transcriptions.create({
  model: "Qwen/Qwen3-ASR-1.7B",
  file: fs.createReadStream("sample_ar.wav"),
  language: "ar",
  response_format: "json",
  temperature: 0,
});

console.log(result.text);
```

### curl (any language)

```bash
# Qwen3-ASR — Arabic
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer not-used" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "file=@sample_ar.wav" \
  -F "language=ar" \
  -F "response_format=json"
```

```bash
# Whisper — English with timestamps
curl -s http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer not-used" \
  -F "model=openai/whisper-large-v3" \
  -F "file=@meeting.mp3" \
  -F "response_format=verbose_json"
```

### Which OpenAI parameters each model respects

| Parameter | Qwen3-ASR-1.7B | Whisper-Large-v3 | Notes |
|---|:-:|:-:|---|
| `file` | ✅ | ✅ | wav, mp3, flac, ogg, m4a, webm, mp4, mpeg, mpga |
| `model` | ✅ | ✅ | Must match `/v1/models` id exactly |
| `language` | ✅ | ✅ | ISO 639-1 (`ar`, `en`, …). Omit for auto-detect. |
| `prompt` | ✅ | ✅ | Free-text biasing; 224 tokens max on Whisper |
| `response_format` | ✅ `json`, `text`, `verbose_json` | ✅ `json`, `text`, `verbose_json`, `srt`, `vtt` | `srt`/`vtt` only on Whisper |
| `temperature` | ✅ | ✅ | `0.0` = deterministic greedy decoding |
| `timestamp_granularities[]` | partial (needs `Qwen3-ForcedAligner` for word-level) | ✅ (`segment`, `word`) | On Whisper, set `response_format=verbose_json` to see them |

### Ports, CORS, and auth if you expose these

By default the two vLLM containers bind to `0.0.0.0:8000` and `0.0.0.0:8001`
on the Docker host. That means they're reachable from the host and (if the
host firewall allows) from the LAN. Before putting them behind a domain:

- **Auth**: `vllm serve --api-key <key>` turns on Bearer-token auth. Add that
  flag to `serving/{qwen,whisper}/entrypoint.sh` and the OpenAI SDK's
  `api_key=...` becomes a real check.
- **CORS**: vLLM sends permissive CORS by default. Lock it down with
  `--allowed-origins 'https://your.app'` if you call directly from a browser.
- **TLS**: terminate at a reverse proxy (nginx / Caddy / Cloudflared).
- **Network isolation**: if you don't want the vLLMs on the host interface,
  unmap the `ports:` in `compose.yml` — they'll still be reachable to
  *other* containers on the `stt-inference_stt` network via
  `http://qwen:8000`, `http://whisper:8001`.

### Diagnosing model-direct calls

- **404 on `/v1/audio/transcriptions`** → you hit the wrong port (Qwen is
  8000, Whisper is 8001).
- **`model_not_found`** → the `model` form field doesn't match exactly what
  `/v1/models` returns. Use `Qwen/Qwen3-ASR-1.7B` and `openai/whisper-large-v3`
  verbatim.
- **Slow first request** → vLLM compiles a CUDA graph on first use for a
  given batch shape. Subsequent calls are much faster.
- **Empty `text`** → audio under ~400 ms can be silently dropped by Whisper's
  internal VAD. Send longer clips or use the backend which enforces a minimum
  through Silero VAD.

---

## Observability

### Logs

All logs go to stdout as JSON (configurable to human-friendly with `LOG_FORMAT=console`).
Each WebSocket session attaches a `session_id` correlation key so you can grep
per-session events.

```bash
bash scripts/logs.sh backend | jq -c 'select(.session_id == "a1b2c3d4")'
```

Important events to grep for:

| Event | Meaning |
|---|---|
| `backend.start` / `backend.stop` | process lifecycle |
| `ws.session_started` | WS session accepted |
| `ws.session_closed` | session ended (duration_s logged) |
| `ws.audio_queue_full_dropping_oldest` | under-capacity — investigate CPU/GPU contention |
| `vllm.ok` | one successful transcription (`duration_ms`, `text_chars`) |
| `vllm.error` | vLLM returned 4xx/5xx |
| `transcribe.fail` | REST route reported an error to the client |

### Metrics

Prometheus scrape:

```yaml
# prometheus.yml snippet
scrape_configs:
  - job_name: stt-backend
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:3000']
```

Useful queries:

```
rate(http_requests_total{handler="/transcribe"}[1m])
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### vLLM internals

vLLM exposes its own `/metrics` endpoint:

```bash
curl -s http://localhost:8000/metrics | grep vllm_
curl -s http://localhost:8001/metrics | grep vllm_
```

Look at `vllm:num_requests_running`, `vllm:num_requests_waiting`,
`vllm:time_to_first_token_seconds`.

---

## Benchmarking

Drop matched pairs in `data/samples/`:

```
data/samples/s01.wav   # 16 kHz mono preferred, but any format vLLM accepts
data/samples/s01.txt   # reference transcript (UTF-8)
data/samples/s02.wav
data/samples/s02.txt
...
```

Then:

```bash
bash scripts/bench.sh            # default: Arabic, both models
LANGUAGE=en bash scripts/bench.sh
```

The script:

1. POSTs each file to `/transcribe` for each model.
2. Normalises text (lowercase, strip punctuation, collapse whitespace,
   preserves Arabic U+0600-U+06FF glyphs).
3. Computes word-level edit distance → WER.
4. Writes `results/bench_<UTC_timestamp>.csv`.
5. Prints a summary.

Result quality targets (from the research phase):

| Model | Target Arabic MSA WER | Acceptable (investigate if worse) |
|---|---|---|
| Qwen3-ASR-1.7B | ≤ 20 % | 20–25 % |
| Whisper-Large-v3 | ≤ 28 % | 28–32 % |

If both exceed the "acceptable" range consistently, consider swapping in an
Arabic-specialist model (see [PLAN.md](../PLAN.md) for fallback options —
HARNESS, ArTST v3, NVIDIA Conformer-CTC-Arabic).

---

## DGX Spark deployment gotchas (Blackwell / sm_121 / aarch64)

These are the **real-world fixes** we had to apply when bringing the stack up
on a GB10 Grace Blackwell DGX Spark. Everything is already codified in the
Dockerfiles + entrypoints — this section exists so you know *why*, and so
anyone forking or upgrading the stack knows what to look out for.

### The "do not touch" rules

1. **Never `pip install vllm` on top of the NVIDIA container.** The NVIDIA
   container ships a CUDA-13 vLLM built for sm_121. Any PyPI vLLM wheel is
   CUDA-12 and will fail with:

   ```
   ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
   ```

   This is what happens if you install `qwen-asr[vllm]` — the `[vllm]` extra
   replaces the container's vLLM. Use `pip install qwen-asr` **without** the
   extra if you ever need it.

2. **Never `pip install torchaudio` from PyPI.** NVIDIA's custom torch
   (`2.11.0a0+...nv26.03`) has FP4 symbols PyPI torchaudio doesn't match, so
   you get:

   ```
   OSError: .../libtorchaudio.abi3.so: undefined symbol: torch_dtype_float4_e2m1fn_x2
   ```

   We avoid needing torchaudio at all by patching vLLM's
   `transformers_utils/processors/__init__.py` to make FunASR / FireRedASR2
   imports optional — we don't use either. The patch is in
   `serving/qwen/Dockerfile`.

3. **Never set `CUDA_VISIBLE_DEVICES=""` (empty string).** An empty value
   is treated as *hide all GPUs*, not *use all*. The `.env.example` keeps
   the variable commented out; leave it that way unless you actually want
   to pin a specific GPU.

### NGC container tag matters — use 26.03-py3

| NGC tag | Qwen3-ASR support? | Notes |
|---|---|---|
| `25.11-py3` | ❌ No — architecture not registered | Whisper works with `VLLM_ATTENTION_BACKEND=TRITON_ATTN` on T4/V100 but **NOT** on sm_121 (Triton lacks encoder-decoder cross-attn on Blackwell). |
| `26.01-py3` | ❌ No — still missing | |
| **`26.03-py3`** | ✅ **First tag with `Qwen3ASRForConditionalGeneration` + `Qwen3ASRRealtimeGeneration`** | Current default. |

To check what a given tag supports:

```bash
docker run --rm --entrypoint="" nvcr.io/nvidia/vllm:<tag> python -c "
from vllm.model_executor.models import ModelRegistry
for a in sorted(ModelRegistry.get_supported_archs()):
    if 'qwen3' in a.lower() or 'whisper' in a.lower(): print(a)
"
```

### Other 26.03-specific changes from 25.11

- `--task transcription` **was removed** from `vllm serve`. vLLM now
  auto-detects Whisper as a transcription model from its architecture. If you
  see `vllm: error: unrecognized arguments: --task transcription`, drop the
  flag from your entrypoint.
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` is **wrong for Blackwell**. Whisper's
  encoder self-attention and encoder/decoder cross-attention aren't
  implemented in the Triton backend on sm_121, so you hit:

  ```
  NotImplementedError: Encoder self-attention and encoder/decoder
  cross-attention are not implemented for TritonAttentionImpl
  ```

  Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (our Whisper entrypoint does).

### Confirming GPU passthrough

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

If this doesn't show your GB10, the NVIDIA Container Toolkit isn't set up;
install `nvidia-ctk` and configure the Docker runtime.

### HuggingFace cache surprises

- The `openai/whisper-large-v3` HF repo ships FP16 safetensors, FP32 `.bin`,
  and PyTorch `.bin` — pulling it naively with `huggingface-cli download`
  gives you ~24 GB instead of ~3 GB because all three are downloaded. Our
  `scripts/download_models.sh` passes `--include '*.safetensors' ...`
  filters to only pull what vLLM actually uses. If you already have the bloated
  version, reclaim the space with:

  ```bash
  rm -rf hf_cache/hub/models--openai--whisper-large-v3
  bash scripts/download_models.sh
  ```

- If you see `models--X--Y/blobs/<hash>.incomplete` hanging around, the
  download got interrupted. Running `download_models.sh` again is safe —
  `huggingface-cli` resumes from the partial files.

### Typical startup timeline (cold start, first `up`)

Watch `docker compose logs -f qwen` and expect, in order:

1. `~2s` — vLLM CLI argparse + config load
2. `~5–15s` — weights load from disk (SSD + unified memory helps)
3. `~30–90s` — CUDA graph compilation (first-run only; cached thereafter)
4. `Application startup complete.` — ready to serve

Total: 1–3 min per container. Subsequent restarts use the CUDA graph cache
and are ~30s.

### Process is restarting in a loop

If `docker compose ps` shows `Restarting (2)` indefinitely, the entrypoint is
failing before the HEALTHCHECK even gets to run. Look at the last ~80 lines of
the relevant service's log:

```bash
docker compose logs whisper --tail=80
```

Common culprits in order of likelihood:

- Unknown CLI argument (vLLM API evolves fast — `--task` was one victim)
- Missing Python package after an image bump (patch the Dockerfile, rebuild)
- OOM — `docker compose logs ... | grep -i "out of memory"`

---

## Troubleshooting

### `docker compose up` hangs on "Pulling"

The NVIDIA image is large (~6 GB). Run `docker pull nvcr.io/nvidia/vllm:26.03-py3`
separately with `--progress=plain` to see real progress.

### `/readyz` returns 503 with `unreachable`

```bash
docker compose ps
docker compose logs qwen --tail=100
docker compose logs whisper --tail=100
```

Common causes:
- vLLM still loading weights → wait. `Application startup complete` marks ready.
- OOM → reduce `QWEN_GPU_MEMORY_UTILIZATION` + `WHISPER_GPU_MEMORY_UTILIZATION`.
- aarch64 / sm_121 wheel mismatch → you've bypassed the NVIDIA container; rebuild from `serving/*/Dockerfile`.

### `sm_121 is not supported` errors

You're running PyPI-installed vLLM instead of the NVIDIA container. The
`serving/{qwen,whisper}/Dockerfile` in this repo pin the base image — make sure
`docker compose build` has succeeded without errors, and that nothing in
`pyproject.toml` got leaked into the serving containers (it isn't — just
sanity-check).

### WebSocket receives nothing after `session.accepted`

Common reasons:
- Not enough speech to trigger VAD (`VAD_MIN_SPEECH_MS` default 250 ms).
- Audio was not 16 kHz mono PCM16. Browser: the AudioWorklet handles this,
  but verify `AudioContext` was constructed with `sampleRate: 16000`.
- Mic was muted, or volume too low for the VAD threshold. Lower `VAD_THRESHOLD`
  (default 0.5, try 0.35) if you're testing a quiet speaker.

Check `bash scripts/logs.sh backend` for `ws.session_started` and the absence
of any errors.

### "Bengali/Tamil bleed" on Arabic

The **Qwen3-ASR-1.7B** we serve does not have this issue — that's a
NVIDIA Parakeet problem (see [PLAN.md](../PLAN.md)). If you ever swap in
Parakeet, expect it to output random Indic scripts on Arabic input and
benchmark before deploying.

### Whisper returns empty text for short clips

Whisper prepends its own VAD-like padding handling; clips under ~400 ms may
be ignored. Your VAD `min_speech_ms` default of 250 ms can produce these — bump
it to 400 ms if short segments come back empty.

---

## Security and hardening

Out of the box, this is **localhost-only dev tooling**. Before exposing it:

1. **Authentication**: wire an API key check into the FastAPI app (stub
   `Depends(get_current_user)` is the place — right now it's a no-op).
2. **Rate limiting**: add middleware (e.g. `slowapi`) or put it behind an
   API gateway.
3. **TLS**: terminate HTTPS at a reverse proxy (nginx/traefik/caddy). Browsers
   refuse `getUserMedia` on non-localhost without HTTPS.
4. **CORS**: lock `ALLOWED_ORIGINS` to your actual frontend host(s).
5. **Network**: keep the vLLM ports 8000 / 8001 internal-only (compose already
   isolates them on the `stt` bridge; don't map them to public interfaces).
6. **Logs**: set `LOG_TRANSCRIPT_CONTENT=false` (default) — transcripts can
   contain PII / sensitive speech.
7. **Upload limits**: the 50 MB cap in `backend/src/stt_backend/routes/transcribe.py`
   is the only guard against abuse. Tighten if your expected inputs are smaller.
8. **Model weight integrity**: pin `QWEN_REVISION` / `WHISPER_REVISION` to
   specific commit SHAs once you validate a version.

---

## FAQ

**Q: Why two models instead of just Qwen?**
Because accuracy varies per language, per speaker, and per domain. Having
Whisper next to Qwen lets you A/B compare live, fall back when one
misbehaves, and justify model-choice decisions with real numbers from
`bench.sh`. Long-term, you might pick one — or route by language.

**Q: What's the latency?**
End-to-end (mic speaks → transcript appears) is typically
0.5–1.5 s after a pause, dominated by VAD silence wait (default 500 ms) + the
vLLM transcription call. Shorter silence threshold = lower latency but more
spurious segments.

**Q: Can it handle continuous speech with no pauses?**
The `VAD_MAX_SEGMENT_MS` hard cut (default 15 s) guarantees you'll get
segments even from a monologue. Accuracy is best when natural pauses exist.

**Q: Can I add a third model?**
Yes — add a new service to `compose.yml` with its own Dockerfile, update
`ModelName` in `backend/src/stt_backend/schemas/transcription.py`, add a route
entry in `vllm_client.py`, and add an option in the frontend dropdown. The
backend is deliberately generic for this.

**Q: Does it support diarization or word timestamps?**
Not yet. The data shape is extensible (fields can be added to
`ServerSegment` / `TranscriptionResult`), but you'd need to swap in a
pipeline like WhisperX + pyannote. [PLAN.md](../PLAN.md) has notes on this.

**Q: Why Silero VAD specifically?**
Small (~2 MB ONNX), CPU-fast, works in 32 ms windows at 16 kHz, has no
external dependencies beyond torch, and is well-known as a solid default.
The wrapper is in `backend/src/stt_backend/services/vad.py` — trivial to
swap.

**Q: Can I run this on a non-DGX-Spark machine?**
Yes, any x86_64 or aarch64 Linux host with a recent NVIDIA GPU and the
Container Toolkit. The Dockerfiles pin the NVIDIA vLLM base image which
works on Ada (sm_89), Hopper (sm_90), and Blackwell (sm_100, sm_121).

**Q: How do I back up transcripts?**
Transcripts are ephemeral by default — they're returned to the caller and
not stored server-side (by design; less liability). If you need persistence,
add a handler to `routes/ws.py` that appends to a file / DB.

---

If anything here is wrong, stale, or unclear, open an issue or send a PR —
the guide is a living document.
