# Quick Guide — Run the vLLM ASR servers standalone

**Goal:** bring up just one vLLM container (Whisper *or* Qwen) so another
application can hit the **OpenAI-compatible** HTTP endpoint directly. No
backend, no UI, no VAD — just the model server.

- Whisper-Large-v3 → `http://localhost:8001/v1`
- Qwen3-ASR-1.7B   → `http://localhost:8000/v1`

Both expose the same API shape as OpenAI's audio transcription API, so any
OpenAI SDK works unchanged — just swap `base_url`.

---

## 0. Prereqs (one time)

- Linux host with an NVIDIA GPU, CUDA ≥ 13, `docker` ≥ 25, `docker compose` v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and working (`docker run --rm --gpus all nvidia/cuda:13.0.0-base nvidia-smi` should print GPUs)
- ~20 GB free disk for the image, ~3 GB for Whisper weights (~4 GB for Qwen)
- You are in the repo root: `cd /path/to/STT_Inference`
- An `.env` file exists: `cp .env.example .env` (edit if you want a non-default port or model revision)

---

## 1. Run ONLY the Whisper vLLM server

```bash
# Build + start just the whisper container (compose skips qwen + backend)
docker compose up -d --build whisper

# Tail logs until you see "Uvicorn running on http://0.0.0.0:8001"
docker compose logs -f whisper
```

First start takes **1–3 minutes** — vLLM downloads `openai/whisper-large-v3`
into `./hf_cache/` and compiles CUDA graphs. Subsequent starts use the cache
(~30 s).

### Verify it's alive

```bash
# Should return a JSON list with one model id
curl -s http://localhost:8001/v1/models | jq
```

Expected:
```json
{
  "object": "list",
  "data": [{ "id": "openai/whisper-large-v3", "object": "model", ... }]
}
```

### Transcribe a file (smoke test)

```bash
curl -s http://localhost:8001/v1/audio/transcriptions \
  -F "model=openai/whisper-large-v3" \
  -F "file=@data/samples/jfk.wav" \
  -F "response_format=json" | jq
```

Expected: JSON with a `"text"` field containing the JFK quote.

### Stop it

```bash
docker compose stop whisper        # keep the container, just stop it
docker compose down whisper        # stop + remove the container
```

---

## 2. Run ONLY the Qwen3-ASR vLLM server

Same pattern, different service name and port:

```bash
docker compose up -d --build qwen
docker compose logs -f qwen

# Verify
curl -s http://localhost:8000/v1/models | jq

# Transcribe (Arabic)
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "file=@sample_ar.wav" \
  -F "language=ar" \
  -F "response_format=json" | jq
```

---

## 3. Use the endpoint from another application

The server speaks the exact same wire protocol as
`POST https://api.openai.com/v1/audio/transcriptions`. Point any OpenAI client
at `http://<host>:8001/v1` and set `api_key` to any non-empty string (it's
ignored unless you pass `--api-key` to `vllm serve`).

### Python — OpenAI SDK

```bash
pip install "openai>=1.0"
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",   # or http://<dgx-host>:8001/v1 from LAN
    api_key="not-used",
)

with open("meeting.mp3", "rb") as f:
    r = client.audio.transcriptions.create(
        model="openai/whisper-large-v3",   # must match /v1/models exactly
        file=f,
        response_format="verbose_json",    # gives you segment timestamps
        # language="en",                   # omit to auto-detect
        # temperature=0.0,
        # prompt="Optional biasing text",
    )

print(r.text)
for seg in r.segments:
    print(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")
```

### Node.js — OpenAI SDK

```bash
npm install openai
```

```javascript
import { OpenAI } from "openai";
import fs from "fs";

const client = new OpenAI({
  baseURL: "http://localhost:8001/v1",
  apiKey: "not-used",
});

const r = await client.audio.transcriptions.create({
  model: "openai/whisper-large-v3",
  file: fs.createReadStream("meeting.mp3"),
  response_format: "verbose_json",
});

console.log(r.text);
```

### curl (any language / shell)

```bash
# Whisper, with timestamps
curl -s http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer not-used" \
  -F "model=openai/whisper-large-v3" \
  -F "file=@meeting.mp3" \
  -F "response_format=verbose_json"

# Qwen, Arabic
curl -s http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer not-used" \
  -F "model=Qwen/Qwen3-ASR-1.7B" \
  -F "file=@sample_ar.wav" \
  -F "language=ar"
```

### Supported form fields

| Field | Whisper | Qwen3-ASR | Notes |
|---|:-:|:-:|---|
| `file` | ✅ | ✅ | wav, mp3, flac, ogg, m4a, webm, mp4, mpeg, mpga |
| `model` | ✅ | ✅ | Must match `/v1/models` id exactly |
| `language` | ✅ | ✅ | ISO 639-1 (`en`, `ar`, …). Omit for auto-detect. |
| `prompt` | ✅ | ✅ | Free-text biasing (Whisper: 224 tokens max) |
| `response_format` | `json` / `text` / `verbose_json` / `srt` / `vtt` | `json` / `text` / `verbose_json` | — |
| `temperature` | ✅ | ✅ | `0.0` = deterministic greedy |
| `timestamp_granularities[]` | `segment`, `word` | segment-level only | Whisper needs `response_format=verbose_json` to return them |

---

## 4. Calling from a different host or container

### From the same host, different process
Use `http://localhost:8001/v1` (or 8000 for Qwen).

### From another machine on the LAN
Use `http://<dgx-host-ip>:8001/v1`. The container already binds to `0.0.0.0`,
so nothing else to do — just make sure the host firewall allows the port.

### From another Docker container on the same host
Attach your app container to the compose network so it can resolve the
service by name:

```bash
docker network connect stt-inference_stt my-app
# then inside my-app: base_url=http://whisper:8001/v1
```

### Change the host port
Edit `.env`:

```
WHISPER_PORT=9001
```

Then `docker compose up -d whisper` again. The *container-internal* port
stays 8001; only the host-side mapping changes.

---

## 5. Common flags you may want to tune

All configured via `.env` (the entrypoint reads them at container start):

| Var | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Any HF Whisper checkpoint works (e.g. `openai/whisper-medium`) |
| `WHISPER_PORT` | `8001` | Host-side port |
| `WHISPER_GPU_MEMORY_UTILIZATION` | `0.35` | Fraction of VRAM reserved. Raise to `0.8` if Whisper is the only model running. |
| `WHISPER_MAX_NUM_SEQS` | `16` | Concurrent requests the server will batch |
| `QWEN_MODEL` | `Qwen/Qwen3-ASR-1.7B` | — |
| `QWEN_PORT` | `8000` | — |

After editing `.env`, recreate the container:

```bash
docker compose up -d whisper --force-recreate
```

---

## 6. Monitoring the container

Once the server is up, these are the commands you'll actually use day-to-day.

### Is it alive? (one-shot status)

```bash
docker compose ps                 # shows STATUS column: "Up X (healthy)"
docker ps --filter name=whisper   # same, if you're not in the repo dir
```

"healthy" means Docker's built-in healthcheck (`curl /v1/models` every 30 s —
see `serving/whisper/Dockerfile`) is passing. "starting" means vLLM is still
loading weights / compiling CUDA graphs — wait.

A 200 from the API itself is the real proof:

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8001/v1/models
# → HTTP 200
```

### Live logs

```bash
docker compose logs -f whisper              # follow
docker compose logs --tail=100 whisper      # last 100 lines, no follow
docker compose logs --since=5m whisper      # last 5 minutes
```

What to look for on a clean startup:
- `Loading safetensors checkpoint shards` — using the local HF cache ✅
- `Capturing CUDA graphs` — first-start compile, ~20–60 s
- `Uvicorn running on http://0.0.0.0:8001` — ready to serve
- Per-request lines like `POST /v1/audio/transcriptions HTTP/1.1 200` as
  traffic comes in

### GPU usage

```bash
nvidia-smi                         # one-shot snapshot
nvidia-smi -l 2                    # refresh every 2 s
watch -n 1 nvidia-smi              # same, via watch
```

Look at:
- `Memory-Usage` — Whisper-Large-v3 uses ~3–4 GB with default settings
- `GPU-Util %` — spikes to 70–100 % during a request, ~0 % idle
- The `stt-whisper` process in the bottom table

### Container CPU / memory / network

```bash
docker stats stt-whisper               # live table, Ctrl-C to exit
docker stats --no-stream stt-whisper   # one-shot snapshot
```

### Request-level timing

The `curl -w` format string is the quickest way to measure a single call:

```bash
curl -s -o /tmp/out.json \
  -w "status=%{http_code}  total=%{time_total}s  ttfb=%{time_starttransfer}s\n" \
  http://localhost:8001/v1/audio/transcriptions \
  -F "model=openai/whisper-large-v3" \
  -F "file=@data/samples/jfk.wav"
```

For continuous throughput / latency, `hey` is a one-binary load tester:

```bash
# 50 requests, 5 concurrent
hey -n 50 -c 5 -m POST \
  -T "multipart/form-data; boundary=X" \
  -D <(printf -- '--X\r\nContent-Disposition: form-data; name="model"\r\n\r\nopenai/whisper-large-v3\r\n--X--\r\n') \
  http://localhost:8001/v1/audio/transcriptions
```

(For realistic load testing with actual audio, the repo's `scripts/bench.sh`
does WER + throughput in one go.)

### Shell into the container

```bash
docker compose exec whisper bash
# inside: ps aux | grep vllm, nvidia-smi, etc.
```

### Resource / OOM inspection

```bash
docker inspect stt-whisper --format '{{.State.Health.Status}}'   # healthy|starting|unhealthy
docker inspect stt-whisper --format '{{.State.OOMKilled}}'       # true = GPU/host OOM
docker events --filter container=stt-whisper                     # live event stream
```

### Persistent watch script (optional)

```bash
# Prints status + /v1/models HTTP + GPU mem every 5 s. Ctrl-C to exit.
while true; do
  printf '%s  ' "$(date +%T)"
  printf 'health=%s  ' "$(docker inspect stt-whisper --format '{{.State.Health.Status}}' 2>/dev/null)"
  printf 'api=%s  ' "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/v1/models)"
  printf 'gpu_mem=%s\n' "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)MB"
  sleep 5
done
```

---

## 7. Run vLLM directly (no Docker Compose)

If you want to skip this repo entirely and just run the NVIDIA vLLM image by
hand:

```bash
docker run --rm -d \
  --name whisper \
  --gpus all --ipc=host --shm-size=8g \
  -p 8001:8001 \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  nvcr.io/nvidia/vllm:26.03-py3 \
  vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization 0.35 \
    --max-num-seqs 16
```

Notes:
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` is required on Blackwell (sm_121);
  Triton lacks encoder-decoder cross-attention there.
- For Qwen, add `--trust-remote-code` and pre-patch the processor import if
  you're not building from the `serving/qwen/` Dockerfile (see that file for
  the exact patch). The repo's Dockerfile is the path of least resistance.

---

## 8. Using with LiveKit Agents

vLLM's `/v1/audio/transcriptions` endpoint is a drop-in target for LiveKit's
`livekit-plugins-openai` STT plugin — just override `base_url` and `model`.
The TL;DR:

```python
from livekit.plugins import openai, silero
from livekit.agents import AgentSession

session = AgentSession(
    vad=silero.VAD.load(),                          # REQUIRED — chunks utterances
    stt=openai.STT(
        base_url="http://localhost:8001/v1",        # Whisper (8001) or Qwen (8000)
        api_key="not-used",
        model="openai/whisper-large-v3",            # or "Qwen/Qwen3-ASR-1.7B"
        language="en",                              # or "ar", or detect_language=True
        use_realtime=False,                         # REQUIRED — vLLM has no /realtime
    ),
)
```

Three rules: **override `model`**, **set `use_realtime=False`**, **wire a VAD**.

For the full walkthrough — complete working agent, both models side-by-side,
fixed-language vs auto-detect, per-participant language, Node.js/TS, VAD
tuning, language cheat sheet, troubleshooting — see
**[docs/LIVEKIT.md](LIVEKIT.md)**.

---

## 9. Adding auth (recommended before exposing beyond localhost)

Add `--api-key <your-key>` to the `vllm serve …` line in
`serving/whisper/entrypoint.sh` (or `serving/qwen/entrypoint.sh`):

```bash
exec vllm serve "${MODEL}" \
    --api-key "${VLLM_API_KEY}" \
    --revision "${REVISION}" \
    ...
```

Set `VLLM_API_KEY=...` in `.env`, rebuild (`docker compose up -d --build whisper`),
then clients must send `Authorization: Bearer <your-key>`.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `connection refused` on `curl /v1/models` | Container still warming up — watch `docker compose logs -f whisper` until "Uvicorn running on …" appears |
| `model_not_found` | `model` field doesn't match `/v1/models` id **exactly**. Use `openai/whisper-large-v3` / `Qwen/Qwen3-ASR-1.7B` verbatim. |
| 404 on `/v1/audio/transcriptions` | Wrong port — Whisper is **8001**, Qwen is **8000** |
| Empty `text` on short clips | Whisper silently drops audio < ~400 ms. Send longer clips. |
| First request is slow | vLLM compiles a CUDA graph on first use for each new batch shape. Subsequent calls are fast. |
| `libcudart.so.12 not found` on start | Don't `pip install vllm` on top of the NVIDIA image — the image ships CUDA 13; PyPI vLLM is CUDA 12. Use `nvcr.io/nvidia/vllm:26.03-py3` as-is. |
| `CUDA_VISIBLE_DEVICES=""` hides GPUs | Leave it unset in `.env` — empty string means "no GPUs visible" |

---

## TL;DR

```bash
# one-liner to run Whisper only and hit it from another app
docker compose up -d --build whisper

# in your other app:
OpenAI(base_url="http://<host>:8001/v1", api_key="x").audio.transcriptions.create(
    model="openai/whisper-large-v3", file=open("a.wav","rb"),
)
```

That's the whole thing.
