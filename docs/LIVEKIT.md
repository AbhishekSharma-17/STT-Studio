# LiveKit + vLLM ASR — complete integration guide

Use your self-hosted **Whisper-Large-v3** and/or **Qwen3-ASR-1.7B** vLLM
servers as the STT backend for LiveKit Agents. Covers both models, language
selection (fixed + auto), full constructor reference, a complete working
agent, Node/TS, network setup, and gotchas.

- Whisper server → `http://<host>:8001/v1`  · model id: `openai/whisper-large-v3`
- Qwen server    → `http://<host>:8000/v1`  · model id: `Qwen/Qwen3-ASR-1.7B`

---

## Table of contents

1. [How it fits together](#1-how-it-fits-together)
2. [Install](#2-install)
3. [Pick a model — Whisper vs Qwen](#3-pick-a-model--whisper-vs-qwen)
4. [Language selection](#4-language-selection)
   - [Fixed language per session](#41-fixed-language-per-session)
   - [Auto-detect language](#42-auto-detect-language)
   - [Per-participant / dynamic language](#43-per-participant--dynamic-language)
5. [Constructor reference](#5-constructor-reference)
6. [Minimal examples](#6-minimal-examples)
   - [Whisper, English](#61-whisper-english-captions-only)
   - [Whisper, Arabic](#62-whisper-arabic)
   - [Qwen, Arabic](#63-qwen-arabic)
   - [Auto-detect any language](#64-whisper-auto-detect-any-language)
   - [Domain biasing with `prompt`](#65-domain-biasing-with-prompt)
7. [Complete working voice agent](#7-complete-working-voice-agent-whisper--stt--llm--tts)
8. [Node.js / TypeScript](#8-nodejs--typescript)
9. [VAD tuning (chunk boundaries)](#9-vad-tuning-chunk-boundaries)
10. [Running both models and switching at runtime](#10-running-both-models-and-switching-at-runtime)
11. [Network topology](#11-network-topology)
12. [Troubleshooting](#12-troubleshooting)
13. [Language code cheat sheet](#13-language-code-cheat-sheet-iso-639-1)

---

## 1. How it fits together

```
┌───────────────────────┐     WebRTC      ┌──────────────────────────────┐
│ Browser / phone / SIP │ ───────────────▶│ LiveKit Room                 │
└───────────────────────┘                 └────────────────┬─────────────┘
                                                           │ audio tracks
                                                           ▼
                                           ┌──────────────────────────────┐
                                           │ AgentSession (your Python)   │
                                           │  ┌──────────────────────┐    │
                                           │  │ silero.VAD           │    │  ← decides "utterance ended"
                                           │  └─────────┬────────────┘    │
                                           │            ▼                 │
                                           │  ┌──────────────────────┐    │
                                           │  │ openai.STT           │    │  ← uses base_url override
                                           │  │  base_url=vLLM:8001  │    │
                                           │  └─────────┬────────────┘    │
                                           └────────────┼─────────────────┘
                                                        ▼ POST WAV
                                           ┌──────────────────────────────┐
                                           │ vLLM /v1/audio/transcriptions│
                                           │  Whisper-Large-v3  (8001)    │
                                           │  Qwen3-ASR-1.7B    (8000)    │
                                           └──────────────────────────────┘
```

**Two critical facts:**

1. LiveKit's `openai.STT` plugin in its **non-realtime mode** buffers an
   utterance and POSTs it as a WAV to `/v1/audio/transcriptions`. That is
   the endpoint vLLM exposes — so it works out of the box.
2. The plugin itself **does not chunk audio**; the `AgentSession` must have
   a VAD (Silero) that decides when to emit each utterance. Without VAD you
   get one giant unbounded transcription at end-of-call.

**Do not** set `use_realtime=True`. That switches the plugin to OpenAI's
`/realtime` WebSocket, which vLLM does not implement — you'll get 404.

---

## 2. Install

```bash
pip install "livekit-agents[silero]" livekit-plugins-openai
```

You also need LiveKit Cloud credentials (or a self-hosted LiveKit server):

```bash
export LIVEKIT_URL="wss://<your-project>.livekit.cloud"
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
```

Make sure your vLLM server is up (see `docs/QUICK_GUIDE.md`):

```bash
docker compose up -d whisper    # and/or:   docker compose up -d qwen
curl -s http://localhost:8001/v1/models | jq   # should return 200
```

---

## 3. Pick a model — Whisper vs Qwen

| | **Whisper-Large-v3** | **Qwen3-ASR-1.7B** |
|---|---|---|
| Port | 8001 | 8000 |
| Model id (for `model=`) | `openai/whisper-large-v3` | `Qwen/Qwen3-ASR-1.7B` |
| Languages supported | 99 | 30 |
| Best for | General multilingual, wide coverage | Arabic (MSA + dialects), East Asian languages |
| `srt` / `vtt` output | ✅ | ❌ (json / text / verbose_json only) |
| Word-level timestamps | ✅ (`timestamp_granularities[]`) | ⚠ partial (needs external aligner) |
| Default VRAM (this repo's config) | ~3 GB | ~4 GB |

**Rule of thumb:** Arabic / Chinese → Qwen. Everything else → Whisper. If
VRAM allows, run **both** and switch at runtime (see §10).

---

## 4. Language selection

`openai.STT` exposes two language controls:

| Param | Purpose |
|---|---|
| `language="<iso-639-1>"` | **Force** this language — every utterance is transcribed as it |
| `detect_language=True` | Let the model **auto-detect** per utterance |

They are mutually exclusive in practice — if you set both, `detect_language`
wins (the plugin sends no `language` form field, letting Whisper guess).

### 4.1 Fixed language per session

Best accuracy. Use when you know the language ahead of time.

```python
stt=openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    language="en",         # English
    use_realtime=False,
)
```

Other examples:
- `language="ar"` — Arabic (MSA)
- `language="es"` — Spanish
- `language="zh"` — Mandarin Chinese
- `language="hi"` — Hindi
- `language="fr"` — French
- `language="de"` — German
- `language="ja"` — Japanese

Full table at §13.

### 4.2 Auto-detect language

Use when the speaker's language is unknown or may switch between calls.

```python
stt=openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    detect_language=True,  # Whisper picks the language per utterance
    use_realtime=False,
)
```

**Accuracy cost.** Forcing a known language is always more accurate than
auto-detect, especially for short or noisy utterances. Only use auto-detect
if you really don't know the language in advance.

**Quirk of the plugin.** LiveKit only parses back Whisper's detected-language
metadata when you set `model="whisper-1"` (OpenAI's hosted model name). With
our self-hosted model id `openai/whisper-large-v3` the plugin requests
`response_format="json"` and ignores the detected-language field. The
**transcription still works correctly** — you just won't see *which*
language Whisper picked exposed as a first-class attribute on the result. If
you need that signal, either:

- Call `/v1/audio/transcriptions` yourself with `response_format=verbose_json`
  when you need the language label, or
- Pass `model="whisper-1"` to the plugin **and** add an alias on the server
  side so both names resolve. (The cleanest fix is to run with `--served-model-name whisper-1` on `vllm serve`; edit `serving/whisper/entrypoint.sh` if you want this.)

### 4.3 Per-participant / dynamic language

If different participants speak different languages, create one STT instance
per participant and bind them at `track_subscribed`:

```python
from livekit import rtc

PARTICIPANT_LANG = {
    "alice": "en",
    "bashir": "ar",
    "chen":  "zh",
}

def make_stt(lang: str):
    return openai.STT(
        base_url="http://localhost:8001/v1",
        api_key="x",
        model="openai/whisper-large-v3",
        language=lang,
        use_realtime=False,
    )

@ctx.room.on("track_subscribed")
def _on_track(track, pub, participant):
    if track.kind != rtc.TrackKind.KIND_AUDIO:
        return
    lang = PARTICIPANT_LANG.get(participant.identity, "en")
    # wire this STT to only this participant's track
    ...
```

(Full wiring in §7 — the pattern is `stt.StreamAdapter(stt=..., vad=...)`
stream-per-track.)

---

## 5. Constructor reference

```python
openai.STT(
    model: str = "gpt-4o-mini-transcribe",     # OVERRIDE — no match on vLLM otherwise
    base_url: str | None = None,               # override → your vLLM
    api_key: str | None = None,                # any non-empty string if vLLM has no --api-key
    language: str = "en",                      # ISO 639-1
    detect_language: bool = False,             # auto-detect
    prompt: str | None = None,                 # free-text biasing
    use_realtime: bool = False,                # LEAVE FALSE — vLLM has no /realtime
    turn_detection: ... = NOT_GIVEN,           # realtime-only (ignored)
    noise_reduction_type: str | None = NOT_GIVEN,  # realtime-only (ignored)
    client: openai.AsyncClient | None = None,  # pre-configured client (alternative to base_url/api_key)
)
```

| Param | Default | Needed for vLLM | Notes |
|---|---|---|---|
| `model` | `gpt-4o-mini-transcribe` | **Yes — override** | Must match `/v1/models` exactly |
| `base_url` | OpenAI | **Yes — override** | e.g. `http://localhost:8001/v1` |
| `api_key` | `$OPENAI_API_KEY` | Any non-empty string | vLLM ignores it unless `--api-key` is set |
| `language` | `"en"` | Optional | ISO 639-1 |
| `detect_language` | `False` | Optional | Skip `language` when using this |
| `prompt` | — | Optional | Max 224 tokens on Whisper |
| `use_realtime` | `False` | **Must be False** | `True` → 404 against vLLM |
| `turn_detection` | — | Ignored | Realtime-API only |
| `noise_reduction_type` | — | Ignored | Realtime-API only |
| `client` | — | Optional | Alternative to `base_url`/`api_key` |

---

## 6. Minimal examples

Each snippet assumes the surrounding `AgentSession` + `await session.start(...)`
boilerplate from §7.

### 6.1 Whisper, English, captions-only

```python
stt = openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    language="en",
    use_realtime=False,
)
```

### 6.2 Whisper, Arabic

```python
stt = openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    language="ar",
    use_realtime=False,
)
```

### 6.3 Qwen, Arabic

Point at port 8000 and use the Qwen model id. Qwen tends to beat Whisper on
Arabic (especially dialectal).

```python
stt = openai.STT(
    base_url="http://localhost:8000/v1",       # Qwen port
    api_key="x",
    model="Qwen/Qwen3-ASR-1.7B",
    language="ar",
    use_realtime=False,
)
```

### 6.4 Whisper, auto-detect any language

```python
stt = openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    detect_language=True,
    use_realtime=False,
)
```

### 6.5 Domain biasing with `prompt`

Improves recognition of uncommon words (product names, people's names,
jargon). Works on both Whisper and Qwen.

```python
stt = openai.STT(
    base_url="http://localhost:8001/v1",
    api_key="x",
    model="openai/whisper-large-v3",
    language="en",
    prompt=(
        "GenAI Protos, DGX Spark, Qwen3-ASR, Whisper-Large-v3, "
        "vLLM, Silero VAD, LiveKit Agents, Blackwell."
    ),
    use_realtime=False,
)
```

---

## 7. Complete working voice agent (Whisper — STT → LLM → TTS)

A full LiveKit agent you can run end-to-end. Answers in the same language it
hears by forcing a matching response style in the LLM prompt.

```python
# agent.py
"""LiveKit voice agent using self-hosted Whisper on vLLM for STT.

Run:
    export LIVEKIT_URL=wss://<project>.livekit.cloud
    export LIVEKIT_API_KEY=...
    export LIVEKIT_API_SECRET=...
    export OPENAI_API_KEY=...           # for the LLM
    export DEEPGRAM_API_KEY=...         # for TTS (swap freely)
    python agent.py dev
"""

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram

# --- STT: our self-hosted Whisper on vLLM ------------------------------
WHISPER_STT = openai.STT(
    base_url="http://localhost:8001/v1",       # vLLM Whisper
    api_key="not-used",                        # vLLM ignores unless --api-key set
    model="openai/whisper-large-v3",
    language="en",                             # or "ar", or detect_language=True
    use_realtime=False,                        # REQUIRED — vLLM has no /realtime
    prompt="GenAI Protos, DGX Spark, vLLM.",   # optional biasing
)

# --- VAD: decides where one utterance ends ----------------------------
VAD = silero.VAD.load(
    min_speech_duration=0.25,
    min_silence_duration=0.5,
    activation_threshold=0.5,
)


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a concise voice assistant. Reply in the same "
                "language the user spoke. Keep answers under 2 sentences."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=VAD,
        stt=WHISPER_STT,
        llm=openai.LLM(model="gpt-4o-mini"),            # hosted — swap for local if you like
        tts=deepgram.TTS(model="aura-asteria-en"),       # swap for any TTS plugin
    )

    await session.start(room=ctx.room, agent=Assistant())


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

Run:

```bash
python agent.py dev
```

Open the [LiveKit Agents Playground](https://agents-playground.livekit.io/)
(or your own client) pointed at the same room, speak, and you'll get voice
replies back.

### Variant: same agent, Qwen STT, Arabic

```python
QWEN_STT = openai.STT(
    base_url="http://localhost:8000/v1",       # vLLM Qwen
    api_key="not-used",
    model="Qwen/Qwen3-ASR-1.7B",
    language="ar",
    use_realtime=False,
)

# ... session = AgentSession(vad=VAD, stt=QWEN_STT, ...)
```

### Variant: auto-detect language, captions only (no LLM/TTS)

```python
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    stt = openai.STT(
        base_url="http://localhost:8001/v1",
        api_key="x",
        model="openai/whisper-large-v3",
        detect_language=True,
        use_realtime=False,
    )
    vad = silero.VAD.load()

    @ctx.room.on("track_subscribed")
    def on_track(track, pub, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        adapter = agents.stt.StreamAdapter(stt=stt, vad=vad)
        stream = adapter.stream()

        async def pump():
            audio_stream = rtc.AudioStream(track)
            async for frame in audio_stream:
                stream.push_frame(frame.frame)

        async def consume():
            async for event in stream:
                if event.type == agents.stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text
                    print(f"[{participant.identity}] {text}")

        ctx.schedule_task(pump())
        ctx.schedule_task(consume())
```

---

## 8. Node.js / TypeScript

Same idea — the JS plugin also accepts `baseURL`:

```typescript
import {
  AgentSession,
  cli,
  defineAgent,
  WorkerOptions,
} from "@livekit/agents";
import * as openai from "@livekit/agents-plugin-openai";
import * as silero from "@livekit/agents-plugin-silero";

export default defineAgent({
  entry: async (ctx) => {
    await ctx.connect();

    const session = new AgentSession({
      vad: await silero.VAD.load(),
      stt: new openai.STT({
        baseURL: "http://localhost:8001/v1",
        apiKey: "not-used",
        model: "openai/whisper-large-v3",
        language: "en",
        useRealtime: false,
      }),
      // llm: new openai.LLM({ model: "gpt-4o-mini" }),
      // tts: ...,
    });

    await session.start({ room: ctx.room, agent: /* your Agent */ });
  },
});

cli.runApp(new WorkerOptions({ agent: import.meta.url }));
```

Swap `baseURL` + `model` to `http://localhost:8000/v1` +
`"Qwen/Qwen3-ASR-1.7B"` for Qwen.

---

## 9. VAD tuning (chunk boundaries)

The "how often does Whisper get called" decision is made by **Silero VAD**,
not by Whisper. Tune it:

```python
vad = silero.VAD.load(
    min_speech_duration=0.25,      # ignore blips shorter than this
    min_silence_duration=0.5,      # emit an utterance after this much silence
    activation_threshold=0.5,      # 0..1 — higher = needs more confident speech
    # max_buffered_speech=30.0,    # hard-cut after N seconds
)
```

**Trade-offs:**

- ⬇ `min_silence_duration` (e.g. `0.3`) → faster partials, more API calls, less context per call, slightly lower accuracy on long sentences.
- ⬆ `min_silence_duration` (e.g. `1.0`) → fewer API calls, more context, better accuracy, slower perceived latency.
- ⬆ `activation_threshold` → fewer false positives in noise, risk of clipping quiet speech.

Whisper's sweet spot is utterances of **2–15 seconds**. Silero defaults put
most conversational speech in that range — usually no tuning needed.

---

## 10. Running both models and switching at runtime

If both vLLM containers are up (`docker compose up -d qwen whisper`), keep
one STT instance per model and swap based on language:

```python
whisper = openai.STT(
    base_url="http://localhost:8001/v1", api_key="x",
    model="openai/whisper-large-v3", language="en", use_realtime=False,
)
qwen = openai.STT(
    base_url="http://localhost:8000/v1", api_key="x",
    model="Qwen/Qwen3-ASR-1.7B", language="ar", use_realtime=False,
)

def pick_stt(lang: str):
    # Qwen's strengths: ar, zh, ja, ko
    if lang in {"ar", "zh", "ja", "ko"}:
        return qwen
    return whisper

session = AgentSession(vad=silero.VAD.load(), stt=pick_stt("ar"))
```

Both servers share the GPU — the `WHISPER_GPU_MEMORY_UTILIZATION=0.35` and
`QWEN_GPU_MEMORY_UTILIZATION=0.35` defaults leave room for both.

---

## 11. Network topology

| Where is your agent? | Set `base_url` to |
|---|---|
| Same host as vLLM, plain Python process | `http://localhost:8001/v1` |
| Different Docker container on the same host | `http://whisper:8001/v1` — after `docker network connect stt-inference_stt <your-agent-container>` |
| Another machine on the LAN | `http://<dgx-host-ip>:8001/v1` (open port 8001 in the host firewall) |
| LiveKit Cloud worker / public internet | Don't expose vLLM raw to the internet. Add auth (see `docs/QUICK_GUIDE.md` §9) + TLS, or tunnel with Tailscale / Cloudflared / ngrok |

Replace `8001` with `8000` for Qwen.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Agent logs show `404 /realtime` | `use_realtime=True` was set. vLLM has no realtime endpoint — set it to `False`. |
| `model_not_found` | `model=` must match `/v1/models` exactly: `openai/whisper-large-v3` or `Qwen/Qwen3-ASR-1.7B`. Not `whisper-1`, not `gpt-4o-transcribe`. |
| No transcripts at all | Missing `vad=` in `AgentSession`. Add `vad=silero.VAD.load()` — the STT plugin does not chunk on its own. |
| Transcripts cut off mid-word | `min_silence_duration` too low. Raise to `0.5`–`0.8` s. |
| First transcript takes 5–10 s | vLLM compiles a CUDA graph on first batch shape. Normal; subsequent calls are ~500 ms for ~10 s of audio. |
| `401 Unauthorized` | vLLM was started with `--api-key` — pass the same key as `api_key=...`. |
| Wrong language in output | You set `language="en"` but the speaker is using a different language. Either correct it or switch to `detect_language=True`. |
| Detected language not in result | Plugin only surfaces that when `model="whisper-1"`. Use `--served-model-name whisper-1` on vLLM or call the HTTP endpoint directly with `response_format=verbose_json` if you need it. |
| Empty transcripts on short speech | Whisper silently drops audio under ~400 ms. Raise Silero `min_speech_duration`. |
| `Connection refused` | vLLM container isn't up yet (warming up / compiling graphs). Check `docker compose ps` — wait for `healthy`. |

---

## 13. Language code cheat sheet (ISO 639-1)

What you put in `language=`. Whisper-Large-v3 supports all of these (99
languages total); Qwen3-ASR supports the bold-face subset of 30.

| Code | Language | | Code | Language | | Code | Language |
|---|---|---|---|---|---|---|---|
| **en** | English | | **ar** | Arabic | | **zh** | Chinese (Mandarin) |
| **es** | Spanish | | **fr** | French | | **de** | German |
| **it** | Italian | | **pt** | Portuguese | | **ru** | Russian |
| **ja** | Japanese | | **ko** | Korean | | **hi** | Hindi |
| **tr** | Turkish | | **nl** | Dutch | | **pl** | Polish |
| **id** | Indonesian | | **vi** | Vietnamese | | **th** | Thai |
| **uk** | Ukrainian | | **cs** | Czech | | **sv** | Swedish |
| **he** | Hebrew | | **fa** | Persian | | **ur** | Urdu |
| **ms** | Malay | | **ta** | Tamil | | **bn** | Bengali |
| ca | Catalan | | da | Danish | | fi | Finnish |
| el | Greek | | hu | Hungarian | | no | Norwegian |
| ro | Romanian | | sk | Slovak | | bg | Bulgarian |

Full Whisper language list:
<https://github.com/openai/whisper/blob/main/whisper/tokenizer.py> (search `LANGUAGES`).

---

## Sources / further reading

- [LiveKit — OpenAI STT plugin guide](https://docs.livekit.io/agents/models/stt/plugins/openai/)
- [LiveKit — Python plugin reference](https://docs.livekit.io/reference/python/livekit/plugins/openai/index.html)
- [LiveKit agents on GitHub](https://github.com/livekit/agents)
- [Whisper model card](https://huggingface.co/openai/whisper-large-v3)
- [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- Your own `docs/QUICK_GUIDE.md` — vLLM server lifecycle + monitoring
