"""Render the STT Studio marketing writeup to a styled PDF."""

from pathlib import Path

import markdown
from weasyprint import HTML, CSS

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"
OUT_DIR.mkdir(exist_ok=True)
OUT_PDF = OUT_DIR / "STT_Studio_Marketing_Writeup.pdf"

MARKDOWN_SRC = r"""
# STT Studio
### Self-Hosted, Real-Time Speech-to-Text on NVIDIA DGX Spark

<p class="subtitle">A marketing & content briefing pack for the GenAI Protos marketing team</p>

---

## Product Overview — Elevator Pitch

**STT Studio** is a production-minded, **fully on-premise speech-to-text platform** that runs two best-in-class open-source ASR models — **Qwen3-ASR-1.7B** and **OpenAI Whisper-Large-v3** — side-by-side on a single NVIDIA DGX Spark. It ships as a one-command Docker stack with a friendly browser UI, an OpenAI-compatible REST API, and a low-latency WebSocket for live microphone transcription.

It is purpose-built for organizations that need **high-quality multilingual transcription — especially Modern Standard Arabic — without sending a single byte of audio to the public cloud.**

<div class="callout">
<strong>One-liner:</strong> <em>Live, multilingual, Arabic-grade speech-to-text that runs entirely on your own GPU — set up in one command.</em>
</div>

---

## Key Features

### Live Microphone Transcription
- Speak into the browser, see the transcript appear in **under 1.5 seconds** end-to-end.
- Built on `getUserMedia` → `AudioWorklet` → **WebSocket** streaming of raw PCM16 audio.
- Server-side **Silero VAD** (Voice Activity Detection) intelligently chunks speech on natural silence boundaries — no clipped words, no runaway segments.

### File Upload Transcription
- Drag-and-drop any audio/video file through the UI, or POST it to a REST endpoint.
- OpenAI-compatible `/v1/audio/transcriptions` — drop-in replacement for cloud APIs in existing code.

### True Multilingual Support
- **30 languages** via Qwen3-ASR (SOTA for Arabic, Chinese, and many low-resource languages).
- **99 languages** via Whisper-Large-v3 (broadest coverage, mature ecosystem).
- Right-to-left aware UI — Arabic script renders correctly out of the box.

### Dual-Model Side-by-Side Comparison
- Switch between Qwen3 and Whisper with a single click in the UI.
- Benchmark **Word Error Rate (WER)** across both on your own audio samples.
- Built-in failover — if one model struggles on a dialect or accent, the other is one flag away.

### Zero Data Leaves the Box
- Runs **100% locally** on your hardware — no outbound calls after first-time model download.
- Ideal for healthcare, legal, defense, government, finance, and any regulated environment.
- CORS-scoped, localhost-bound by default.

### OpenAI-Compatible API
- Point the OpenAI Python or Node SDK at `localhost:8000` — it just works.
- Migrating from the OpenAI Whisper API? **Change one URL, keep your code.**

### Built-in Observability & Benchmarking
- `/healthz`, `/readyz`, `/metrics` endpoints for liveness, readiness and Prometheus-style metrics.
- Structured JSON logs (structlog) — pipe directly into ELK, Loki or Datadog.
- `scripts/bench.sh` runs an end-to-end **WER benchmark** on your own audio samples.

### One-Command Deploy
- `bash scripts/setup.sh && bash scripts/start.sh` — that's it.
- Three containers (Qwen vLLM + Whisper vLLM + FastAPI backend) orchestrated by Docker Compose.
- Hot-reload dev mode for engineers, lean prod mode for operators.

---

## Technical Highlights

| Layer | Technology | Why it matters |
|---|---|---|
| Primary ASR model | Qwen3-ASR-1.7B (Alibaba, Apache 2.0) | State-of-the-art open-source ASR, streaming-capable, 30 languages including Arabic MSA. |
| Secondary ASR model | Whisper-Large-v3 (OpenAI, MIT) | Industry benchmark, 99 languages, mature tooling. |
| Inference serving | **vLLM** × 2 (NGC `26.03-py3`) | Production-grade LLM serving with paged attention, CUDA graphs, OpenAI-compatible HTTP API. |
| Backend | FastAPI + Starlette + Pydantic v2 + httpx + structlog | Async, type-safe, low-overhead routing between browser and vLLM. |
| Voice Activity Detection | Silero VAD v5 (MIT) | Server-side chunking on 500 ms silence after ≥250 ms speech, with 15 s max segment — tuned for Arabic utterance lengths. |
| Streaming protocol | WebSocket + raw PCM16 @ 16 kHz mono | No WebRTC, no proprietary protocol — simple, debuggable, firewall-friendly. |
| Audio capture | Browser `AudioWorklet` (Float32 → Int16) | Runs on the audio rendering thread — glitch-free, no main-thread jank. |
| Hardware | NVIDIA DGX Spark (GB10 Blackwell, sm_121, aarch64) | 128 GB unified memory, ~8 GB VRAM consumed by both models combined — massive headroom. |
| Container runtime | NVIDIA vLLM container `26.03-py3` | First NGC tag with native `Qwen3ASRForConditionalGeneration` support on Blackwell. |
| Orchestration | Docker Compose v2 + Makefile + shell scripts | Reproducible, no Kubernetes cognitive tax for single-node deployments. |
| Dev-experience | `uv`, `ruff`, `mypy`, `pytest`, smoke tests, WER bench | Enforced lint + types + tests — production-grade Python hygiene. |

### Clever engineering calls worth highlighting in a technical post

1. **Chunk-on-silence instead of continuous streaming.** Rather than fight the known Qwen3-ASR realtime-endpoint quality bug, we chunk on natural silence boundaries via VAD and send each utterance as a fresh transcription call. Result: *sub-1.5 s perceived latency with full-utterance quality* — no proprietary ML streaming code required.
2. **Two models, one GPU.** Both models fit comfortably together; switching is a config flag, not a redeploy.
3. **Blackwell / sm_121 pitfalls solved.** The Dockerfiles encode non-obvious fixes: Flash-Attention backend for Whisper (Triton lacks encoder-decoder cross-attention on sm_121), avoiding PyPI torchaudio (ABI-incompatible with NVIDIA's FP4 torch), vLLM patch for processor imports. **This alone saves a new team ~2 weeks of debugging.**
4. **OpenAI wire-compatible from day one.** Any client that speaks the OpenAI audio API — Python SDK, Node SDK, curl, Postman — works unmodified against either model.

---

## Benefits

### For CTOs & Architects
- **Cloud-cost independence** — no per-minute transcription fees, predictable capex.
- **Data sovereignty** — fully compliant with on-prem and air-gapped requirements (HIPAA, GDPR Article 44, UAE PDPL, KSA PDPL).
- **Vendor neutrality** — open models, open licenses (Apache 2.0 / MIT), no lock-in.

### For Product & Engineering Teams
- **Drop-in OpenAI replacement** — migrate existing Whisper API code in minutes.
- **Reproducible stack** — same stack on a developer laptop (with a GPU) as in production.
- **Benchmark-ready** — built-in WER tooling lets teams pick the right model *for their own audio*, not generic leaderboards.

### For Operations
- **One-box deployment** — a single DGX Spark runs the whole thing.
- **Healthchecks + metrics + structured logs** — observability built in, not bolted on.
- **Low maintenance surface** — three containers, no Kubernetes, no managed-service dependencies.

### For End Users
- **Real-time captions in the browser** — click, speak, read.
- **Accurate Arabic transcription** — MSA and major dialects handled with care.
- **No uploads to third parties** — private by design.

---

## Suggested Taglines & Headlines

- *"Arabic-grade speech-to-text, entirely on your own GPU."*
- *"One box. Two models. Ninety-nine languages. Zero data leakage."*
- *"Real-time transcription with the quality of a cloud API and the privacy of a local network."*
- *"From silence to subtitle in under 1.5 seconds — on a single DGX Spark."*

---

## Architecture Diagram — Image-Generation Prompt

Paste this into your image-gen model of choice (Midjourney, DALL·E 3, Imagen, Flux, Stable Diffusion with a technical LoRA):

<div class="prompt-box">

<strong>Prompt:</strong>

<em>A clean, modern enterprise software architecture diagram, isometric 3D perspective, flat vector style with soft gradients, dark navy and teal color palette with orange accent highlights, white background, sharp technical illustration.</em>

<em>Top layer: a web browser window labeled 'Browser UI' showing a microphone icon, a waveform, and a live transcript in both English and Arabic (right-to-left). An arrow labeled 'WebSocket + REST (port 3000)' flows downward.</em>

<em>Middle layer: a single rounded rectangle labeled 'FastAPI Backend' containing four small icons — a microphone (WebSocket /ws/transcribe), an upload arrow (POST /transcribe), a heart-pulse (healthz/readyz/metrics), and a soundwave with scissors (Silero VAD chunker). Two arrows flow downward from it, labeled 'http://qwen:8000' and 'http://whisper:8001'.</em>

<em>Lower-middle layer: two parallel glowing server-rack rectangles side by side. The left one labeled 'vLLM Server #1 — Qwen3-ASR-1.7B, 30 languages, Arabic-primary'. The right one labeled 'vLLM Server #2 — Whisper-Large-v3, 99 languages'. Both display 'OpenAI-compatible /v1/audio/transcriptions'.</em>

<em>Bottom layer: a large stylized NVIDIA DGX Spark box with green GPU lighting, labeled 'NVIDIA DGX Spark · GB10 Blackwell · 128 GB Unified Memory · sm_121 · aarch64'. Both vLLM servers sit on top of it, sharing the GPU.</em>

<em>Around the whole stack: a subtle dashed boundary labeled 'On-Premise — No Data Leaves The Box' with a padlock icon in the corner.</em>

<em>Typography: clean sans-serif (Inter-like), generous whitespace, technical but friendly. Style: Stripe-docs / Linear-blog / Vercel aesthetic. No cartoon characters, no people, no clutter.</em>

</div>

**Tip for the designer:** if the image-gen model struggles with the layered layout, generate it in two passes — first the DGX Spark base layer with both vLLM servers on top, then composite the Browser → FastAPI top portion in Figma.

---

## Content Angles — Post Ideas for the Marketing Team

1. **"Why we run Qwen3-ASR and Whisper side-by-side"** — dual-model architecture, benchmarking story.
2. **"Getting Arabic speech-to-text right: what the cloud APIs miss"** — Arabic / MSA quality angle.
3. **"Deploying vLLM on DGX Spark Blackwell: the 5 gotchas that cost us a week"** — developer blog, SEO goldmine.
4. **"Replacing the OpenAI Whisper API in your product without changing a line of client code"** — migration playbook.
5. **"Real-time, sub-1.5-second browser transcription — no WebRTC, no magic, just VAD and WebSockets"** — engineering deep-dive.
6. **"Data sovereignty in speech AI: why regulated industries are bringing ASR back on-prem"** — compliance / thought-leadership angle.

---

<p class="footer-note">© 2026 GenAI Protos · STT Studio is released under Apache License 2.0 · Bundled models retain their original licenses (Qwen3-ASR: Apache 2.0, Whisper: MIT).</p>
"""

CSS_STYLE = """
@page {
    size: A4;
    margin: 20mm 18mm 22mm 18mm;
    @bottom-center {
        content: "STT Studio — Marketing Writeup · Page " counter(page) " of " counter(pages);
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-size: 9pt;
        color: #94a3b8;
    }
}

html {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    color: #0f172a;
    font-size: 10.5pt;
    line-height: 1.55;
}

body {
    margin: 0;
}

h1 {
    font-size: 28pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 4pt 0;
    letter-spacing: -0.02em;
}

h1 + h3 {
    font-size: 14pt;
    font-weight: 500;
    color: #475569;
    margin-top: 0;
    margin-bottom: 6pt;
    letter-spacing: -0.01em;
}

.subtitle {
    color: #64748b;
    font-size: 10pt;
    font-style: italic;
    margin-top: 0;
    margin-bottom: 18pt;
}

h2 {
    font-size: 18pt;
    font-weight: 700;
    color: #0b3b63;
    margin-top: 22pt;
    margin-bottom: 8pt;
    padding-bottom: 4pt;
    border-bottom: 2px solid #e2e8f0;
    letter-spacing: -0.01em;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    font-weight: 700;
    color: #0b3b63;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 10pt;
    margin-bottom: 3pt;
}

p {
    margin: 6pt 0;
}

strong {
    color: #0b3b63;
    font-weight: 700;
}

em {
    color: #334155;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 18pt 0;
}

ul, ol {
    margin: 6pt 0 8pt 0;
    padding-left: 20pt;
}

li {
    margin: 3pt 0;
}

code {
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
    background: #f1f5f9;
    color: #0b3b63;
    padding: 1pt 4pt;
    border-radius: 3pt;
    font-size: 9.5pt;
}

pre {
    background: #0f172a;
    color: #e2e8f0;
    padding: 10pt;
    border-radius: 6pt;
    font-size: 9pt;
    overflow-x: auto;
    line-height: 1.45;
}

pre code {
    background: transparent;
    color: inherit;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th {
    background: #0b3b63;
    color: white;
    padding: 6pt 8pt;
    text-align: left;
    font-weight: 600;
    font-size: 9.5pt;
}

td {
    padding: 5pt 8pt;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f8fafc;
}

.callout {
    background: linear-gradient(90deg, #fff7ed 0%, #ffedd5 100%);
    border-left: 4px solid #f97316;
    padding: 10pt 14pt;
    border-radius: 4pt;
    margin: 12pt 0;
    font-size: 10.5pt;
}

.callout strong {
    color: #9a3412;
}

.prompt-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0b3b63;
    padding: 12pt 14pt;
    border-radius: 4pt;
    margin: 10pt 0;
    font-size: 10pt;
    line-height: 1.55;
}

.prompt-box strong {
    color: #0b3b63;
    display: block;
    margin-bottom: 4pt;
}

.prompt-box em {
    display: block;
    margin-bottom: 6pt;
    color: #334155;
    font-style: italic;
}

.footer-note {
    font-size: 8.5pt;
    color: #94a3b8;
    text-align: center;
    margin-top: 22pt;
    font-style: italic;
}
"""


def main() -> None:
    html_body = markdown.markdown(
        MARKDOWN_SRC,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>STT Studio — Marketing Writeup</title>
</head>
<body>
{html_body}
</body>
</html>
"""

    HTML(string=full_html).write_pdf(
        str(OUT_PDF),
        stylesheets=[CSS(string=CSS_STYLE)],
    )
    print(f"Wrote: {OUT_PDF}")


if __name__ == "__main__":
    main()
