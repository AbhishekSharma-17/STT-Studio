/**
 * STT Studio — frontend controller.
 *
 * Audio flow:
 *   getUserMedia → AudioContext → AudioWorkletNode (resamples to 16 kHz PCM16)
 *   → WebSocket binary frames → backend VAD → vLLM → transcription.segment
 */

const TARGET_RATE = 16000;

const $ = (id) => document.getElementById(id);

const els = {
  stackStatus: $("stack-status"),

  // sidebar
  modelSeg: $("model-seg"),
  langSel: $("language"),
  autoClear: $("auto-clear"),
  statSegments: $("stat-segments"),
  statEmpty: $("stat-empty"),
  statRtf: $("stat-rtf"),
  statAudio: $("stat-audio"),

  // live card
  recordBtn: $("record-btn"),
  recordLabel: $("record-label"),
  recordStatus: $("record-status"),
  clearBtn: $("clear-btn"),
  copyBtn: $("copy-btn"),
  exportBtn: $("export-btn"),
  levelFill: $("level-fill"),
  levelPeak: $("level-peak"),
  telemetry: $("live-telemetry"),
  transcript: $("transcript"),
  transcriptCount: $("transcript-count"),

  // batch
  uploadForm: $("upload-form"),
  fileInput: $("file-input"),
  dropzone: $("dropzone"),
  dropzoneLabel: $("dropzone-label"),
  fileModelSeg: $("file-model-seg"),
  fileStatus: $("file-status"),
  fileResult: $("file-result"),

  // debug
  debugLog: $("debug-log"),
};

const state = {
  ws: null,
  audioCtx: null,
  workletNode: null,
  micStream: null,
  source: null,
  isRecording: false,

  liveModel: "qwen3-asr",
  fileModel: "qwen3-asr",

  segmentCount: 0,
  emptyCount: 0,
  chunksSent: 0,
  bytesSent: 0,
  inputSampleRate: null,
  outputSampleRate: TARGET_RATE,

  rtfSum: 0,
  rtfCount: 0,
};

// ---------- Helpers ------------------------------------------------------

function setPill(el, text, kind = "neutral", withPulse = false) {
  el.innerHTML = (withPulse ? '<span class="pulse"></span>' : "") + text;
  el.className = `pill pill-${kind}`;
}

function setRecordUi(label, kind = "neutral") {
  els.recordLabel.textContent = label;
  setPill(els.recordStatus, kind === "ok" ? "recording" : kind === "err" ? "error" : label, kind,
          kind === "ok" || kind === "active");
}

function setSegmented(segEl, value) {
  segEl.querySelectorAll(".seg-btn").forEach((b) => {
    const active = b.dataset.value === value;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active ? "true" : "false");
  });
}

function writeStatsSidebar() {
  const rtf = state.rtfCount ? (state.rtfSum / state.rtfCount).toFixed(2) : "—";
  const audioS = state.outputSampleRate
    ? (state.bytesSent / (state.outputSampleRate * 2)).toFixed(1)
    : "0.0";
  els.statSegments.textContent = state.segmentCount;
  els.statEmpty.textContent = state.emptyCount;
  els.statRtf.textContent = rtf;
  els.statAudio.textContent = `${audioS} s`;
}

function writeTelemetry() {
  const sentSec = state.outputSampleRate
    ? (state.bytesSent / (state.outputSampleRate * 2)).toFixed(1)
    : "0.0";
  const rtf = state.rtfCount ? (state.rtfSum / state.rtfCount).toFixed(2) : "—";
  const map = {
    session:   state.ws?._sessionId || "—",
    "in-rate": state.inputSampleRate ? `${state.inputSampleRate} Hz` : "—",
    "out-rate": `${state.outputSampleRate} Hz`,
    sent:      `${state.chunksSent} chunks · ${sentSec} s`,
    segments:  `${state.segmentCount}${state.emptyCount ? ` (${state.emptyCount} empty)` : ""}`,
    rtf,
  };
  els.telemetry.querySelectorAll("[data-k]").forEach((s) => {
    const k = s.dataset.k;
    if (k in map) s.textContent = map[k];
  });
  els.transcriptCount.textContent = `${state.segmentCount} segments`;
  writeStatsSidebar();
}

function debug(kind, payload) {
  const ts = new Date().toISOString().slice(11, 23);
  const line = `[${ts}] ${kind} ${typeof payload === "string" ? payload : JSON.stringify(payload)}\n`;
  els.debugLog.textContent += line;
  els.debugLog.scrollTop = els.debugLog.scrollHeight;
}

function clearTranscript() {
  els.transcript.innerHTML = "";
  state.segmentCount = 0;
  state.emptyCount = 0;
  state.rtfSum = 0;
  state.rtfCount = 0;
  writeTelemetry();
}

function appendSegment(seg) {
  const node = document.createElement("div");
  const text = (seg.text || "").trim();
  const isArabic = seg.language === "ar" || /[\u0600-\u06FF]/.test(text);
  const empty = text.length === 0;
  const isError = (seg.reason || "").startsWith("error:");
  node.className = `segment ${isArabic ? "rtl" : "ltr"}` + (empty ? " empty" : "") + (isError ? " error" : "");

  const rtf = seg.audio_duration_ms > 0
    ? seg.upstream_duration_ms / seg.audio_duration_ms
    : NaN;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML = [
    `<span class="chip">#${seg.segment_id}</span>`,
    `<span class="chip">${seg.model}</span>`,
    `<span class="chip">${seg.language || "?"}</span>`,
    `<span class="chip">audio ${(seg.audio_duration_ms / 1000).toFixed(2)}s</span>`,
    `<span class="chip">up ${seg.upstream_duration_ms}ms</span>`,
    `<span class="chip">RTF ${Number.isFinite(rtf) ? rtf.toFixed(2) : "—"}</span>`,
    `<span class="chip chip-muted">${seg.reason}</span>`,
  ].join("");

  const body = document.createElement("div");
  body.className = "text";
  if (isError) {
    body.textContent = `vLLM error: ${seg.reason.replace("error:", "")}`;
  } else if (empty) {
    body.textContent =
      "Empty — model returned no text. Check: mic level, selected language matches speech, utterance > 400 ms.";
  } else {
    body.textContent = text;
  }

  node.appendChild(meta);
  node.appendChild(body);
  els.transcript.appendChild(node);
  els.transcript.scrollTop = els.transcript.scrollHeight;

  state.segmentCount += 1;
  if (empty) state.emptyCount += 1;
  if (Number.isFinite(rtf)) {
    state.rtfSum += rtf;
    state.rtfCount += 1;
  }
  writeTelemetry();
}

function allTranscriptText() {
  return Array.from(els.transcript.querySelectorAll(".segment"))
    .filter((s) => !s.classList.contains("empty") && !s.classList.contains("error"))
    .map((s) => s.querySelector(".text").textContent.trim())
    .filter(Boolean)
    .join("\n");
}

// ---------- Stack readiness ----------------------------------------------

async function checkStackStatus() {
  try {
    const r = await fetch("/readyz");
    const body = await r.json();
    if (r.ok && body.ready) {
      setPill(els.stackStatus, "stack ready", "ok", true);
    } else {
      const offline = Object.entries(body.upstreams || {})
        .filter(([, v]) => !v.ok)
        .map(([k]) => k)
        .join(", ") || "unknown";
      setPill(els.stackStatus, `not ready · ${offline}`, "err", true);
    }
  } catch {
    setPill(els.stackStatus, "backend unreachable", "err", true);
  }
}

// ---------- Recording ----------------------------------------------------

async function startRecording() {
  if (state.isRecording) return;
  if (els.autoClear.checked) clearTranscript();
  state.chunksSent = 0;
  state.bytesSent = 0;
  writeTelemetry();

  setRecordUi("connecting…", "active");

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch (err) {
    setRecordUi("mic denied", "err");
    debug("mic.error", err.message);
    return;
  }

  const audioCtx = new AudioContext({ sampleRate: TARGET_RATE });

  try {
    await audioCtx.audioWorklet.addModule("/worklet.js");
  } catch (err) {
    setRecordUi("worklet load failed", "err");
    debug("worklet.error", err.message);
    stream.getTracks().forEach((t) => t.stop());
    return;
  }

  const source = audioCtx.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(audioCtx, "pcm16-processor");
  source.connect(worklet);
  // Silent sink so the worklet stays in the audio graph but no echo is audible.
  const sink = audioCtx.createGain();
  sink.gain.value = 0;
  worklet.connect(sink).connect(audioCtx.destination);

  const wsUrl = new URL("/ws/transcribe", window.location.href);
  wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    const payload = {
      type: "session.start",
      model: state.liveModel,
      language: els.langSel.value || null,
      sample_rate: TARGET_RATE,
    };
    ws.send(JSON.stringify(payload));
    debug("→", payload);
    setRecordUi("listening…", "active");
  });

  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    debug("←", msg);
    if (msg.type === "session.accepted") {
      ws._sessionId = msg.session_id;
      writeTelemetry();
      setRecordUi("listening", "ok");
    } else if (msg.type === "transcription.segment") {
      appendSegment(msg);
    } else if (msg.type === "session.ended") {
      setRecordUi(`done · ${msg.total_segments} segments`, "ok");
    } else if (msg.type === "error") {
      setRecordUi(`error: ${msg.code}`, "err");
    }
  });

  ws.addEventListener("error", () => { debug("ws.error", "transport error"); setRecordUi("ws error", "err"); });
  ws.addEventListener("close", (ev) => {
    if (state.isRecording) setRecordUi("disconnected", "err");
    debug("ws.close", `code=${ev.code} reason=${ev.reason || "(none)"}`);
    teardownAudio();
  });

  worklet.port.onmessage = (ev) => {
    const { type } = ev.data;
    if (type === "pcm" && ws.readyState === WebSocket.OPEN) {
      ws.send(ev.data.buffer);
      state.chunksSent += 1;
      state.bytesSent += ev.data.buffer.byteLength;
      writeTelemetry();
    } else if (type === "level") {
      const p = Math.min(1, ev.data.peak);
      const pct = Math.min(100, Math.round(p * 180));
      els.levelFill.style.width = pct + "%";
      els.levelPeak.textContent = p > 1e-4 ? (20 * Math.log10(p)).toFixed(1) + " dB" : "-∞ dB";
    } else if (type === "rate") {
      state.inputSampleRate = ev.data.inputSampleRate;
      state.outputSampleRate = ev.data.outputSampleRate;
      writeTelemetry();
      debug("rate", ev.data);
    }
  };

  Object.assign(state, {
    ws, audioCtx, workletNode: worklet, micStream: stream, source,
    isRecording: true,
  });

  els.recordBtn.classList.add("recording");
  els.recordBtn.setAttribute("aria-label", "Stop recording");
}

function teardownAudio() {
  state.isRecording = false;
  els.recordBtn.classList.remove("recording");
  els.recordBtn.setAttribute("aria-label", "Start recording");
  els.recordLabel.textContent = "Tap to record";
  els.levelFill.style.width = "0%";
  els.levelPeak.textContent = "-∞ dB";

  try { state.workletNode?.disconnect(); } catch {}
  try { state.source?.disconnect(); } catch {}
  if (state.micStream) for (const t of state.micStream.getTracks()) t.stop();
  if (state.audioCtx && state.audioCtx.state !== "closed") state.audioCtx.close().catch(() => {});
  state.workletNode = null;
  state.source = null;
  state.micStream = null;
  state.audioCtx = null;
}

function stopRecording() {
  if (!state.isRecording) return;
  const ws = state.ws;
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: "session.commit" })); } catch {}
    debug("→", { type: "session.commit" });
  }
  setRecordUi("finishing…", "active");
  teardownAudio();
}

// ---------- Event wiring --------------------------------------------------

els.recordBtn.addEventListener("click", () => {
  if (state.isRecording) stopRecording(); else startRecording();
});

els.clearBtn.addEventListener("click", clearTranscript);

els.copyBtn.addEventListener("click", async () => {
  const text = allTranscriptText();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const prev = els.copyBtn.textContent;
    els.copyBtn.textContent = "Copied!";
    setTimeout(() => (els.copyBtn.textContent = prev), 1200);
  } catch {
    alert("Clipboard access was blocked by the browser.");
  }
});

els.exportBtn.addEventListener("click", () => {
  const text = allTranscriptText();
  if (!text) return;
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `transcript-${new Date().toISOString().replace(/[:.]/g, "-")}.txt`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(a.href);
});

// Segmented model selectors
els.modelSeg.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".seg-btn");
  if (!btn) return;
  state.liveModel = btn.dataset.value;
  setSegmented(els.modelSeg, state.liveModel);
});
els.fileModelSeg.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".seg-btn");
  if (!btn) return;
  state.fileModel = btn.dataset.value;
  setSegmented(els.fileModelSeg, state.fileModel);
});

// Drag + drop file upload
["dragenter", "dragover"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); }),
);
["dragleave", "drop"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("drag"); }),
);
els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files?.[0]) {
    els.fileInput.files = e.dataTransfer.files;
    els.dropzoneLabel.textContent = e.dataTransfer.files[0].name;
  }
});
els.fileInput.addEventListener("change", () => {
  const f = els.fileInput.files?.[0];
  if (f) els.dropzoneLabel.textContent = f.name;
});

els.uploadForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const file = els.fileInput.files?.[0];
  if (!file) return;

  setPill(els.fileStatus, "uploading…", "active", true);
  els.fileResult.textContent = "";

  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", state.fileModel);
  if (els.langSel.value) fd.append("language", els.langSel.value);

  try {
    const t0 = performance.now();
    const r = await fetch("/transcribe", { method: "POST", body: fd });
    const dt = Math.round(performance.now() - t0);
    const body = await r.json();
    if (r.ok) {
      setPill(els.fileStatus, `ok · ${body.duration_ms}ms upstream / ${dt}ms wall`, "ok");
      els.fileResult.textContent = body.text || "(empty — vLLM returned no text for this file)";
      els.fileResult.dir = /[\u0600-\u06FF]/.test(body.text || "") ? "rtl" : "ltr";
    } else {
      setPill(els.fileStatus, `error ${r.status}`, "err");
      els.fileResult.textContent = JSON.stringify(body, null, 2);
    }
  } catch (err) {
    setPill(els.fileStatus, "network error", "err");
    els.fileResult.textContent = String(err);
  }
});

// Custom language dropdown → syncs to hidden <select id="language">
(function initLanguageDropdown() {
  const dd = document.getElementById("language-dropdown");
  if (!dd) return;
  const trigger = dd.querySelector(".dropdown-trigger");
  const menu = dd.querySelector(".dropdown-menu");
  const valueEl = dd.querySelector(".dropdown-value");
  const hiddenSel = dd.querySelector("select");

  const open = () => {
    menu.hidden = false;
    requestAnimationFrame(() => dd.classList.add("open"));
    trigger.setAttribute("aria-expanded", "true");
  };
  const close = () => {
    dd.classList.remove("open");
    trigger.setAttribute("aria-expanded", "false");
    setTimeout(() => { if (!dd.classList.contains("open")) menu.hidden = true; }, 180);
  };
  const toggle = () => (dd.classList.contains("open") ? close() : open());

  const select = (li) => {
    const value = li.dataset.value;
    menu.querySelectorAll("li").forEach((x) => x.removeAttribute("aria-selected"));
    li.setAttribute("aria-selected", "true");
    const flag = li.querySelector(".dd-flag")?.textContent ?? "";
    const label = li.querySelector(".dd-label")?.textContent ?? "";
    const code = li.querySelector(".dd-code")?.textContent ?? "";
    valueEl.innerHTML =
      `<span class="dd-flag">${flag}</span>` +
      `<span class="dd-label">${label}</span>` +
      (code ? `<span class="dd-code">${code}</span>` : "");
    hiddenSel.value = value;
    hiddenSel.dispatchEvent(new Event("change", { bubbles: true }));
    close();
  };

  trigger.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
  menu.addEventListener("click", (e) => {
    const li = e.target.closest("li[role='option']");
    if (li) select(li);
  });
  document.addEventListener("click", (e) => {
    if (!dd.contains(e.target)) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && dd.classList.contains("open")) close();
  });
})();

// Keyboard shortcut: spacebar toggles recording (when not typing in an input)
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  e.preventDefault();
  if (state.isRecording) stopRecording(); else startRecording();
});

// Initial render
writeTelemetry();
checkStackStatus();
setInterval(checkStackStatus, 15000);
