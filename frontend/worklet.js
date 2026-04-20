/**
 * PCM16 encoder + resampler AudioWorkletProcessor.
 *
 * Browsers typically ignore the requested AudioContext sample rate and return
 * 44_100 or 48_000 Hz. This worklet resamples to 16_000 Hz with a simple
 * linear interpolator (good enough for ASR; the low-pass filter in the mic
 * preamp plus speech's narrow bandwidth make fancier SRC unnecessary) and
 * emits Int16 PCM so the server can treat every client identically.
 *
 * Messages posted to the main thread:
 *   {type: "pcm",   buffer: ArrayBuffer}   Int16 little-endian, 16 kHz mono
 *   {type: "level", peak: number}          0..1, for the level meter
 *   {type: "rate",  inputSampleRate: number, outputSampleRate: 16000}
 *                                           sent once at startup for display
 */
const TARGET_RATE = 16000;
const FLUSH_EVERY_MS = 128;               // send ~128 ms per frame (2048 samples @ 16 kHz)
const FLUSH_EVERY_OUT = Math.round(TARGET_RATE * FLUSH_EVERY_MS / 1000);
const LEVEL_EVERY_MS = 50;

class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._inputRate = sampleRate;           // provided by AudioWorkletGlobalScope
    this._ratio = this._inputRate / TARGET_RATE;
    this._resampleCursor = 0;               // fractional read-position in the input buffer
    this._outBuf = new Int16Array(FLUSH_EVERY_OUT);
    this._outIdx = 0;
    this._peak = 0;
    this._peakAccumIn = 0;
    this._peakEveryIn = Math.round(this._inputRate * LEVEL_EVERY_MS / 1000);
    this._announced = false;
  }

  process(inputs) {
    if (!this._announced) {
      this.port.postMessage({
        type: "rate",
        inputSampleRate: this._inputRate,
        outputSampleRate: TARGET_RATE,
      });
      this._announced = true;
    }

    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;

    // Track peak for the level meter
    for (let i = 0; i < ch.length; i++) {
      const a = Math.abs(ch[i]);
      if (a > this._peak) this._peak = a;
    }
    this._peakAccumIn += ch.length;
    if (this._peakAccumIn >= this._peakEveryIn) {
      this.port.postMessage({ type: "level", peak: this._peak });
      this._peak = 0;
      this._peakAccumIn = 0;
    }

    // Resample ch (at inputRate) into this._outBuf (at TARGET_RATE).
    // Linear interpolation over the fractional cursor.
    let cursor = this._resampleCursor;
    const ratio = this._ratio;
    const outLen = Math.max(0, Math.floor((ch.length - cursor) / ratio));
    for (let i = 0; i < outLen; i++) {
      const pos = cursor + i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = ch[idx];
      const s1 = ch[idx + 1] !== undefined ? ch[idx + 1] : s0;
      const sample = s0 + (s1 - s0) * frac;
      // Clip + convert to int16
      const clipped = sample < -1 ? -1 : sample > 1 ? 1 : sample;
      this._outBuf[this._outIdx++] = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
      if (this._outIdx === FLUSH_EVERY_OUT) {
        // Detach by transferring
        const copy = new Int16Array(this._outBuf);
        this.port.postMessage({ type: "pcm", buffer: copy.buffer }, [copy.buffer]);
        this._outIdx = 0;
      }
    }
    // Carry the fractional leftover so we don't lose samples between blocks
    this._resampleCursor = cursor + outLen * ratio - ch.length;

    return true;
  }
}

registerProcessor("pcm16-processor", PCM16Processor);
