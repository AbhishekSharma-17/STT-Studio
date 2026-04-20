#!/usr/bin/env python
"""Minimal WER benchmark.

Given a directory of audio+transcript pairs, calls /transcribe for each model
and writes a CSV with per-sample WER + summary stats.

Pairs: ``<id>.wav`` + ``<id>.txt`` (reference transcript in the same dir).
Usage::

    uv run python scripts/bench_wer.py --samples data/samples --out results/
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx


def normalise(s: str) -> str:
    """Lowercase, collapse whitespace, strip most punctuation. Arabic-friendly."""
    s = s.strip().lower()
    # Keep Arabic letters (U+0600..U+06FF) and Latin alphanumerics + whitespace
    s = re.sub(r"[^\w\u0600-\u06FF\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate via classic edit distance over tokens."""
    ref = normalise(reference).split()
    hyp = normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # Levenshtein over word tokens
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n] / m


def collect_pairs(samples_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for wav in sorted(samples_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.is_file():
            pairs.append((wav, txt))
    return pairs


def transcribe(backend_url: str, wav: Path, model: str, language: str | None) -> dict:
    files = {"file": (wav.name, wav.read_bytes(), "audio/wav")}
    data = {"model": model}
    if language:
        data["language"] = language
    r = httpx.post(f"{backend_url}/transcribe", data=data, files=files, timeout=120.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="http://localhost:3000")
    p.add_argument("--samples", type=Path, default=Path("data/samples"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--language", default="ar")
    p.add_argument("--models", nargs="+", default=["qwen3-asr", "whisper"])
    args = p.parse_args()

    pairs = collect_pairs(args.samples)
    if not pairs:
        print(f"No samples found in {args.samples} (need <id>.wav + <id>.txt)", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = args.out / f"bench_{stamp}.csv"

    per_model: dict[str, list[float]] = {m: [] for m in args.models}

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "sample", "model", "language", "wer",
            "duration_ms", "reference", "hypothesis",
        ])

        for wav, txt in pairs:
            reference = txt.read_text(encoding="utf-8")
            for model in args.models:
                try:
                    res = transcribe(args.backend, wav, model, args.language)
                except httpx.HTTPError as exc:
                    print(f"! {wav.name} / {model}: {exc}", file=sys.stderr)
                    continue
                hyp = res.get("text", "")
                err = wer(reference, hyp)
                per_model[model].append(err)
                w.writerow([
                    wav.stem, model, res.get("language", ""),
                    f"{err:.4f}", res.get("duration_ms", ""),
                    reference, hyp,
                ])
                print(f"  {wav.stem} / {model:9s} WER={err:.3f}")

    print("\n== Summary ==")
    for m, xs in per_model.items():
        if xs:
            avg = sum(xs) / len(xs)
            print(f"  {m:9s} avg WER={avg:.3f} over {len(xs)} samples")
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
