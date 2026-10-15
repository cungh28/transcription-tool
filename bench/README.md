# Phase 0 — ASR benchmark

Goal: pick the final-pass (and draft-pass) models by measuring WER/CER, real-time factor and RAM on **our hardware** and **our audio**.

## Setup
```bash
cd bench
uv sync          # pins Python 3.12 (deps lack wheels for 3.14)
```

## Data
Put ~30 min of labelled audio in `data/` (gitignored):
`vi_meeting1.wav` + `vi_meeting1.txt`, `en_talk1.mp3` + `en_talk1.txt`, `mix_standup1.wav` + `mix_standup1.txt`.
Prefix = language (`vi`, `en`, `mix` for code-switching). Aim for 10 clips each, 1-3 min, with real noise/accents.
Public sets to supplement: Common Voice vi/en, FLEURS vi/en, VIVOS (vi).

## Run
```bash
uv run run_bench.py --model small --compute int8
uv run run_bench.py --model medium --compute int8
uv run run_bench.py --model large-v3-turbo --compute int8            # CPU
uv run run_bench.py --model large-v3-turbo --device cuda --compute int8_float16  # MX350 has 2 GB VRAM: likely won't fit
```
PhoWhisper: convert the HF checkpoint with `ct2-transformers-converter` and pass the output dir to `--model`.

## Decision rule
A model is viable for the final pass if avg RTF <= ~0.5 on the target CPU (segments of ~10 s decode in <= 5 s... tune to the 3 s budget) and VI WER is acceptable. Record results in `results/` and summarize in PLAN.md.

## Still to add
- sherpa-onnx streaming backend (draft pass): needs a Vietnamese streaming Zipformer model — verify it exists in the sherpa-onnx model zoo first.
- Simulated-stream latency (feed audio in 100 ms chunks, measure time-to-first/final text).
