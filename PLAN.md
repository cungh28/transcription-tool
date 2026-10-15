# Transcription Tool — Plan

## Goals
- Real-time speech-to-text, latency <= 3 s
- Vietnamese + English (including code-switching)
- Lightweight, cross-platform desktop app (Windows / macOS / Linux)
- Local STT models; cloud API (Claude) only for summary / Q&A on transcript text

## Architecture

```
Mic / system audio -> VAD (Silero) -+-> Streaming ASR -> live "draft" text (~0.5-1.5 s)
                                    +-> on pause: Whisper / PhoWhisper -> corrected final text (~1-3 s)
                                                   |
                              Transcript store (SQLite) -> Claude API (summary / Q&A, text only)
```

- Pass 1 (drafts): streaming model gives instant partial text.
- Pass 2 (finals): on VAD pause, a stronger model re-decodes the segment and replaces the draft.

## Tech stack

| Layer | Choice |
|---|---|
| App shell | Tauri 2 (Rust) + Svelte or React |
| Inference | sherpa-onnx (streaming) + whisper.cpp via `whisper-rs` |
| Mic capture | `cpal` |
| System audio | WASAPI loopback (Win), ScreenCaptureKit (macOS 13+), PulseAudio/PipeWire monitor (Linux) |
| VAD | Silero VAD (ONNX) |
| Storage | SQLite; API key in OS keychain |
| LLM | Claude API: Haiku 5.5 (live Q&A), Sonnet 5.5 (summaries) |

Fallback if faster delivery is needed: Python sidecar (FastAPI + faster-whisper / sherpa-onnx) behind the Tauri UI.

## Models (to be confirmed by Phase 0 benchmark)
- Streaming drafts: sherpa-onnx streaming Zipformer (EN); check model zoo for a Vietnamese streaming Zipformer, else short-chunk Whisper.
- Final pass: Whisper large-v3-turbo (quantized); benchmark PhoWhisper (VinAI) for Vietnamese.
- CPU-only fallback: Whisper small/medium int8.
- Avoid: Parakeet/Canary (no Vietnamese), Moonshine (English-focused).

## Phases

| Phase | Time | Deliverable |
|---|---|---|
| 0. Benchmark spike | 3-5 days | Test candidates on ~30 min of VI/EN/mixed audio: WER, latency, RAM on target hardware |
| 1. Engine core | 1-2 wks | CLI: mic -> VAD -> streaming draft -> Whisper final, with latency logging |
| 2. Desktop app | 1-2 wks | Tauri UI: live transcript, language switch, model download manager, export txt/srt/md |
| 3. System audio | 1 wk | Loopback capture per OS, device picker |
| 4. LLM features | 1 wk | Summary, action items, "ask about this meeting", rolling-window context, streaming responses |
| 5. Polish | 1-2 wks | Installers, auto-update, settings, crash handling, optional diarization |

Total: ~6-9 weeks for one developer.

## Risks
1. Vietnamese accuracy (tones/diacritics) on small models -> Phase 0 benchmark.
2. VI/EN code-switching -> per-segment language detection, short segments.
3. CPU-only machines can't run large-v3-turbo in real time -> model tiers, auto-select by hardware.
4. System-audio capture differs per OS -> most likely schedule slip.

## Dev hardware
Windows 11, Intel i7-1165G7 (4C/8T), 31.7 GB RAM, NVIDIA MX350 (2 GB VRAM). Treat as CPU-only: large-v3-turbo on CPU is unlikely to hit real time, so expect small/medium int8 as the default final-pass tier. Benchmark confirms.

## Open questions
- Is speaker diarization required?
