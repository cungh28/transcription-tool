"""Phase 0 benchmark: run ASR candidates over a labelled audio set.

Dataset layout (bench/data/):
    <lang>_<name>.wav|mp3|m4a|flac   audio; <lang> is one of vi / en / mix
    <lang>_<name>.txt                reference transcript (same stem)

Usage:
    uv run run_bench.py --model small --compute int8
    uv run run_bench.py --model large-v3-turbo --device cuda --compute int8_float16
    uv run run_bench.py --model small --language vi     # force language
"""
import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import psutil
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"


def normalize(text: str) -> str:
    """Lowercase, NFC, strip punctuation; keep Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lang_of(path: Path) -> str:
    prefix = path.stem.split("_", 1)[0]
    return prefix if prefix in {"vi", "en", "mix"} else "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. tiny, small, medium, large-v3-turbo, or a CTranslate2 dir/HF repo (PhoWhisper conversions)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--compute", default="int8", help="int8, int8_float16, float16, ...")
    ap.add_argument("--language", default=None, help="force vi/en; default = auto-detect")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    files = sorted(p for p in DATA.iterdir() if p.suffix.lower() in AUDIO_EXT and p.with_suffix(".txt").exists())
    if not files:
        raise SystemExit(f"No labelled audio found in {DATA} (need audio + same-name .txt)")

    proc = psutil.Process()
    t0 = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute, cpu_threads=args.threads)
    load_s = time.perf_counter() - t0
    print(f"model={args.model} device={args.device} compute={args.compute} load={load_s:.1f}s")

    rows = []
    peak_rss = proc.memory_info().rss
    for f in files:
        audio = decode_audio(str(f), sampling_rate=16000)
        dur = len(audio) / 16000
        t = time.perf_counter()
        segs, info = model.transcribe(audio, language=args.language, beam_size=args.beam, vad_filter=True)
        hyp = " ".join(s.text.strip() for s in segs)
        wall = time.perf_counter() - t
        peak_rss = max(peak_rss, proc.memory_info().rss)
        ref = normalize(f.with_suffix(".txt").read_text(encoding="utf-8"))
        hyp_n = normalize(hyp)
        wer = jiwer.wer(ref, hyp_n) if ref else float("nan")
        cer = jiwer.cer(ref, hyp_n) if ref else float("nan")
        rows.append(dict(file=f.name, lang=lang_of(f), detected=info.language, dur_s=round(dur, 2),
                         wall_s=round(wall, 2), rtf=round(wall / dur, 3), wer=round(wer, 3), cer=round(cer, 3)))
        r = rows[-1]
        print(f"{r['file']:<28} {r['lang']:<5} det={r['detected']:<3} dur={r['dur_s']:>6}s rtf={r['rtf']:<6} WER={r['wer']:<6} CER={r['cer']}")

    print("\n== summary by language ==")
    for lang in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lang]
        print(f"{lang:<6} n={len(sub)} avg_WER={sum(r['wer'] for r in sub)/len(sub):.3f} "
              f"avg_CER={sum(r['cer'] for r in sub)/len(sub):.3f} avg_RTF={sum(r['rtf'] for r in sub)/len(sub):.3f}")
    print(f"peak RSS: {peak_rss / 1e6:.0f} MB  (RTF < 1 needed for real time; final-pass latency ~ RTF x segment length)")

    RESULTS.mkdir(exist_ok=True)
    tag = f"{args.model.replace('/', '_')}_{args.device}_{args.compute}" + (f"_{args.language}" if args.language else "")
    out = RESULTS / f"{tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
