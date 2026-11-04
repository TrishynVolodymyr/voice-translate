"""Push-to-talk latency on the recorded test phrases (voices folder, phrase_*.wav from Record-Samples.bat), fed as if the
key had just been released: recognition -> quickmt -> first sound of the voice.

    .venv\\Scripts\\python.exe scripts\\bench_ptt.py cpu            recognition and quickmt on the CPU (Parakeet Ultra)
    .venv\\Scripts\\python.exe scripts\\bench_ptt.py cuda           ... on the GPU (Parakeet v3 fp32)
    .venv\\Scripts\\python.exe scripts\\bench_ptt.py cpu --loaded   with 12 busy-loop processes on every CPU thread

Prints medians and maxima; the recognized and translated texts go to logs\\bench-ptt-<device>-<idle|loaded>.json.
The voice sample is VT_REF (default ref_<VT_SOURCE_LANG>)."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as p  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("device", choices=["cpu", "cuda"])
ap.add_argument("--loaded", action="store_true", help="load every CPU thread with a busy loop")
ap.add_argument("--tts", default="1.7B", choices=["0.6B", "1.7B"])
a = ap.parse_args()

files = sorted(p.VOICES.glob("phrase_*.wav"))
if not files:
    sys.exit(f"No phrase_*.wav in {p.VOICES} — record them with Record-Samples.bat")
asr = p.ASR(provider=a.device, kind="ultra" if a.device == "cpu" else "parakeet")  # before torch starts CUDA
mt = p.MT(a.device)
tts = p.TTS(a.tts, p.VOICES / f"{p.env('VT_REF', f'ref_{p.SRC}')}.wav", "", xvec_only=True)
tts.warmup()
audio0, sr0 = sf.read(files[0], dtype="float32")
asr(audio0, sr0), mt("OK")

busy = []
if a.loaded:
    busy = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(os.cpu_count() or 12)]
    time.sleep(2)
rows, texts = [], []
try:
    for f in files:
        audio, sr = sf.read(f, dtype="float32")
        t0 = time.perf_counter()
        src = asr(audio, sr)
        t1 = time.perf_counter()
        dst = mt(src)
        t2 = time.perf_counter()
        first = None
        for _chunk, _sr in tts.stream(dst):
            first = first or time.perf_counter()
        rows.append((len(audio) / sr, (t1 - t0) * 1000, (t2 - t1) * 1000, (first - t0) * 1000))
        texts.append({"file": f.stem, "src": src, "dst": dst})
finally:
    for b in busy:
        b.kill()

label = f"{a.device}-{'loaded' if a.loaded else 'idle'}"
p.LOGS.mkdir(parents=True, exist_ok=True)
(p.LOGS / f"bench-ptt-{label}.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8")
med = lambda i: statistics.median(r[i] for r in rows)  # noqa: E731
top = lambda i: max(r[i] for r in rows)  # noqa: E731
print(f"{label}: {len(rows)} phrases {min(r[0] for r in rows):.1f}-{max(r[0] for r in rows):.1f} s | "
      f"recognition median {med(1):.0f} ms (max {top(1):.0f}) | quickmt median {med(2):.0f} ms (max {top(2):.0f}) | "
      f"key release -> first sound median {med(3):.0f} ms (max {top(3):.0f})")
