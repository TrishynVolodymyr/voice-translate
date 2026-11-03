"""Compares ASR models on the user's recordings. For logs\\samples\\phrase_*.wav the exact text is known
(prompts.json), so WER is computed (lower case, punctuation removed, ' and ’ unified, є/е in «проєкт» kept as is).
Live phrases from logs\\out\\*_uk.wav have no reference and are printed side by side. Results go to
logs\\asr-bench-<stamp>.csv."""
import csv
import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import ASR, LOGS, VOICES, HybridASR, Timer  # noqa: E402

# Usage: bench_asr.py [--set NAME] [kinds...] (kind:cuda = on the GPU); --set NAME uses the <voices>-NAME folder (recorded with Record-Samples --set).
args = sys.argv[1:]
SET = None
if args[:1] == ["--set"]:
    SET, args = args[1], args[2:]
KINDS = args or ["parakeet", "nemo-uk"]


def words(s: str) -> list:
    s = s.lower().replace("’", "'").replace("ʼ", "'")
    return re.sub(r"[^\w' -]+", " ", s).split()


def wer(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


samples = VOICES.parent / f"{VOICES.name}-{SET}" if SET else VOICES
prompts = json.loads((VOICES / "prompts.json").read_text(encoding="utf-8"))
files = sorted(samples.glob("phrase_*.wav")) + ([] if SET else sorted((LOGS / "out").glob("*_uk.wav")))

def make(k: str):  # "parakeet:cuda" = the same model on the GPU (Parakeet then is the fp32 export)
    kind, _, prov = k.partition(":")
    return HybridASR(provider=prov or "cpu") if kind == "hybrid" else ASR(provider=prov or "cpu", kind=kind)


models = {k: make(k) for k in KINDS}
rows, errs, total, times = [], {k: 0 for k in KINDS}, 0, {k: 0.0 for k in KINDS}
for wav in files:
    audio, sr = sf.read(wav, dtype="float32")
    ref = prompts.get(wav.stem, "")
    row = {"file": wav.stem, "said": ref}
    for k, m in models.items():
        t = Timer()
        row[k] = m(audio, sr)
        times[k] += t.ms()
        if ref:
            e = wer(words(ref), words(row[k]))
            row[f"{k}_errors"] = e
            errs[k] += e
    total += len(words(ref))
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
out = LOGS / f"asr-bench-{time.strftime('%Y%m%d-%H%M%S')}.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k != "said", k))
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
for k in KINDS:
    print(f"{k}: WER {100 * errs[k] / total:.1f}% ({errs[k]} помилок на {total} слів), "
          f"середній час {times[k] / len(files):.0f} мс")
print("Таблиця:", out)
