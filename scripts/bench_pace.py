"""Voice pace limit (VT_TTS_PACE): the recorded test phrases are recognized and translated on the CPU, then spoken by the
voice with each given limit (0 = off). Per setting: generation speed vs real time, GPU busy % and power while generating
(nvidia-smi every 100 ms), time to the first chunk, and gaps a 1x player would hear.

    .venv\\Scripts\\python.exe scripts\\bench_pace.py 0 2 1.5 1.2 0
"""
import datetime as dt
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as p  # noqa: E402

paces = [float(x) for x in sys.argv[1:]] or [0.0, 1.5]
files = sorted(p.VOICES.glob("phrase_*.wav"))
if not files:
    sys.exit(f"No phrase_*.wav in {p.VOICES} — record them with Record-Samples.bat")
asr, mt = p.ASR(provider="cpu", kind="ultra"), p.MT("cpu")
texts = []
for f in files:
    audio, sr = sf.read(f, dtype="float32")
    texts.append(mt(asr(audio, sr)))
tts = p.TTS("1.7B", p.VOICES / f"{p.env('VT_REF', f'ref_{p.SRC}')}.wav", "", xvec_only=True)
tts.warmup()

samples = []  # (epoch s, GPU busy %, power W)


def sampler(proc):
    for line in proc.stdout:
        try:
            ts, util, power = [x.strip() for x in line.split(",")]
            samples.append((dt.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(util), float(power)))
        except ValueError:
            pass


smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,power.draw", "--format=csv,noheader,nounits",
                        "-lms", "100"], stdout=subprocess.PIPE, text=True)
threading.Thread(target=sampler, args=(smi,), daemon=True).start()
time.sleep(2)
try:
    for pace in paces:
        tts.pace = pace
        windows, first, audio_s, gen_s, gaps, n_gaps = [], [], 0.0, 0.0, 0.0, 0
        for text in texts:
            start = time.time()
            arrivals = [(time.time(), len(chunk) / sr) for chunk, sr in tts.stream(text)]
            windows.append((start, time.time()))
            first.append(arrivals[0][0] - start)
            audio_s += sum(d for _, d in arrivals)
            gen_s += windows[-1][1] - arrivals[0][0]
            play_t = arrivals[0][0]  # a 1x player starting with the first chunk: a late chunk is a gap
            for t_arr, d in arrivals:
                if t_arr > play_t + 0.005:
                    gaps += t_arr - play_t
                    n_gaps += 1
                    play_t = t_arr
                play_t += d
            time.sleep(1.5)
        time.sleep(0.5)
        inside = [s for s in samples if any(a <= s[0] <= b for a, b in windows)]
        print(f"pace {pace if pace else 'off':>4}: {audio_s:.0f} s of speech at {audio_s / gen_s:.2f}x real time | "
              f"GPU busy {statistics.mean(s[1] for s in inside):.0f}% at {statistics.mean(s[2] for s in inside):.0f} W "
              f"while generating | first chunk median {statistics.median(first) * 1000:.0f} ms | "
              f"gaps: {n_gaps} ({gaps:.2f} s)", flush=True)
finally:
    smi.kill()
