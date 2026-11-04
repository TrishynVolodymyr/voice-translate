"""Push-to-talk voice bridge (milestone M0); Ukrainian -> English by default, see VT_SOURCE_LANG / VT_TARGET_LANG.

Hold the key and speak; on release the phrase goes through Parakeet (ASR) -> the translator (quickmt / Hy-MT2 / GGUF) ->
Qwen3-TTS (your cloned voice) and is played on the output device while it is still being generated. Every phrase is
also saved as WAV in logs\\out\\ and its timings appended to logs\\out\\timings.csv. Ctrl+C stops."""
import argparse
import csv
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import (ASR, LOGS, SRC, SRC_NAME, TGT, TGT_NAME, TRANSLATORS, TTS, VOICES, HybridASR, Timer,  # noqa: E402
                      check_languages, device_arg, env, mt_label, mt_where, output_players, print_placement,
                      resample, where)

MIC_SR = 48000


def parse_key(name: str):
    return getattr(keyboard.Key, name) if hasattr(keyboard.Key, name) else keyboard.KeyCode.from_char(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=env("VT_PUSH_KEY", "ctrl_r"), help="клавіша «говорю» (pynput: ctrl_r, alt_r, f9, x …)")
    ap.add_argument("--tts", default="1.7B", choices=["0.6B", "1.7B"], help="1.7B звучить людяніше; 0.6B — на 2 ГБ відеопам'яті менше")
    ap.add_argument("--ref", default=env("VT_REF", f"ref_{SRC}"), help="зразок голосу: ім'я в теці голосів (VT_VOICES_DIR) або шлях до WAV")
    ap.add_argument("--mode", default="xvec", choices=["xvec", "icl"],
                    help="xvec — лише тембр, вимова від моделі (без акценту); icl — копіює й манеру/акцент зразка")
    ap.add_argument("--mic", default=env("VT_MIC"), help="мікрофон (номер/назва, див. Record-Samples.bat --list)")
    ap.add_argument("--out", default=env("VT_OUT"), help="пристрій виводу (номер або частина назви; за замовчуванням — системний)")
    ap.add_argument("--cable", action="store_true", help="голос також у віртуальний мікрофон VB-CABLE (Discord, ігри)")
    ap.add_argument("--tts-pace", type=float, default=float(env("VT_TTS_PACE", "0") or 0),
                    help="генерувати голос не швидше за стільки × реального часу (напр. 1.5) — відеокарта відпочиває "
                         "між шматками; 0 — без обмеження (~2,5× для 1.7B)")
    ap.add_argument("--chunk", type=int, default=4, help="кроків TTS на шматок (менше — швидший перший звук)")
    # Parakeet only by default (user's choice 2026-11-02): in the live stream test the hybrid handed 19 of 46 phrases
    # to NeMo because English words (QuickMT, "in this spirit") look like a slip, and NeMo's text was much worse.
    ap.add_argument("--asr", default="auto", choices=["auto", "parakeet", "ultra", "hybrid"],
                    help="auto — Parakeet Ultra на процесорі (точніша за v3: 7,7–8,3%% проти 9,4–12,2%% на голосі користувача), Parakeet v3 fp32 на відеокарті; parakeet / ultra — примусово; hybrid — Parakeet + запасна NeMo-uk")
    ap.add_argument("--mt", default="hymt7", choices=list(TRANSLATORS),
                    help="hymt7 — Hy-MT2-7B (найточніший); hymt1.8 — Hy-MT2-1.8B; "
                         "quickmt — найшвидший, буквальний; gguf — свій GGUF з VT_MT (сервер запускає Start-скрипт)")
    ap.add_argument("--asr-device", default=env("VT_ASR_DEVICE", "auto"), choices=["auto", "cpu", "cuda"],
                    help="auto — процесор (Parakeet Ultra точніший, відеокарта економила лише ~0,15 с на фразу); "
                         "cuda — Parakeet fp32 і quickmt на GPU, +3,5 ГБ VRAM, не гальмує від зайнятого процесора")
    ap.add_argument("--ngl", default="999", help="шарів перекладача на GPU (обробляє Start-скрипт)")
    a = ap.parse_args()

    ref = Path(a.ref) if Path(a.ref).suffix else VOICES / f"{a.ref}.wav"
    prompts = json.loads((VOICES / "prompts.json").read_text(encoding="utf-8")) if (VOICES / "prompts.json").exists() else {}
    if not ref.exists():
        sys.exit(f"Немає зразка голосу {ref}. Запишіть свій голос (Record-Samples.bat) або створіть голос за описом "
                 "(Design-Voice.bat), див. README.")
    outdir = LOGS / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    check_languages(a.mt, a.asr)
    dev = a.asr_device if a.asr_device != "auto" else "cpu"
    # ASR before TTS: once torch has started CUDA, sherpa-onnx fails to load its CUDA provider (DLL error 1114) and
    # silently falls back to the CPU (2026-11-02).
    kind = a.asr if a.asr != "auto" else ("ultra" if dev == "cpu" else "parakeet")
    print_placement([
        ("розпізнавання", {"ultra": "Parakeet Ultra", "hybrid": "Parakeet v3 + NeMo-uk"}.get(kind, "Parakeet v3"), where(dev)),
        ("переклад", mt_label(a.mt), mt_where(a.mt, dev, a.ngl)),
        ("голос", f"Qwen3-TTS {a.tts}", "відеокарта" + (f" (не швидше за {a.tts_pace:g}× реального часу)" if a.tts_pace else "")),
    ])
    print("Завантаження моделей…", flush=True)
    asr = HybridASR(provider=dev) if kind == "hybrid" else ASR(provider=dev, kind=kind)
    mt = TRANSLATORS[a.mt][1](dev) if a.mt == "quickmt" else TRANSLATORS[a.mt][1]()
    tts = TTS(a.tts, ref, prompts.get(ref.stem, ""), xvec_only=(a.mode == "xvec"), pace=a.tts_pace)
    tts.warmup()
    mt("Привіт")
    player = output_players(a.out, a.cable)

    mic_q: "queue.Queue[np.ndarray]" = queue.Queue()
    recording = threading.Event()
    jobs: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue()

    def mic_cb(indata, frames, t, status):
        if recording.is_set():
            mic_q.put(indata[:, 0].copy())

    key = parse_key(a.key)

    def refresh_output():
        try:
            player.refresh()
        except Exception as e:  # noqa: BLE001 — a failed reopen must not stop the translator
            print(f"  (вихід звуку не перевідкрився: {e})", flush=True)

    def on_press(k):
        if k == key and not recording.is_set():
            while not mic_q.empty():
                mic_q.get_nowait()
            recording.set()
            print("● говоріть…", flush=True)
            # The Sound Mapper output follows a changed Windows default by itself; reopening it while the user speaks
            # also catches a device that appeared later (Bluetooth). ~0.26 s, so not in the keyboard hook's thread.
            threading.Thread(target=refresh_output, daemon=True).start()

    def on_release(k):
        if k == key and recording.is_set():
            recording.clear()
            released = time.perf_counter()
            parts = []
            while not mic_q.empty():
                parts.append(mic_q.get_nowait())
            if parts:
                jobs.put((np.concatenate(parts), released))

    def worker():
        log = outdir / "timings.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "speech_s", "asr_ms", "mt_ms", "first_audio_after_release_ms", "tts_total_ms",
                            "audio_s", "uk", "en", "tts", "ref", "mode", "mt"])
                f.flush()  # otherwise a run killed before its first phrase leaves an empty file without the header
            while True:
                audio, released = jobs.get()
                speech_s = len(audio) / MIC_SR
                if speech_s < 0.3:
                    continue
                t = Timer()
                uk = asr(audio, MIC_SR)
                asr_ms = t.ms()
                if not uk:
                    print("  (нічого не розпізнано)", flush=True)
                    continue
                t = Timer()
                en = mt(uk)
                mt_ms = t.ms()
                print(f"  {SRC.upper()}: {uk}\n  {TGT.upper()}: {en}", flush=True)
                if not en:
                    continue
                t, first, chunks, sr = Timer(), None, [], 24000
                for chunk, sr in tts.stream(en, a.chunk):
                    if first is None:
                        first = (time.perf_counter() - released) * 1000
                    player.put(chunk)
                    chunks.append(chunk)
                out = np.concatenate(chunks)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                sf.write(outdir / f"{stamp}.wav", out, sr)
                # The Ukrainian original too (16 kHz): needed to compare ASR models on real phrases later.
                sf.write(outdir / f"{stamp}_uk.wav", resample(audio, MIC_SR, 16000), 16000, subtype="PCM_16")
                w.writerow([stamp, round(speech_s, 2), round(asr_ms), round(mt_ms), round(first), round(t.ms()),
                            round(len(out) / sr, 2), uk, en, a.tts, ref.stem, a.mode, a.mt])
                f.flush()
                print(f"  ⏱ розпізнавання {asr_ms:.0f} мс · переклад {mt_ms:.0f} мс · "
                      f"перший звук через {first:.0f} мс після відпускання", flush=True)

    stop = threading.Event()

    def wait_for_quit():
        # Typing q + Enter works in any terminal, including ones that swallow Ctrl+C.
        for line in sys.stdin:
            if line.strip().lower() in ("q", "й", "exit", "quit", "вихід"):
                stop.set()
                return

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=wait_for_quit, daemon=True).start()
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", callback=mic_cb, device=device_arg(a.mic, "input")):
        print(f"\nГотово. Тримайте [{a.key}] і говоріть ({SRC_NAME}); відпустіть — почуєте переклад ({TGT_NAME}).\n"
              f"Голос: Qwen3-TTS {a.tts}, зразок {ref.name}, режим {a.mode}; переклад: {mt_label(a.mt)}.\n"
              f"Вихід: q + Enter (або Ctrl+C).\n", flush=True)
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            # Polling instead of listener.join(): a blocking join never lets Ctrl+C through on Windows.
            while listener.is_alive() and not stop.is_set():
                time.sleep(0.2)
    print("Зупинено.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
