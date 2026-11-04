"""Text-only test of simultaneous translation (M2, step 1): speak without pauses and watch the English grow.

Sources: the microphone (default) or WAV files played at real-time speed (--wav ...; "--wav phrases" = the 20
recorded test phrases one after another with 0.3 s gaps), so file runs behave exactly like live speech.
Every emitted English chunk is logged with its time to logs\\stream-<stamp>\\events.csv; a summary of delays,
CPU load and the final texts is printed at the end (or after q + Enter / Ctrl+C)."""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import LOGS, SRC, TTS, VOICES, check_languages, mt_label, mt_where, print_placement, where, device_arg, env, output_players, resample  # noqa: E402
from stream_core import SR, HyMTPrefix, QuickMTPrefix, Segmenter, Speaker, StreamingASR, StreamingMT  # noqa: E402

CHUNK = 0.1  # seconds of audio per producer chunk


def wav_producer(paths, q, gap=0.3):
    t0 = time.perf_counter()
    sent = 0.0
    for p in paths:
        a, sr = sf.read(p, dtype="float32")
        a = resample(a if a.ndim == 1 else a.mean(1), sr, SR)
        a = np.concatenate([a, np.zeros(int(gap * SR), np.float32)])
        for i in range(0, len(a), int(CHUNK * SR)):
            q.put(a[i:i + int(CHUNK * SR)])
            sent += len(a[i:i + int(CHUNK * SR)]) / SR  # the last piece of a file is shorter than CHUNK
            time.sleep(max(0.0, t0 + sent - time.perf_counter()))
    q.put(np.zeros(int(2 * SR), np.float32))  # trailing silence so the last segment closes
    q.put(None)


def mic_producer(q, device, stop):
    import sounddevice as sd

    def cb(indata, frames, t, status):
        q.put(resample(indata[:, 0].copy(), 48000, SR))

    with sd.InputStream(samplerate=48000, channels=1, dtype="float32", callback=cb, device=device,
                        blocksize=int(48000 * CHUNK)):
        stop.wait()
    q.put(None)


def voice_summary(speaker, segments, inp, out):
    """voice.wav: left — the Ukrainian input, right — the English voice, both on the input's timeline. The voice is
    brought to the input's peak level: as generated it is ~3× quieter, and the user barely heard it next to their own."""
    log = speaker.log
    (out / "voice.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in log), encoding="utf-8")
    sr = speaker.sr
    src = resample(np.concatenate(inp), SR, sr) if inp else np.zeros(0, np.float32)
    end = max([len(src) / sr] + [s + len(c) / sr for s, c in speaker.audio])
    wav = np.zeros((int(end * sr) + 1, 2), np.float32)
    wav[:len(src), 0] = src
    for s, c in speaker.audio:
        i = max(0, int(s * sr))
        wav[i:i + len(c), 1] += c[:len(wav) - i]
    peaks = np.abs(wav).max(axis=0)
    if peaks[1] > 0:
        wav[:, 1] *= min(4.0, max(peaks[0], 0.3) / peaks[1])
    sf.write(out / "voice.wav", wav, sr, subtype="PCM_16")
    if not log:
        return
    gen = [e["first_audio"] - e["ready"] for e in log]
    wait = [e["play_start"] - e["first_audio"] for e in log]
    voiced = [i for i in range(len(segments)) if any(e["seg"] == i for e in log)]
    first_voice = [min(e["play_start"] for e in log if e["seg"] == i) - segments[i]["start"] for i in voiced]
    voice_end = [max(e["play_end"] for e in log if e["seg"] == i) - segments[i]["speech_end"] for i in voiced]
    print(f"Голос: {len(log)} шматків; від готового тексту до першого звуку шматка медіана {np.median(gen):.2f} с "
          f"(макс {max(gen):.2f}); очікування в черзі медіана {np.median(wait):.2f} с (макс {max(wait):.2f})")
    if voiced:
        print(f"Перший звук перекладу після початку висловлювання: медіана {np.median(first_voice):.2f} с "
              f"(макс {max(first_voice):.2f}); голос замовкає після кінця мовлення: медіана {np.median(voice_end):.2f} с "
              f"(макс {max(voice_end):.2f})")
    print(f"Запис (ліворуч ви, праворуч голос): {out / 'voice.wav'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngl", default="999", help="шарів перекладача на GPU (обробляє Start-скрипт)")
    ap.add_argument("--mt", default="quickmt", choices=["quickmt", "hymt7", "hymt1.8", "gguf"],
                    help="перекладач; hymt* і gguf — через llama-server, який запускає Start-скрипт")
    ap.add_argument("--wav", nargs="*", help="файли або 'phrases' (20 тестових фраз поспіль)")
    # 0.5, not 0.25: two passes only 0.25 s apart agree on wrong words more easily — on the user's live test the
    # translation diverged from the whole-utterance one by 26% instead of 20%, for a 0.25 s faster start (2026-11-02).
    ap.add_argument("--step", type=float, default=0.5, help="як часто перераховувати (с)")
    ap.add_argument("--silence", type=float, default=0.6, help="пауза, що закриває висловлювання (с)")
    ap.add_argument("--max-seg", type=float, default=25.0, help="примусово закрити висловлювання довше за це (с)")
    ap.add_argument("--asr", default="auto", choices=["auto", "parakeet", "ultra", "hybrid"],
                    help="auto — Parakeet Ultra на процесорі (точніша за v3: 7,7–8,3%% проти 9,4–12,2%% на голосі користувача), Parakeet v3 fp32 на відеокарті; parakeet / ultra — примусово; hybrid — Parakeet + запасна NeMo-uk")
    ap.add_argument("--asr-device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto — на відеокарті з quickmt (перекладач на процесорі), на процесорі з Hy-MT2/Gemma (їм потрібна відеопам'ять); cuda — Parakeet fp32 і quickmt на GPU, +4,5 ГБ VRAM, не гальмує від зайнятого процесора")
    ap.add_argument("--hold", type=int, default=2, help="скільки останніх узгоджених слів притримувати")
    ap.add_argument("--mic", default=env("VT_MIC"))
    ap.add_argument("--voice", action="store_true", help="озвучувати переклад по ходу (Qwen3-TTS, ваш тембр)")
    ap.add_argument("--tts", default="1.7B", choices=["0.6B", "1.7B"])
    ap.add_argument("--ref", default=env("VT_REF", "ref_uk"), help="зразок голосу: ім'я в теці голосів (VT_VOICES_DIR) або шлях до WAV")
    ap.add_argument("--out", default=env("VT_OUT"), help="пристрій виводу голосу (номер або частина назви; за замовчуванням — системний)")
    ap.add_argument("--cable", action="store_true", help="голос також у віртуальний мікрофон VB-CABLE (Discord, ігри)")
    ap.add_argument("--first-words", type=int, default=1, help="перший шматок фрази озвучувати, щойно стільки слів")
    ap.add_argument("--no-voice-fx", action="store_true", help="без прискорення й стискання пауз (для порівняння на слух)")
    ap.add_argument("--skip-key", default=env("VT_PUSH_KEY", "ctrl_r"), help="клавіша «замовкни»: голос скидає все, що ще не сказав "
                                                       "(назви pynput: ctrl_r, f8, pause…)")
    ap.add_argument("--mute", action="store_true", help="не відтворювати голос, лише записати voice.wav (для тестів)")
    a, _ = ap.parse_known_args()  # the launcher may pass options meant for voice_translate.py (--ngl …)

    out = LOGS / f"stream-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    check_languages(a.mt, a.asr)
    dev = a.asr_device if a.asr_device != "auto" else ("cuda" if a.mt == "quickmt" else "cpu")
    kind = a.asr if a.asr != "auto" else ("ultra" if dev == "cpu" else "parakeet")
    print_placement([
        ("паузи", "Silero VAD", "процесор"),
        ("розпізнавання", {"ultra": "Parakeet Ultra", "hybrid": "Parakeet v3 + NeMo-uk"}.get(kind, "Parakeet v3"), where(dev)),
        ("переклад", mt_label(a.mt), mt_where(a.mt, dev, a.ngl)),
    ] + ([("голос", f"Qwen3-TTS {a.tts}", "відеокарта")] if a.voice else []))
    print("Завантаження моделей…", flush=True)
    asr = StreamingASR(hybrid=(kind == "hybrid"), provider=dev, kind="ultra" if kind == "ultra" else "parakeet")
    mt = StreamingMT(QuickMTPrefix(dev) if a.mt == "quickmt" else HyMTPrefix(), context_segments=2 if a.mt == "hymt7" else 0, hold=a.hold)
    mt.translate("Привіт", "", None)
    vad = Segmenter(a.silence)
    speaker = None
    if a.voice:  # after the ASR: once torch has started CUDA, sherpa-onnx can no longer load its CUDA provider
        ref = Path(a.ref) if Path(a.ref).suffix else VOICES / f"{a.ref}.wav"
        pf = VOICES / "prompts.json"
        prompts = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
        tts = TTS(a.tts, ref, prompts.get(ref.stem, ""), xvec_only=True)
        tts.warmup()
        speaker = Speaker(tts, None if a.mute else output_players(a.out, a.cable), first_words=a.first_words, fx=not a.no_voice_fx)

    q: "queue.Queue" = queue.Queue()
    stop = threading.Event()
    if a.wav:
        paths = sorted(VOICES.glob("phrase_*.wav")) if a.wav == ["phrases"] else [Path(p) for p in a.wav]
        threading.Thread(target=wav_producer, args=(paths, q), daemon=True).start()
        print(f"Відтворюю {len(paths)} файл(ів) у реальному часі…\n", flush=True)
    else:
        mic = device_arg(a.mic, "input")
        threading.Thread(target=mic_producer, args=(q, mic, stop), daemon=True).start()

        def wait_quit():
            for line in sys.stdin:
                if line.strip().lower() in ("q", "й", "exit", "quit", "вихід"):
                    stop.set()
                    return
        threading.Thread(target=wait_quit, daemon=True).start()
        print("Говоріть без пауз — переклад ростиме по ходу. Вихід: q + Enter.", flush=True)
        if speaker and not a.mute:
            print("Голос увімкнено — слухайте в навушниках, інакше мікрофон почує голос перекладу.", flush=True)
            from pynput import keyboard
            k = a.skip_key
            skip_key = getattr(keyboard.Key, k) if hasattr(keyboard.Key, k) else keyboard.KeyCode.from_char(k)

            def on_press(key):
                if key == skip_key:  # the utterance being spoken now, or the last one if the user is silent
                    speaker.skip(len(segments) if in_seg else len(segments) - 1)
                    print("\n   ⏹ голос зупинено\n", flush=True)
            keyboard.Listener(on_press=on_press, daemon=True).start()
            print(f"Перебили? [{k}] — голос замовкає й скидає все, що ще не сказав.", flush=True)
        print(flush=True)

    events, segments, steps = [], [], []
    seg, seg_start, in_seg, last_step, clock = np.zeros(0, np.float32), 0.0, False, 0.0, 0.0
    preroll = np.zeros(0, np.float32)
    t_wall0, cpu0 = time.perf_counter(), time.process_time()
    busy = 0.0
    clock0 = 0.0  # wall time of the first audio sample: opening the microphone takes ~0.8 s, so wall != clock
    inp = []
    uk_cut: list = []  # Ukrainian words of this utterance already translated and cut from the ASR window
    seg_times: dict = {}  # when, from the utterance start, its first word was confirmed / its English appeared
    seg_step0 = 0

    def now():
        return time.perf_counter() - t_wall0 - clock0

    def emit(words, final=False):
        if words and "en" not in seg_times:
            seg_times["en"] = now() - seg_start
        if speaker:
            speaker.add(words, len(segments), final)
        if words:
            events.append({"t": round(now(), 2), "seg": len(segments), "words": " ".join(words), "final": final})
            print(" ".join(words), end=" ", flush=True)

    def close_segment(forced=False):
        nonlocal seg, in_seg, uk_cut
        t = time.perf_counter()
        if not a.wav:  # live speech is kept so the same test can be replayed later with --wav
            sf.write(out / f"seg{len(segments):02d}.wav", seg, SR, subtype="PCM_16")
        src = asr.step(seg, final=True)
        emit(mt.step(src, final=True), final=True)
        done = now()
        uk = " ".join(uk_cut + src.split())
        segments.append({"start": round(seg_start, 2), "speech_end": round(clock, 2), "done": round(done, 2),
                         "uk": uk, "en": " ".join(mt.spoken), "forced": forced, "asr_passes": asr.passes,
                         "uk_model": asr.use_uk})
        print(f"\n   {SRC.upper()}: {uk}", flush=True)
        print_timing(done)
        uk_cut = []
        asr.reset()
        mt.reset()
        seg, in_seg = np.zeros(0, np.float32), False
        return time.perf_counter() - t

    def print_timing(done):
        """Where the time of this utterance went, counted from its start (the voice may still be generating)."""
        f = lambda x: f"{x:.1f} с".replace(".", ",") if x is not None else "—"  # noqa: E731
        st = steps[seg_step0:]
        voice = [e for e in (speaker.log if speaker else []) if e["seg"] == len(segments) - 1]
        parts = [f"слово підтверджено {f(seg_times.get('stable'))}", f"переклад {f(seg_times.get('en'))}"]
        if speaker:
            parts.append(f"голос почав {f(voice[0]['play_start'] - seg_start)}" if voice else "голос почне після паузи")
        parts.append(f"кінець перекладу через {f(done - clock)} після паузи")
        if st:
            parts.append(f"на крок: розпізнавання ~{int(np.median([x['asr_ms'] for x in st]))} мс, "
                         f"переклад ~{int(np.median([x['mt_ms'] for x in st]))} мс")
        print("   ⏱ " + " · ".join(parts) + "\n", flush=True)

    try:
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                if speaker:
                    speaker.tick()
                continue
            if chunk is None:
                break
            if not inp:
                clock0 = time.perf_counter() - t_wall0 - len(chunk) / SR
                if speaker:
                    speaker.t0 = t_wall0 + clock0
            chunks = [chunk]
            while not q.empty():
                c = q.get_nowait()
                if c is None:
                    q.put(None)
                    break
                chunks.append(c)
            audio = np.concatenate(chunks)
            inp.append(audio)
            clock += len(audio) / SR
            speaking = vad.feed(audio)
            if speaking and not in_seg:
                in_seg, seg, seg_start, last_step = True, preroll.copy(), clock - len(preroll) / SR, clock
                seg_times, seg_step0 = {}, len(steps)
            if in_seg:
                seg = np.concatenate([seg, audio])
                if not speaking:
                    busy += close_segment()
                elif (len(seg) - asr.offset) / SR > a.max_seg:
                    busy += close_segment(forced=True)
                elif clock - last_step >= a.step:
                    t = time.perf_counter()
                    behind = now() - clock  # how far processing runs behind the audio
                    window_s = (len(seg) - asr.offset) / SR
                    src = asr.step(seg)
                    if src and "stable" not in seg_times:
                        seg_times["stable"] = clock - seg_start
                    t_asr = time.perf_counter() - t
                    new = mt.step(src)
                    t_mt = time.perf_counter() - t - t_asr
                    emit(new)
                    closed_src = src.split()[:mt.closed_now]
                    cut = bool(mt.closed_now) and asr.cut(mt.closed_now)  # translated sentences leave the ASR window
                    if cut:
                        uk_cut += closed_src
                        mt.after_cut(mt.closed_now)
                    steps.append({"clock": round(clock, 2), "behind_s": round(behind, 2), "window_s": round(window_s, 1),
                                  "asr_ms": round(t_asr * 1000), "mt_ms": round(t_mt * 1000),
                                  "raw": " ".join(asr.prev), "stable": src,
                                  "done_words": mt.done_words, "closed": mt.closed_now, "cut": cut,
                                  "offset_s": round(asr.offset / SR, 2), "new": " ".join(new)})
                    busy += time.perf_counter() - t
                    last_step = clock
            preroll = np.concatenate([preroll, audio])[-int(0.3 * SR):]
            if speaker:
                speaker.tick()
    except KeyboardInterrupt:
        pass

    wall = time.perf_counter() - t_wall0
    if speaker:
        if speaker.pending:
            speaker.add([], len(segments), final=True)
        speaker.drain()
    cpu = (time.process_time() - cpu0) / wall / os.cpu_count() * 100
    with (out / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["t", "seg", "words", "final"])
        w.writeheader()
        w.writerows(events)
    (out / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "steps.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in steps), encoding="utf-8")

    first = [next((e["t"] for e in events if e["seg"] == i), None) for i in range(len(segments))]
    lag_first = [f - s["start"] for f, s in zip(first, segments) if f is not None]
    lag_end = [s["done"] - s["speech_end"] for s in segments]
    partial = sum(1 for e in events if not e["final"])
    print(f"\n=== {len(segments)} висловлювань, {wall:.0f} с; mt={a.mt}, asr={dev}, step={a.step}, silence={a.silence}")
    if lag_first:
        print(f"Перший текст перекладу після початку висловлювання: медіана {np.median(lag_first):.2f} с "
              f"(мін {min(lag_first):.2f}, макс {max(lag_first):.2f})")
    if lag_end:
        print(f"Від кінця мовлення до кінця перекладу: медіана {a.silence + np.median(lag_end):.2f} с (з них {a.silence} с — очікування паузи)")
    print(f"Шматків перекладу до кінця висловлювань: {partial} з {len(events)}")
    print(f"Процесор (цей процес): ~{cpu:.0f}% від усіх ядер; робота моделей {busy:.1f} с з {wall:.0f} с")
    if speaker:
        voice_summary(speaker, segments, inp, out)
    print("Журнал:", out)


if __name__ == "__main__":
    main()
