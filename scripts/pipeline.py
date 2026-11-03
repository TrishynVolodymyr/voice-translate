"""Building blocks of the Ukrainian -> English voice bridge: ASR (Parakeet-TDT-0.6B-v3 via sherpa-onnx),
MT (quickmt-uk-en via CTranslate2) and TTS (Qwen3-TTS via faster-qwen3-tts). Shared by the benchmarks and the
push-to-talk tool."""
import os
import queue
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Settings from .env next to the launchers (see .env.example); variables already set in the environment win."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()


def env(name: str, default=None):
    return os.environ.get(name) or default


def _folder(var: str, default: str) -> Path:
    p = Path(env(var, default))
    return p if p.is_absolute() else ROOT / p


MODELS = _folder("VT_MODELS_DIR", "models")
VOICES = _folder("VT_VOICES_DIR", "voices")  # voice samples and prompts.json — personal, never in git
LOGS = _folder("VT_LOGS_DIR", "logs")
LLAMA_URL = f"http://127.0.0.1:{env('VT_LLAMA_PORT', '8093')}"
ASR_SR = 16000


def resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    # scipy, not torchaudio: torch started its 6 OpenMP threads for every 0.1 s microphone chunk, and their busy
    # waiting alone took ~36% of all cores while the user was silent (2026-11-02).
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr, target)
    return resample_poly(audio.astype(np.float32), target // g, sr // g).astype(np.float32)


class ASR:
    """kind="parakeet": NVIDIA Parakeet-TDT-0.6B-v3 int8 (25 languages, may slip into Russian);
    kind="nemo-uk": NVIDIA stt_ua_fastconformer_hybrid_large_pc as sherpa-onnx CTC (Ukrainian only).
    kind="ultra": Moondream Parakeet Ultra (post-trained v3, Ukrainian FLEURS 4.75% vs 5.94% on its card), community
    int8 export chrmongeau/sherpa-onnx-nemo-parakeet-ultra-int8 — int8 only, also with provider="cuda".
    provider="cuda": runs on the GPU so other programs loading the CPU don't slow it down. Parakeet then uses the
    fp32 export — the int8 one is not faster on CUDA (1026 vs 926 ms on CPU, 20 s of audio, 2026-11-02)."""

    def __init__(self, threads: int = 6, provider: str = "cpu", kind: str = "parakeet"):
        if provider == "cuda":
            import torch  # the CUDA build of sherpa-onnx finds cuDNN/cuBLAS in torch's bundled CUDA libraries
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        import sherpa_onnx
        if kind == "nemo-uk":
            d = MODELS / "nemo-fastconformer-ctc-uk"
            self.rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(d / "model.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads, provider=provider)
            return
        q = "" if provider == "cuda" and kind != "ultra" else ".int8"
        d = MODELS / ("parakeet-ultra-int8" if kind == "ultra" else
                      "parakeet-tdt-0.6b-v3-fp32" if provider == "cuda" else "parakeet-tdt-0.6b-v3-int8")
        self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / f"encoder{q}.onnx"), decoder=str(d / f"decoder{q}.onnx"),
            joiner=str(d / f"joiner{q}.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=threads, provider=provider, model_type="nemo_transducer", decoding_method="greedy_search")

    def __call__(self, audio: np.ndarray, sr: int) -> str:
        s = self.rec.create_stream()
        audio = resample(audio, sr, ASR_SR)
        if len(audio) < ASR_SR:  # under ~0.8 s Parakeet on CUDA takes ~220 ms instead of ~80; trailing silence is free
            audio = np.concatenate([audio, np.zeros(ASR_SR - len(audio), np.float32)])
        s.accept_waveform(ASR_SR, audio)
        self.rec.decode_stream(s)
        self.last = s.result  # tokens + timestamps, used by streaming to cut already translated audio
        # Parakeet writes <unk> for a mark outside its vocabulary (a dash: «набоїв<unk> хтось може поділитися?»);
        # quickmt then copied it and the voice would read it out.
        # It also sometimes starts a phrase with «♪».
        return re.sub(r"\s*<unk>", ",", s.result.text).replace("♪", "").strip(" ,")


RUSSIAN_ONLY = re.compile(r"[ыэъёЫЭЪЁ]")


def slipped(text: str) -> bool:
    """Parakeet left Ukrainian: Russian-only letters, or mostly Latin letters (it once produced Polish:
    «Obereżne za tym budynkiem się dwoje» for «Обережно, за тим будинком ще двоє»)."""
    latin = len(re.findall(r"[A-Za-zÀ-ž]", text))
    cyr = len(re.findall(r"[А-Яа-яІіЇїЄєҐґ]", text))
    return bool(RUSSIAN_ONLY.search(text)) or latin > cyr


class HybridASR:
    """Parakeet first; if its text contains letters that exist only in Russian (ы э ъ ё) — it slipped into
    Russian, as with «Хто йде зі мною…» → «Кто, да, земною…» — the phrase is re-recognised by the
    Ukrainian-only NeMo model, which cannot output Russian (+~0.1 s, only for such phrases)."""

    def __init__(self, threads: int = 6, provider: str = "cpu"):
        self.main, self.uk = ASR(threads, provider, "parakeet"), ASR(threads, provider, "nemo-uk")
        self.fallbacks = 0

    def __call__(self, audio: np.ndarray, sr: int) -> str:
        text = self.main(audio, sr)
        if slipped(text):
            self.fallbacks += 1
            return self.uk(audio, sr)
        return text


class MT:
    # CPU by default so it takes no VRAM: with the end-of-source token it is 40–80 ms per phrase on a 6-core Ryzen vs
    # ~30 ms on the GPU (2026-11-02). The earlier "1–2 s on CPU" was the missing </s> making it decode to max length.
    # 2 threads, not 6: when other programs load every core, 6 OpenMP threads wait for each other and one phrase took
    # up to 86 s; 2 threads stay at ~110 ms under the same load (2026-11-02).
    def __init__(self, device: str = "cpu", threads: int = 2, beam: int = 5):
        import ctranslate2
        import sentencepiece as spm
        if device == "cuda":
            import torch  # CTranslate2 finds cuBLAS in torch's bundled CUDA libraries
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        d = MODELS / "quickmt-uk-en"
        self.tr = ctranslate2.Translator(str(d), device=device, compute_type="int8" if device == "cpu" else "int8_float16",
                                         intra_threads=threads)
        self.src = spm.SentencePieceProcessor(model_file=str(d / "src.spm.model"))
        self.tgt = spm.SentencePieceProcessor(model_file=str(d / "tgt.spm.model"))
        self.beam = beam

    def __call__(self, text: str) -> str:
        # Sentence by sentence: given two sentences at once the model sometimes repeated one of them.
        sents = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
        if not sents:
            return ""
        # quickmt models expect an explicit end-of-source token; without it short inputs loop ("Thank you. Thank you…").
        res = self.tr.translate_batch([self.src.encode(s, out_type=str) + ["</s>"] for s in sents],
                                      beam_size=self.beam, max_decoding_length=256)
        out = []
        for r in res:
            s = re.sub(r"(\b[^.!?]+[.!?])(\s+\1)+", r"\1", self.tgt.decode(r.hypotheses[0]).strip())  # "Hi. Hi." -> "Hi."
            if s and (not out or s != out[-1]):
                out.append(s)
        return " ".join(out)


class LLMMT:
    """TranslateGemma-4B through this folder's own llama-server (runtime\\). The prompt is the model's own chat
    template (see logs\\translategemma-template.jinja), sent raw to /completion."""

    PROMPT = ("<start_of_turn>user\nYou are a professional Ukrainian (uk) to English (en) translator. Your goal is to "
              "accurately convey the meaning and nuances of the original Ukrainian text while adhering to English "
              "grammar, vocabulary, and cultural sensitivities.\nProduce only the English translation, without any "
              "additional explanations or commentary. Please translate the following Ukrainian text into English:"
              "\n\n\n{text}<end_of_turn>\n<start_of_turn>model\n")

    def __init__(self, url: str = LLAMA_URL):
        self.url = url

    def __call__(self, text: str) -> str:
        import json
        import urllib.request
        if not text.strip():
            return ""
        body = json.dumps({"prompt": self.PROMPT.format(text=text.strip()), "n_predict": 160, "temperature": 0,
                           "stop": ["<end_of_turn>"], "cache_prompt": True}).encode()
        req = urllib.request.Request(self.url + "/completion", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=30))["content"].strip()


class HyMT:
    """Tencent Hy-MT2 (1.8B / 7B) through this folder's llama-server, using the model's own chat template (--jinja)
    and the prompts from its model card. With context=N the previous N Ukrainian phrases go in as
    "[Background Information]", the card's documented way to give the model the conversation so far."""

    PLAIN = ("Translate the following text into English. Note that you should only output the translated result "
             "without any additional explanation:\n\n{text}")
    WITH_BG = ("[Background Information]\n{bg}\n\nPlease translate the following text into English, taking the "
               "provided background information into consideration.\n\n[Source Text]\n{text}")
    # Card recommendation for 1.8B/7B; greedy=True switches to deterministic decoding for comparison.
    SAMPLING = {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05}

    def __init__(self, url: str = LLAMA_URL, context: int = 0, greedy: bool = False):
        self.url, self.context, self.greedy = url, context, greedy
        self.history: list = []

    def __call__(self, text: str) -> str:
        import json
        import urllib.request
        text = text.strip()
        if not text:
            return ""
        bg = " ".join(self.history[-self.context:]) if self.context else ""
        prompt = self.WITH_BG.format(bg=bg, text=text) if bg else self.PLAIN.format(text=text)
        params = {"temperature": 0} if self.greedy else dict(self.SAMPLING)
        body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, **params}).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=30))["choices"][0]["message"]["content"].strip()
        self.history.append(text)
        return out


# Translator choices for --mt: name -> (GGUF served by llama-server or None, constructor). Hy-MT2-7B gets the two
# previous phrases as context (works on 7B; on 1.8B the context prompt leaked into the output, 2026-11-02).
TRANSLATORS = {
    "hymt7": ("hy-mt2\\Hy-MT2-7B-Q4_K_M.gguf", lambda: HyMT(context=2, greedy=True)),
    "hymt1.8": ("hy-mt2\\Hy-MT2-1.8B-Q8_0.gguf", lambda: HyMT(greedy=True)),    "gemma": ("translategemma-4b-it\\translategemma-4b-it.Q6_K.gguf", lambda: LLMMT()),
    "quickmt": (None, lambda device="cpu": MT(device)),
}


def load_qwen_tts(name: str):
    """A Qwen3-TTS model from MODELS through faster-qwen3-tts (CUDA graphs)."""
    # The `sox` package (pulled in by qwen-tts) probes for the SoX program on import and prints "SoX could not be
    # found!". Only the old 25 Hz tokenizer uses it; the 12 Hz models here never do, so a stub keeps it quiet.
    import sys
    import types

    def _no_sox(*args, **kwargs):
        raise RuntimeError("SoX is not installed; only the 25 Hz Qwen3-TTS tokenizer needs it")
    sys.modules.setdefault("sox", types.SimpleNamespace(Transformer=_no_sox))
    from faster_qwen3_tts import FasterQwen3TTS
    return FasterQwen3TTS.from_pretrained(str(MODELS / name), local_files_only=True)


class TTS:
    """Qwen3-TTS Base voice cloning with CUDA graphs. xvec_only=True needs no reference transcript and switches
    languages more cleanly (useful for a Ukrainian reference); ICL (False) needs the exact transcript."""

    def __init__(self, size: str = "0.6B", ref_audio=None, ref_text: str = "", xvec_only: bool = True):
        self.model = load_qwen_tts(f"Qwen3-TTS-12Hz-{size}-Base")
        self.ref_audio, self.ref_text, self.xvec_only = str(ref_audio) if ref_audio else None, ref_text, xvec_only

    def warmup(self) -> None:
        self.model.warmup(prefill_len=100)
        for _ in self.stream("Warming up the voice."):
            pass

    def stream(self, text: str, chunk_size: int = 4, trim: bool = True):
        """Yields (float32 audio chunk, sample rate). trim: drop the silence the model puts before the speech —
        ~0.45 s per phrase (2026-11-02); 50 ms before the first sound (> -45 dBFS in 20 ms frames) are kept."""
        head = None if trim else False
        for audio, sr, _ in self.model.generate_voice_clone_streaming(
                text=text, language="English", ref_audio=self.ref_audio, ref_text=self.ref_text,
                xvec_only=self.xvec_only, chunk_size=chunk_size):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if head is False:
                yield audio, sr
                continue
            head = audio if head is None else np.concatenate([head, audio])
            fr = int(0.02 * sr)
            loud = [i for i in range(0, len(head) - fr + 1, fr) if np.sqrt(np.mean(head[i:i + fr] ** 2)) > 10 ** (-45 / 20)]
            if loud:
                yield head[max(0, loud[0] - int(0.05 * sr)):], sr
                head = False
        if head is not None and head is not False and len(head):
            yield head, sr


CABLE = "CABLE Input"  # VB-CABLE: what is played into it comes out of the «CABLE Output» microphone


def device_arg(v, kind: str = None):
    """Device number, or part of its name (case-insensitive), resolved to an MME device of the given kind
    ("input"/"output"): a name stays valid when the numbering changes or when the Windows default device is switched
    to the cable by the game mode."""
    if v is None or str(v).isdigit():
        return int(v) if v is not None else None
    import sounddevice as sd
    ch = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, dev in enumerate(sd.query_devices()):
        if dev["hostapi"] == 0 and dev[ch] > 0 and str(v).lower() in dev["name"].lower():
            return i
    raise SystemExit(f"Аудіопристрій «{v}» не знайдено (список: Record-Samples.bat --list)")


class Player:
    """One open output stream fed from a queue, so streamed chunks play without gaps."""

    def __init__(self, device, sr: int = 24000):
        import sounddevice as sd
        self.sd, self.sr = sd, sr
        # None = the Windows default through the Sound Mapper, which finds the default each time it is opened:
        # PortAudio's own default is fixed when the program starts, so after the user switched Windows from the
        # speakers to the headphones the voice stayed on the speakers (2026-11-03). See refresh().
        self.follow = device is None
        self.device = device_arg("Sound Mapper - Output", "output") if device is None else device
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.buf = np.zeros(0, dtype=np.float32)
        self._open()

    def _open(self):
        self.stream = self.sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32", device=self.device,
                                           callback=self._cb, blocksize=1024)
        self.stream.start()

    def refresh(self) -> None:
        """Reopens the default output while nothing is playing, so it follows the current Windows default
        (called when the user starts speaking: the voice is silent then)."""
        if self.follow and self.q.empty() and not len(self.buf):
            self.stream.stop()
            self.stream.close()
            self._open()

    def _cb(self, out, frames, t, status):
        while len(self.buf) < frames and not self.q.empty():
            self.buf = np.concatenate([self.buf, self.q.get_nowait()])
        n = min(frames, len(self.buf))
        out[:n, 0] = self.buf[:n]
        out[n:, 0] = 0
        self.buf = self.buf[n:]

    def put(self, chunk: np.ndarray) -> None:
        self.q.put(chunk)

    def clear(self) -> None:
        """Drops everything not yet played (the «be quiet» key)."""
        while not self.q.empty():
            self.q.get_nowait()
        self.buf = np.zeros(0, dtype=np.float32)


class Players:
    """The same voice to several outputs at once: the user's headphones and the virtual microphone (VB-CABLE)."""

    def __init__(self, devices: list, sr: int = 24000):
        self.players = [Player(dev, sr) for dev in devices]

    def put(self, chunk: np.ndarray) -> None:
        for p in self.players:
            p.put(chunk)

    def clear(self) -> None:
        for p in self.players:
            p.clear()

    def refresh(self) -> None:
        for p in self.players:
            p.refresh()


def output_players(out, cable: bool) -> Players:
    """Headphones / speakers (`out`, default: the Windows default device) and, with cable=True, VB-CABLE too."""
    return Players([device_arg(out, "output")] + ([device_arg(CABLE, "output")] if cable else []))


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000
