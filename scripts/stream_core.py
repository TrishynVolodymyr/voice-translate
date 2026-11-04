"""Simultaneous (streaming) translation on top of the existing models — milestone M2.

Nothing here replaces the PTT pipeline: the same Parakeet / NeMo-uk, quickmt / Hy-MT2 are called more often.
- StreamingASR: every step the current speech segment is recognised again from its start; only words on which the
  last two passes agree count as stable (LocalAgreement-2). If Parakeet slips into Russian once, the rest of the
  segment is recognised by the Ukrainian-only NeMo model, so consecutive passes come from the same model.
- StreamingMT: the stable source is translated again with the English already spoken forced as the prefix
  (CTranslate2 `target_prefix` / raw completion after the assistant tag), so output is append-only; new English
  words are released only when two translations of different source prefixes agree on them.
- Segmenter: Silero VAD (sherpa-onnx) closes a segment after a pause; then the whole segment is recognised and the
  translation finished.
The idea of the policy follows LocalAgreement as used in whisper_streaming / NLLW / WhisperLiveKit; the code is ours.
"""
import json
import os
import queue
import re
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from pipeline import ASR, LLAMA_URL, MODELS, QUICKMT, TGT_NAME, slipped

SR = 16000


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower().replace("’", "'"))


def common_prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if norm(x) != norm(y):
            break
        n += 1
    return n


class StreamingASR:
    def __init__(self, threads: int = 6, hybrid: bool = True, provider: str = "cpu", kind: str = "parakeet"):
        self.parakeet = ASR(threads, provider, kind)
        self.uk = ASR(threads, provider, "nemo-uk") if hybrid else None
        self.reset()

    def reset(self):
        self.prev: list = []
        self.use_uk = False
        self.passes = 0
        self.offset = 0  # samples of the segment already translated and cut off
        self.word_ends: list = []

    # Confirmed words are recomputed on every pass (common prefix of the last two) rather than committed for good:
    # committing them (2026-11-03) locked in errors of early, short passes («шо», «наказав» for «не казав», a «.»
    # in mid-sentence) and the translation drifted from the whole-utterance one by 32% instead of 20%.
    def step(self, audio: np.ndarray, final: bool = False) -> str:
        """audio: whole current segment at 16 kHz (the already translated start is skipped).
        Returns the stable source text (everything when final)."""
        self.passes += 1
        audio = audio[self.offset:]
        model = self.uk if self.use_uk else self.parakeet
        text = model(audio, SR)
        if self.uk is not None and not self.use_uk and slipped(text):
            self.use_uk, self.prev, model = True, [], self.uk
            text = model(audio, SR)
        words = text.split()
        self.word_ends = self._word_ends(model.last)
        if len(self.word_ends) != len(words):  # token grouping disagrees with the text: don't cut on this pass
            self.word_ends = []
        if final:
            return text
        n = common_prefix(self.prev, words)
        self.prev = words
        return " ".join(words[:n])

    @staticmethod
    def _word_ends(res) -> list:
        """End time (s) of every whitespace-separated word, from token timestamps; [] if the model gives none."""
        toks, ts = list(res.tokens), list(res.timestamps)
        if not ts or len(ts) != len(toks):
            return []
        durs = list(getattr(res, "durations", []) or [])
        spans = []  # [start, end] per word
        for i, (tok, t) in enumerate(zip(toks, ts)):
            end = t + (durs[i] if i < len(durs) else 0.08)
            if tok.startswith(" ") or tok.startswith("▁") or not spans:
                spans.append([t, end])
            else:
                spans[-1][1] = end
        return spans

    def cut(self, n_words: int) -> bool:
        """Drops the audio of the first n_words (already translated) so later passes stay short.
        Cuts halfway between the end of word n and the start of word n+1."""
        if n_words <= 0 or n_words >= len(self.word_ends):
            return False
        end, nxt = self.word_ends[n_words - 1][1], self.word_ends[n_words][0]
        self.offset += int(max(end, (end + nxt) / 2 if nxt > end else end) * SR)
        self.prev = self.prev[n_words:]
        return True


class QuickMTPrefix:
    """quickmt (model for the VT_SOURCE_LANG-VT_TARGET_LANG pair) with a forced prefix (CTranslate2 target_prefix)."""

    def __init__(self, device: str = "cpu", threads: int = 2):  # 2 threads: see pipeline.MT
        import ctranslate2
        import sentencepiece as spm
        if device == "cuda":
            import torch  # CTranslate2 finds cuBLAS in torch's bundled CUDA libraries
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        d = MODELS / QUICKMT
        self.tr = ctranslate2.Translator(str(d), device=device, compute_type="int8" if device == "cpu" else "int8_float16",
                                         intra_threads=threads)
        self.src = spm.SentencePieceProcessor(model_file=str(d / "src.spm.model"))
        self.tgt = spm.SentencePieceProcessor(model_file=str(d / "tgt.spm.model"))

    def __call__(self, text: str, prefix: str = "", context: list = None) -> str:
        kw = {"target_prefix": [self.tgt.encode(prefix, out_type=str)]} if prefix else {}
        r = self.tr.translate_batch([self.src.encode(text, out_type=str) + ["</s>"]], beam_size=5,
                                    max_decoding_length=256, **kw)
        return self.tgt.decode(r[0].hypotheses[0]).strip()


class HyMTPrefix:
    """Hy-MT2 through the folder's llama-server: raw /completion with the model's chat template and the English
    already spoken appended after the assistant tag. Optional context = previous segments (card's
    "[Background Information]" prompt; works on 7B)."""

    PLAIN = ("Translate the following text into " + TGT_NAME + ". Note that you should only output the translated "
             "result without any additional explanation:\n\n{text}")
    WITH_BG = ("[Background Information]\n{bg}\n\nPlease translate the following text into " + TGT_NAME + ", taking "
               "the provided background information into consideration.\n\n[Source Text]\n{text}")

    def __init__(self, url: str = LLAMA_URL):
        self.url = url

    def __call__(self, text: str, prefix: str = "", context: list = None) -> str:
        bg = " ".join(context or [])
        user = self.WITH_BG.format(bg=bg, text=text) if bg else self.PLAIN.format(text=text)
        # The prompt is rendered by the server from the GGUF's own template: 7B and 1.8B use different special
        # tokens (<|startoftext|>…<|extra_0|> vs <｜hy_User｜>…<｜hy_Assistant｜>); hard-coding the 1.8B one made the
        # 7B hallucinate unrelated text.
        prompt = self._post("/apply-template", {"messages": [{"role": "user", "content": user}]})["prompt"] + prefix
        cont = self._post("/completion", {"prompt": prompt, "n_predict": 200, "temperature": 0,
                                          "cache_prompt": True})["content"]
        return (prefix + cont).strip()

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=30))


SENT_END = re.compile(r"(?<=[.!?…])\s+")


class StreamingMT:
    """Works sentence by sentence: once the stable source shows a sentence followed by another one, that sentence
    is translated to the end and frozen (with the whole utterance as one source, quickmt stopped after the forced
    prefix «Okay, then we got a deal.» and dropped the second sentence). Progress is counted in source words, not
    sentences: Parakeet moves punctuation between passes («Привіт.» → «Привіт, це…»), and counting sentences
    once lost a whole sentence. After step(), closed_now = source words frozen in this call (for StreamingASR.cut)."""

    def __init__(self, translate, context_segments: int = 0, hold: int = 2, max_open: int = 14):
        self.translate, self.context_segments, self.hold, self.max_open = translate, context_segments, hold, max_open
        self.history: list = []
        self.reset()

    def reset(self):
        """New utterance."""
        self.done_words = 0
        self.closed_now = 0
        self.spoken: list = []
        # (source, English) of a clause closed at a comma: the rest of the sentence is translated after it, with
        # its English forced, so it continues the sentence («…which should have been, because That I stopped a
        # little» when the rest was translated alone, live test 2026-11-03).
        self.carry = None
        self._new_sentence()

    def _new_sentence(self):
        self.emitted: list = []
        self.prev_cand: list = None
        self.last_src = ""

    def step(self, src: str, final: bool = False) -> list:
        """src: stable source of the utterance (after any cut). Returns English words to speak now (append-only)."""
        sents, cur = [], []
        for w in src.split()[self.done_words:]:
            cur.append(w)
            if re.search(r"[.!?…]$", w):
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        # Words already being translated as the open sentence stay one sentence even if a "." appeared inside them
        # later («Привіт, це короткий запис…» → «Привіт. Це короткий запис…» once printed the English twice).
        n_open = len(self.last_src.split())
        while n_open and len(sents) > 1 and len(sents[0]) < n_open:
            sents[0:2] = [sents[0] + sents[1]]
        # A long run-on sentence is closed at its last comma: otherwise nothing is frozen, the ASR window is never
        # cut and every pass re-reads ever longer audio (live test: 28 s without a full stop, passes up to 15 s apart
        # while other programs loaded the CPU). The clause must contain everything already being translated.
        clause_end = None
        if not final and sents and len(sents[-1]) >= self.max_open:
            open_ = sents[-1]
            for i in range(len(open_) - 3, max(n_open - 1, 4) - 1, -1):
                if re.search(r"[,;:—–]$", open_[i]):
                    sents[-1:] = [open_[:i + 1], open_[i + 1:]]
                    clause_end = len(sents) - 2
                    break
        out, self.closed_now = [], 0
        for j, s in enumerate(sents if final else sents[:-1]):
            new = self._sentence(" ".join(s), final=True)
            if j == clause_end and new:  # closed at a comma: no invented full stop
                new[-1] = new[-1].rstrip(".")
            out += new
            self.done_words += len(s)
            self.closed_now += len(s)
            if j == clause_end and self.emitted:  # without the invented full stop, or the rest starts a new sentence
                self.carry = (" ".join(s), self.emitted[:-1] + [self.emitted[-1].rstrip(".")])
            else:
                self.carry = None
            self._new_sentence()
        if not final and sents:
            out += self._sentence(" ".join(sents[-1]), final=False)
        out = self._drop_repeat(out)
        if out and self.spoken and self.spoken[-1].endswith(",") and out[0][:1].isupper() and out[0] != "I":
            out[0] = out[0][0].lower() + out[0][1:]  # the next clause after a comma-close: «…team, Not» → «not»
        self.spoken += out
        return out

    def _drop_repeat(self, out: list, k: int = 4) -> list:
        """Safety net: if a cut landed a bit early, the next sentence can start with source words already translated
        (seen once: «…faster, sometimes a bit slower. Sometimes I speak a bit faster, sometimes slower.»).
        Drops the leading words of `out` that repeat a run of >= k words from the last 30 spoken ones."""
        recent = [norm(w) for w in self.spoken[-30:]]
        o = [norm(w) for w in out]
        best = 0
        for start in range(len(recent)):
            n = 0
            while n < len(o) and start + n < len(recent) and recent[start + n] == o[n]:
                n += 1
            best = max(best, n)
        return out[best:] if best >= k else out

    def after_cut(self, n_words: int):
        """The ASR dropped the first n_words of the source (already translated)."""
        self.done_words -= n_words

    def _sentence(self, src: str, final: bool) -> list:
        if not src or (src == self.last_src and not final):
            return []
        self.last_src = src
        ctx = self.history[-self.context_segments:] if self.context_segments else None
        c_src, c_en = self.carry or ("", [])
        prefix = c_en + self.emitted
        hyp = re.sub(r"(\b[^.!?]+[.!?])(\s+\1)+", r"\1",
                     self.translate(f"{c_src} {src}".strip(), " ".join(prefix), ctx)).split()
        cand = hyp[common_prefix(prefix, hyp):]  # the forced prefix normally comes back unchanged
        if final:
            new = cand
        else:
            # Two translations of different source prefixes must agree, and the last `hold` agreed words wait for
            # more source: translations of an unfinished prefix tend to invent its ending («Мені подобаються» →
            # «I like that.», «іноді повільніше» → «…slower, sometimes faster.»).
            n = max(0, common_prefix(self.prev_cand or [], cand) - self.hold)
            new = [w for w in (x.rstrip(".!?…") if i == n - 1 else x for i, x in enumerate(cand[:n])) if w]
        self.emitted += new
        self.prev_cand = cand[len(new):]
        if final:
            self.history.append(src)
        return new


class Segmenter:
    """Silero VAD: tells when the current utterance has ended (pause >= min_silence)."""

    def __init__(self, min_silence: float = 0.6):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(MODELS / "silero-vad" / "silero_vad.onnx")
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.sample_rate = SR
        self.window = cfg.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        self.buf = np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> bool:
        """Feeds 16 kHz audio; returns True while speech is going on."""
        self.buf = np.concatenate([self.buf, audio])
        while len(self.buf) >= self.window:
            self.vad.accept_waveform(self.buf[:self.window])
            self.buf = self.buf[self.window:]
        return self.vad.is_speech_detected()


CLAUSE_END = re.compile(r"[.,!?;:…]$")


def squeeze_silence(audio: np.ndarray, sr: int, state: dict, max_gap: float = 0.6) -> np.ndarray:
    """Shortens abnormal pauses inside a voice piece to max_gap s, chunk by chunk (state carries the silence so far):
    once Qwen3-TTS put 2.75 s and 1.25 s of silence into a 12-word piece, and the voice fell 11 s behind. Only true
    silence (< -50 dB) is dropped and normal pauses are untouched: with -45 dB and 0.3 s the user heard cuts."""
    fr = int(0.02 * sr)
    keep = []
    for i in range(0, len(audio), fr):
        f = audio[i:i + fr]
        if np.sqrt(np.mean(f ** 2)) > 10 ** (-50 / 20) or np.abs(f).max() > 0.01:
            state["run"] = 0
        else:
            state["run"] = state.get("run", 0) + len(f)
            if state["run"] > max_gap * sr:
                state["dropped"] = state.get("dropped", 0) + len(f) / sr
                continue
        keep.append(f)
    return np.concatenate(keep) if keep else np.zeros(0, np.float32)


class Stretcher:
    """Plays a piece `speed` times faster without changing the pitch (WSOLA, audiotsm), chunk by chunk as Qwen3-TTS
    produces it, so a faster piece starts as soon as an ordinary one."""

    def __init__(self, speed: float):
        from audiotsm import wsola
        self.tsm = wsola(channels=1, speed=speed)

    def set_speed(self, speed: float):
        self.tsm.set_speed(speed)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        from audiotsm.io.array import ArrayReader, ArrayWriter
        w = ArrayWriter(channels=1)
        self.tsm.run(ArrayReader(chunk.reshape(1, -1)), w, flush=False)
        return w.data[0].astype(np.float32)

    def flush(self) -> np.ndarray:
        from audiotsm.io.array import ArrayWriter
        w = ArrayWriter(channels=1)
        while not self.tsm.flush_to(w)[1]:
            pass
        return w.data[0].astype(np.float32)


class Speaker:
    """Voices the stable English of the simultaneous mode while the user keeps talking (M2, step 2).

    Words are spoken in clause-sized pieces (up to a comma or full stop), so each piece gets natural intonation;
    without punctuation a piece goes after max_words words, or after `idle` seconds without new ones if it has at
    least min_words (single words spoken alone — «I», «think» — sounded choppy), or after `idle_any` seconds.
    The first piece of an utterance goes as soon as it has first_words words: waiting for a comma there made the
    voice start ~4 s after the user began (2026-11-02, the user's limit is 2 s). Pieces queue up: the
    next one is generated while the previous one plays. `log` keeps, per piece, when its text was ready, when its
    first audio was generated and when it starts/stops playing (seconds since t0) — the voice delay and whether the
    queue grows. `audio` keeps every chunk with its playing time for a timeline WAV.

    Catching up: English came out about as long as the user's Ukrainian, so on a 22 s monologue the voice finished
    10.6 s after the user (live test 2026-11-03). The voice is played faster without changing the pitch — `speed`
    when more than `catch_up` s of audio wait unplayed, `speed_max` when more than twice that (the user's ceiling is
    1.3×) — but only while there is something to catch up to: more text queued, or the user has finished the
    utterance (the translator sometimes releases ~20 words only then), with hysteresis between pieces. The speed is
    set once per piece and the stretcher is used only when it is above 1.0: changing it inside a piece and running
    every piece through the stretcher made the user hear cuts. skip() drops everything unsaid.
    Abnormal pauses inside a piece are shortened to 0.6 s, and a piece longer than max_per_word s per word (+1 s) is
    cut off. fx=False switches both the speed-up and the pause shortening off (A/B listening)."""

    def __init__(self, tts, player=None, t0: float = None, chunk_size: int = 4, max_words: int = 8, idle: float = 1.0,
                 min_words: int = 3, idle_any: float = 2.5, first_words: int = 1, catch_up: float = 1.5,
                 speed: float = 1.2, speed_max: float = 1.3, max_per_word: float = 0.8, fx: bool = True):
        self.tts, self.player, self.chunk_size = tts, player, chunk_size
        self.max_words, self.idle, self.min_words, self.idle_any = max_words, idle, min_words, idle_any
        self.first_words = first_words
        self.catch_up, self.speed, self.speed_max = catch_up, speed, speed_max
        self.release, self.hurry = 0.7, False  # hysteresis: back to normal speed only below `release` s behind
        self.fx = fx
        self.max_per_word = max_per_word  # a runaway piece (long pauses, babble) is cut off after this many s a word
        self.skipped = -1  # utterances up to this one are not voiced any more (skip key)
        self.generation = 0  # bumped by skip(): a piece being generated stops
        self.started: set = set()  # utterances whose first piece has gone
        self.t0 = time.perf_counter() if t0 is None else t0
        self.pending: list = []
        self.pending_seg = 0
        self.last_add = 0.0
        self.q: "queue.Queue" = queue.Queue()
        self.play_end = 0.0  # when the audio queued so far finishes playing
        self.log: list = []
        self.audio: list = []
        self.sr = 24000
        threading.Thread(target=self._work, daemon=True).start()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def add(self, words: list, seg: int, final: bool = False):
        if seg <= self.skipped:
            return
        if words:
            if not self.pending:
                self.pending_seg = seg
            self.pending += words
            self.last_add = self.now()
        cut = max((i + 1 for i, w in enumerate(self.pending) if CLAUSE_END.search(w)), default=0)
        if (final or len(self.pending) - cut >= self.max_words
                or (seg not in self.started and len(self.pending) >= self.first_words)):
            cut = len(self.pending)
        if cut:
            self._flush(cut, final and cut == len(self.pending))

    def tick(self):
        quiet = self.now() - self.last_add
        if self.pending and (quiet >= self.idle_any or (quiet >= self.idle and len(self.pending) >= self.min_words)):
            self._flush(len(self.pending))

    def _flush(self, n: int, last: bool = False):
        self.started.add(self.pending_seg)
        self.q.put((" ".join(self.pending[:n]), self.now(), self.pending_seg, last))
        self.pending = self.pending[n:]

    def skip(self, seg: int):
        """The «be quiet» key: drops queued pieces, the one being generated and what is not played yet; the rest of
        the current utterance is not voiced either (it would start mid-sentence)."""
        self.skipped, self.pending = seg, []
        self.generation += 1
        while not self.q.empty():
            self.q.get_nowait()
            self.q.task_done()
        if self.player:
            self.player.clear()
        now = self.now()
        self.audio = [(s, c) for s, c in self.audio if s < now]
        self.play_end = now

    def _place(self, entry: dict, chunk: np.ndarray, sr: int):
        if not len(chunk):  # the stretcher holds back its first input until it has a whole frame
            return
        t, self.sr = self.now(), sr
        start = max(t, self.play_end)
        self.play_end = start + len(chunk) / sr
        if "first_audio" not in entry:
            entry.update(first_audio=round(t, 2), play_start=round(start, 2))
        self.audio.append((start, chunk))
        if self.player:
            self.player.put(chunk)

    def _speed(self, last: bool, long_last: bool = False) -> float:
        """Chosen when a piece starts. Hysteresis keeps it from flapping: faster once more than catch_up s wait
        unplayed, back to normal only below `release` s. A long last piece (the user has stopped, nothing can come
        after it) is faster from its start."""
        if not self.fx:
            return 1.0
        behind = self.play_end - self.now()  # audio generated but not played yet
        if long_last:
            return self.speed_max if behind > 2 * self.catch_up else self.speed
        if not (last or not self.q.empty() or len(self.pending) >= self.min_words):
            self.hurry = False  # nothing more to say yet: hurrying would only leave a gap
        elif behind > self.catch_up:
            self.hurry = True
        elif behind < self.release:
            self.hurry = False
        if not self.hurry:
            return 1.0
        return self.speed_max if behind > 2 * self.catch_up else self.speed

    def _work(self):
        while True:
            text, ready, seg, last = self.q.get()
            gen = self.generation
            entry = {"seg": seg, "text": text, "ready": round(ready, 2), "gen_start": round(self.now(), 2),
                     "late": round(self.play_end - self.now(), 2)}
            limit, made, quiet = self.max_per_word * len(text.split()) + 1.0, 0.0, {}
            entry["speed"] = speed = self._speed(last, last and len(text.split()) >= 10)
            fast, sr = Stretcher(speed) if speed > 1.0 else None, self.sr
            for chunk, sr in self.tts.stream(text, self.chunk_size):
                if gen != self.generation:
                    break
                if self.fx:
                    chunk = squeeze_silence(chunk, sr, quiet)
                made += len(chunk) / sr
                self._place(entry, fast(chunk) if fast else chunk, sr)
                if made > limit:
                    entry["cut_off"] = round(made, 1)
                    break
            if fast and gen == self.generation:
                self._place(entry, fast.flush(), sr)
            if quiet.get("dropped"):
                entry["pause_cut_s"] = round(quiet["dropped"], 2)
            entry.update(gen_end=round(self.now(), 2), play_end=round(self.play_end, 2), skipped=gen != self.generation)
            self.log.append(entry)
            self.q.task_done()

    def drain(self):
        """Waits until every queued piece has been generated (not played)."""
        self.q.join()
