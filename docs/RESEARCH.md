# Research notes: what we measured and why the pipeline looks like this

[Українська версія](RESEARCH.uk.md) · [README](../README.md) · [Roadmap](../ROADMAP.md)

> All measurements were made on one PC — Windows 11, NVIDIA RTX 4070 12 GB (driver 621.01), AMD Ryzen 5 7500F
> (6 cores / 12 threads), 96 GB DDR5 — with one speaker's voice: the 20 standard test phrases from
> `Record-Samples.bat`, recorded twice (40 recordings, 181 words per take), plus live sessions. Dates: November 2026.
> The numbers are indicative, not benchmarks: small sets, one voice, one machine, and some runs were made while other
> programs were using the CPU.

Terms used below. **WER**: word error rate against the known text, ignoring case and punctuation. **Take A / take B**:
the two recordings of the 20 phrases (take B: closer to the microphone, cardioid pattern, fewer hesitations; noise was
not a factor in either, SNR ~47 dB). **Divergence**: WER of the English produced in simultaneous mode against the
English produced from the whole phrase (Parakeet on the whole phrase + quickmt on the whole text). It measures
stability, not quality against a human reference.

## 1. Speech recognition

### Accuracy on one speaker (40 recordings, `scripts/bench_asr.py`)

| Model | Runs on | WER take A | WER take B | Time per phrase |
|---|---|---|---|---|
| Parakeet-TDT-0.6B-v3 int8 | CPU | 12.2% | 9.4% | ~215 ms |
| **Parakeet Ultra int8** (community int8 export) | CPU | **7.7%** | **8.3%** | ~190 ms |
| **Parakeet-TDT-0.6B-v3 fp32** | GPU | 8.3% | 8.8% | ~165 ms |
| Parakeet Ultra int8 | GPU | 8.3% | 8.8% | ~230 ms |
| NeMo FastConformer CTC uk (Ukrainian-only) | CPU | 26.0% | 23.2% | ~106 ms |
| Nemotron 3.5 ASR Streaming int8, best chunk size | CPU | 18.2% | 20.4% | streaming |

- Earlier runs of Parakeet v3 int8 gave 11.0–12.2% (A) and 8.8–9.9% (B), depending on the onnxruntime build
  (the CUDA build of sherpa-onnx was the worse one). int8 on CUDA is not faster than on the CPU (1026 vs 926 ms for 20 s
  of audio), so the GPU uses the fp32 export.
- Published scores do not transfer to conversational speech: the Ukrainian-only FastConformer reports 5.66% on Common
  Voice 12 but made 26% here; Nemotron's card reports 13–16% on FLEURS-uk, here 18–25%.
- Parakeet Ultra (a fine-tuned Parakeet v3, same architecture) fixed typical errors of this speaker — «Я не встигаю»
  is no longer heard as «на», hesitations and repeated word starts are dropped — but writes some numbers as digits and
  occasionally changes a verb form. It is the default when recognition runs on the CPU. On the GPU, v3 fp32 stays the
  default (Ultra int8 on CUDA is neither more accurate nor faster); there is no fp32 export of Ultra yet.

### Speed: CPU vs GPU, idle vs fully loaded CPU

Time per recognition pass by audio length. "Loaded" = 12 busy-loop processes occupying every CPU thread.

| Model, device | 3 s | 7 s | 20 s | 7 s / 20 s, CPU loaded | VRAM |
|---|---|---|---|---|---|
| Parakeet int8, CPU | 144 ms | 266 ms | 741 ms | 611 / 1549 ms | — |
| NeMo uk, CPU | 47 ms | 99 ms | 289 ms | 272 / 734 ms | — |
| **Parakeet fp32, GPU** | **31 ms** | **48 ms** | **132 ms** | **62 / 141 ms** | ~3.4 GB |
| NeMo uk, GPU | 21 ms | 29 ms | 47 ms | 40 / 76 ms | ~1.0 GB |

In a simultaneous text test (20 phrases back to back, quickmt), recognition on the CPU kept up on an idle CPU (~26%
CPU used; end of speech → end of translation 1.04 s) but not at 100% CPU load (3.17 s; only 6 of 20 phrases done in
89 s). With recognition on the GPU the results were the same idle or loaded (1.01 / 1.06 s, ~2–3% CPU). Under full load
a CPU pass of Parakeet grew from 0.26 s to 1–2 s and the first English text came after 19 s.

Pitfalls: if PyTorch initializes CUDA first, sherpa-onnx cannot load its CUDA provider (DLL error 1114) and silently
falls back to the CPU, so recognition is loaded first; on CUDA, audio shorter than 0.8 s took ~220 ms instead of
~80 ms, so short audio is padded with silence to 1 s.

### Nemotron 3.5 ASR Streaming (truly streaming, sherpa-onnx int8, CPU, language `uk`)

| Chunk | WER take A | WER take B | First whole word after speech start (median, A / B) |
|---|---|---|---|
| 160 ms | 23.8% | 22.7% | 0.95 / 0.89 s |
| 320 ms | 20.4% | 22.7% | 1.10 / 1.03 s |
| 560 ms | 18.8% | 25.4% | 1.09 / 1.07 s |
| 1120 ms | 18.2% | 20.4% | 1.65 / 1.66 s |
| *Parakeet v3 fp32 GPU, whole phrase* | *8.3%* | *8.8%* | *our re-reading scheme: 0.9–1.5 s* |

Cheap to run (up to ~0.1 s per chunk on the CPU) and the partial text only grows, but it makes 2.5–3× more errors for
a 0–0.5 s gain. Language `uk`/`uk-UA` gives Ukrainian; `auto` or `ru` drifts into Russian. Rejected.

### Language slips and the hybrid fallback

Parakeet has no language setting. «Хто йде зі мною на правий фланг?» came out in Russian in both takes, and once a
phrase came out in Polish Latin script («Обережно, за тим будинком» → «Obereżne za tym budynkiem»); short partial
passes in simultaneous mode slip more often. A hybrid was tried: if Parakeet's text contains ы/э/ъ/ё or mostly Latin
letters, re-recognize with the Ukrainian-only NeMo model. On the read phrases it helped (GPU: 7.2% / 6.6% instead of
8.3% / 8.8%). In a 46-phrase live session it was triggered on 19 phrases, mostly by English words and interjections
written in Latin script, and the NeMo text was much worse (merged and broken words). The hybrid is therefore off; a
better trigger would look only at Russian-only or Polish-only letters. Remaining systematic errors: «не»→«на»,
«Та ні»→«Тані». No translator repairs a phrase that was recognized in the wrong language.

## 2. Translation

Set: 28 sentences — the 20 standard phrases as recognized (with their recognition errors), 6 with clock times and 2
with numbers. All LLMs with greedy decoding (temperature 0), so results are deterministic; the set is small.

| Sentence | quickmt (CPU) | TranslateGemma-4B Q6_K | Hy-MT2-1.8B Q8_0 | **Hy-MT2-7B Q4_K_M** |
|---|---|---|---|---|
| до одинадцятої вечора | eleven o'clock in the evening ✓ | 10 PM ✗ | 11 p.m. ✓ | 11 p.m. ✓ |
| о пів на третю | half past three ✗ | 10:30 AM ✗ | 3:30 p.m. ✗ | 2:30 ✓ |
| о пів на восьму | 7:30 ✓ | half past eight ✗ | half past eight ✗ | half past seven ✓ |
| за чверть дванадцята | a quarter of 12 ≈ | 12:15 ✗ | a quarter to twelve ✓ | a quarter to twelve ✓ |
| двісті п'ятдесят тисяч гривень | two hundred and fifty thousand ✓ | 50,000 ✗ | 250,000 ✓ | 250,000 ✓ |
| лишилося сімнадцять відсотків | 17 percent of the budget ≈ | 17% remaining ✓ | 17 percent less ✗ | seventeen percent left ✓ |
| за тим будинком ще двоє | Watch out for the other two ✗ | two other houses nearby ✗ | around that house—two others ≈ | two more behind that house ✓ |
| підемо на боса | go to the boss ≈ | we'll go for it ✗ | go ahead without any preparation ✗ | go take on the boss ✓ |
| «Привіт, усім чути мене.» (no «?») | everyone can hear me ≈ | can you hear me? ✓ | I hope you can hear me ≈ | can everyone hear me? ✓ |
| Time per phrase | 40–80 ms CPU / ~30 ms GPU | ~200 ms | ~120 ms | ~210 ms |
| VRAM | 0 (CPU) | ~3.4 GB | ~2.2 GB | ~4.7–4.9 GB |

- **Hy-MT2-7B** was the only one without errors in times and numbers, and keeps questions as questions.
  **Hy-MT2-1.8B** is roughly at quickmt's level (better on some sentences, worse on others). **TranslateGemma-4B** was
  the weakest on numbers and times. **quickmt** is literal but fast.
- Sampling matters: the vendor-recommended sampling (temperature 0.7, top_p 0.6, top_k 20) turned «до одинадцятої» into
  "7 p.m." on 1.8B; greedy decoding got it right.
- Context: giving Hy-MT2-7B the two previous phrases helps with garbled recognition. In one live case the recognizer
  turned «не» into «на» and every translator inverted the meaning; 7B with the previous phrase as context restored the
  negation. On 1.8B the same context format broke the output (it echoed parts of the prompt).
- Bugs found and fixed: without the `</s>` token at the end of the source, quickmt decoded to the maximum length and
  looped on short phrases (this looked like "1–2 s per phrase"); with six CPU threads on a busy CPU, quickmt sometimes
  stalled for up to 86 s (median 77 ms) — on 2 threads it was ~110 ms, on the GPU 32 ms (+0.4 GB VRAM). Under a 100%
  busy CPU even 2 threads stalled (40 phrases: median 1.3 s, max 23 s), 1 thread did not (median 0.16 s, max 0.55 s),
  so quickmt now uses 1 thread on the CPU (2026-11-04, measured with other programs also loading the CPU).
  quickmt also invents continuations for interjections. Hy-MT2-7B and 1.8B use different chat templates, so the
  prompt is built by llama-server (`/apply-template`).

Keeping only part of Hy-MT2-7B on the GPU saves little: all layers ~4.9 GB and ~0.21–0.24 s per phrase, `-ngl 28`
~4.3 GB and ~0.52 s, `-ngl 24` ~3.7 GB and ~0.77 s.

**Hy-MT2-30B-A3B (MoE, Q4_K_M, 17 GB) with experts in system RAM** — VRAM is the whole card, including ~1.45 GB used
by the desktop:

| Setup | VRAM | Translation, average |
|---|---|---|
| all experts in RAM (`--n-cpu-moe 999`) | 3.4 GB | 1.13 s |
| experts of 9 layers on GPU | 6.5 GB | 0.92 s |
| experts of 18 layers on GPU | 9.7 GB | 0.74 s |
| all in RAM + 1.8B speculative draft, Q4_K_M / Q8_0 | 4.4 / 5.1 GB | 1.36 / 1.34 s |

Generation ~23–27 tokens/s, prompt reading ~15–20 ms per token, both bound by the experts on the CPU. The draft was
accepted 58–65% of the time but made things slower: verifying several tokens at once touches more experts in RAM.
Quality: 5 of 28 translations identical to 7B, the rest mostly style; no errors in times and numbers, one regression
("behind" → "near"); no jump in quality over 7B (0.18 s median on the GPU). With two previous phrases as context the prompt grew from ~30 to 59–86
tokens, and live translation took 2.38–2.44 s (first sound ~3.05 s after the key). Too slow; removed from the project.

## 3. Voice (Qwen3-TTS via faster-qwen3-tts, CUDA graphs)

Cloned from a ~20 s Ukrainian sample; 5 English phrases of 1.4–8 s.

| Variant | First audio chunk | Speed vs real time | VRAM (PyTorch) |
|---|---|---|---|
| 0.6B, timbre only (xvec) | 174–187 ms | 2.5–2.9× | 2.6 GB |
| **1.7B, timbre only (xvec)** — default | 173–182 ms | 2.5–2.6× | 4.5 GB |
| 0.6B, continue the sample (ICL) | 270–305 ms | 1.6–2.2× | 2.9 GB |
| 1.7B, continue the sample (ICL) | 294–325 ms | 1.1–1.8× | 4.9 GB |

- **xvec** takes only the timbre; the English pronunciation comes from the model, so the voice has no Ukrainian accent.
  In listening tests 1.7B sounded noticeably more natural than 0.6B.
- **Leading silence:** Qwen3-TTS puts ~0.45 s (0.28–0.72 s) of silence before each phrase. It is now trimmed (threshold
  −45 dBFS, 50 ms kept). The first *audible* sound comes ~0.32 s after the request (codec chunk 4; chunk 2: 0.33 s,
  chunk 1: 0.37 s — smaller chunks do not help). Once it inserted 2.75 + 1.25 s of silence into 12 words, hence the
  pause squeeze in §5.
- **GPU load while speaking** (1.7B, sampled every 100 ms, 3.8–4.6 s phrases generated in 1.6–2.05 s): utilization
  median 88–89% (max 96–99%), power median 55–70 W of the card's ~200 W; idle ~1%. "Utilization" is the share of time
  any kernel runs: the autoregressive steps are small and sequential.

**Push-to-talk end to end** (`scripts/bench_e2e.py`: recorded phrases fed as if the key had just been released;
Parakeet v3 int8 on the CPU):

| Translator | Key release → first audio chunk, median | Min–max | Parts |
|---|---|---|---|
| TranslateGemma-4B Q6_K | 666 ms | 539–863 ms | ASR 0.21–0.41 s + MT 0.12–0.30 s + TTS ~0.18 s |
| **Hy-MT2-7B Q4_K_M** | **640 ms** | 536–955 ms | ASR 0.20–0.36 s + MT 0.11–0.38 s + TTS ~0.18 s |

This bench counted the first audio chunk, before the leading silence was trimmed. Recognition and translation grow
roughly linearly with phrase length: a 36 s monologue took 1.8 s to recognize, 1.6 s to translate (Hy-MT2-7B) and
3.6 s to the first sound; with quickmt, live phrases of 7.9–37.7 s gave 0.97–5.90 s.

### Push-to-talk: recognition and quickmt on the CPU or the GPU (2026-11-04)

20 recorded phrases (2.5–6.4 s) fed as if the key had just been released; the 1.7B voice on the GPU in every row.
"Loaded" = 12 busy-loop processes on every CPU thread — harder than a game (in Path of Exile 2 the CPU setup gave
0.6–1.0 s, §6).

| Recognition + quickmt | CPU | Recognition, median | quickmt, median | Key release → first sound, median (max) |
|---|---|---|---|---|
| CPU: Parakeet Ultra int8, quickmt int8 on 1 thread | idle | 140 ms | 120 ms | 0.54–0.56 s (0.77) |
| GPU: Parakeet v3 fp32, quickmt int8_float16 | idle | 96 ms | 19 ms | 0.41 s (0.55) |
| CPU | loaded 100% | 447 ms | 217 ms | 1.05 s (1.34) |
| GPU | loaded 100% | 129 ms | 36 ms | 0.51 s (0.60) |

- The GPU saves ~0.14 s on an idle CPU and ~0.55 s on a fully loaded one, for ~3.5 GB more VRAM; recognition on the
  CPU (Ultra) is the more accurate one (§1; the two recognizers wrote 9 of these 20 phrases differently). So
  push-to-talk defaults to the CPU, the simultaneous mode to the GPU (`VT_ASR_DEVICE` changes it).
- quickmt translates the same on both: 18 of 20 identical on the same input, the other two differ by a comma / "we'll".
- quickmt threads on a fully loaded CPU (40 translations; other programs also loaded the CPU during this one): 2 threads
  median 1.3 s, max 23 s; 2 threads with `OMP_WAIT_POLICY=PASSIVE` 0.19 / 0.45 s; 1 thread 0.16 / 0.55 s — now 1 thread.

### Voice pace limit (`VT_TTS_PACE`, 2026-11-04)

The 20 translated phrases (~3 s each) spoken by the 1.7B voice; after the first 1 s of speech, generation is held to
the given multiple of real time, so the GPU rests between chunks. GPU sampled every 100 ms while generating.

| Limit | Speed over the whole phrase | GPU busy | Power | First chunk | Gaps a 1× player would hear |
|---|---|---|---|---|---|
| off | 2.8× | 72–73% | 79 W | 0.29 s | none* |
| 2× | 2.7× | 73% | 80 W | 0.29 s | none |
| 1.5× | 2.2× | 63% | 80 W | 0.29 s | none* |
| 1.2× | 1.75× | 56% | 76 W | 0.29 s | none* |

\* 0.01–0.09 s in total over 62 s of speech, the same with and without the limit — timing noise, not audible.

It works without side effects — the first sound and the continuity are unchanged — but it spreads the same work over
a longer time: the GPU is busy a smaller share of the time, while power hardly drops (the card stays at high clocks).
On short phrases the unlimited first second dilutes the effect. Whether a game gets more FPS from it is not measured
yet; the expected gain is smaller, more even bursts of GPU load while the voice speaks.

## 4. VRAM budget (12 GB card; the desktop itself used ~1.5–2.4 GB)

| Setup | Used by the app | Free on the card | Use |
|---|---|---|---|
| quickmt (CPU) + voice 0.6B | 3.0 GB | ~7.6 GB | lightest, next to games |
| quickmt (CPU) + voice 1.7B | 4.9 GB | ~5.7 GB | calls / games |
| Hy-MT2-1.8B + voice 1.7B, recognition on CPU | ~7.0 GB (measured 2026-11-04) | ~4 GB | quality ≈ quickmt; `VT_MT=hymt1.8` for pairs quickmt lacks (0.1 s per phrase; on the CPU 0.65–0.9 s) |
| TranslateGemma-4B + voice 1.7B | 8.4 GB | ~1.5 GB | not used (numbers) |
| Hy-MT2-7B (all layers) + voice 1.7B, recognition on CPU | 9.6 GB, peak 9.8 | ~0.75 GB at peak | most accurate, tight |
| Hy-MT2-30B-A3B experts in RAM + voice 1.7B, recognition on CPU | 7.9 GB for the whole card | ~4.4 GB | 7B quality, too slow |
| Parakeet fp32 (GPU) + quickmt (GPU) + voice 1.7B | 10.5 GB for the whole card | ~1.8 GB | simultaneous mode |

Hy-MT2-7B together with GPU recognition does not fit, so with LLM translators recognition runs on the CPU. The
current per-launcher figures are in the README.

## 5. Simultaneous mode

How it works: Silero VAD closes an utterance after a 0.6 s pause; Parakeet re-reads the growing utterance every 0.5 s
and a word is confirmed when two passes agree; the translator continues the English already released (forced target
prefix) and releases new words when two consecutive translations agree, holding back the last 2; finished sentences
are fixed and their audio is cut off by word timestamps (an open sentence of ≥14 words is closed at its last comma, so
the window stays ≤ ~6–7 s); the voice speaks clause-sized chunks (to a comma or period, 8 words, 1 s of silence with
≥3 words, or 2.5 s), and the first chunk of each phrase is spoken at once.

**Where the start goes** (median from phrase start, live session replayed, 18 phrases): first recognition pass
0.9 s → first confirmed Ukrainian word 1.5 s → **first English word 3.5 s** → first sound 3.9 s. Half of it is the
translator: offline translators change the beginning as words arrive («А ні, я сказав» → "Oh no, I said"; with
«навпаки» → "No, I said the opposite"), so English appears only after ~4 Ukrainian words agree. On the read test
phrases the first English text comes at a median of 1.5–1.9 s; live speech is slower. A comma that flips between passes
(«Так, ну» ↔ «Так ну») made translations disagree and delayed one phrase start to 4.4 s.

**Policies tried** (start = median first sound or first English word from phrase start; divergence on a 32-phrase,
7.5-minute live session replayed):

| Change | Start | Divergence | Decision |
|---|---|---|---|
| Baseline: re-read every 0.5 s, first voice chunk ≥2 words | 3.8 s | 19.3% (8 phrases >30%) | — |
| First word of a phrase spoken at once | 3.5 s (read phrases, 0.25 s step: 2.75 → 2.41 s) | 20.2% (8) | **kept** |
| Re-read every 0.25 s instead of 0.5 s | 3.3 s | 26.1% (11) | rejected: near-identical passes agree on wrong words (0.2 s: no further gain) |
| Hold back 1 word instead of 2 | read phrases, 0.25 s step: 2.41 → 2.17 s | 3 of 20 phrases worse (lost or doubled words at joins) | rejected |
| Never revoke confirmed words (+ clause context) | 3.4 s | 32.3% | rejected: locks in errors of early short passes |
| Clause context carry (the rest of a sentence cut at a comma is translated after its first part) | 3.6 s | 19.1% | **kept** |
| Minimum 3 words per later voice chunk | — | single-word chunks lasted ~0.9 s and broke intonation | **kept** |

**AlignAtt** (release an English word when the translator's attention points at already-stable source) was simulated
word by word on the 20 phrases, without recognition, using quickmt's attention:

| Rule | Ukrainian words before the first English word | Divergence from whole-phrase translation |
|---|---|---|
| **Agreement of two translations, hold 2 (current)** | 4.05 | 13.8% |
| Agreement, hold 1 | 3.35 | 21.6% |
| AlignAtt, last 4 tokens unsafe | 4.00 | 12.1% |
| AlignAtt, last 2 words unsafe | 3.40 | 18.5% |
| AlignAtt, last 2 tokens unsafe | 2.50 | 23.3% |
| AlignAtt, last token unsafe | 1.85 | 30.6% |

The same earlier-vs-worse curve as holding back words: each Ukrainian word is ≈0.35–0.4 s of speech and ≈4–8 points
of divergence. quickmt's attention is diffuse (max 0.05–0.5, often a word ahead). Not adopted.

**Voice tail and catch-up.** English is about as long as Ukrainian (86 s of English for 92 s of Ukrainian in one
session), so on continuous speech the lag only grows. Seconds from the end of each phrase until the voice stops
(15 phrases, one live session replayed):

| Variant | Per phrase | Total English |
|---|---|---|
| Live, no catch-up | 2.8 1.8 2.3 10.6 9.6 8.4 12.2 4.9 4.0 5.6 5.4 3.9 1.9 1.9 3.9 | 85.9 s |
| 1.2× / 1.35× from 2 s lag + pause squeeze | 2.8 1.9 2.7 6.9 5.9 7.3 6.5 5.5 4.9 5.3 4.4 4.2 3.2 1.6 4.8 | 75.3 s |
| 1.3× / 1.5× from 1.5 s lag | 2.6 2.0 2.4 6.0 5.1 5.7 5.7 5.2 4.5 5.4 4.0 3.5 2.6 1.4 4.2 | 70.8 s |

Listening put the comfortable ceiling at 1.2–1.3× (1.5× is too fast). Current behaviour: pitch-preserving speed-up
(WSOLA, audiotsm) of 1.2× / 1.3×, decided at the start of each chunk with hysteresis (faster above 1.5 s of lag,
normal below 0.7 s) and only when there is something to catch up; pauses longer than 0.6 s below −50 dB inside a chunk
are squeezed; a chunk longer than 0.8 s per word + 1 s is cut; the "be quiet" key drops everything not yet spoken. In
an A/B on the same recording, tails without processing were 2.6 / 5.7 / 2.4 / 7.3 s and with it 3.0 / 7.6 / 4.3 /
7.1 s: the run-to-run spread is larger than the effect. The floor is ~2.5–3 s after the end of speech (0.6 s pause +
finishing the translation + generation + playing the last chunk).

**Replay-variance caveat.** Replays feed recorded audio in real time, but translation timing and chunking differ
between runs (±0.5 s on tails). A test-player bug counted the last piece of each file as a full 0.1 s, so replays
before the fix ran ~0.05 s per file slower than real time (up to ~0.9 s over 46 phrases) and overstated the lag; live
microphone tests were not affected.

**Resources.** CPU ~30–45% of all cores with quickmt on the CPU, ~20–28% with Hy-MT2 on the GPU, ~2–3% with
recognition and translation on the GPU. Resampling the microphone 48→16 kHz with torchaudio used ~36% of all cores in
silence; `scipy.signal.resample_poly` uses ~0.6%. Recognition next to the voice on the GPU: 119 ms median (89–114 ms
without the voice).

## 6. Games: Path of Exile 2

| Mode | Voice speed vs real time | Key release → first sound | Game FPS while speaking |
|---|---|---|---|
| Full mode (~8.5 GB) next to the game | 8–20× **slower** than real time (VRAM overflow, the game keeps priority) | — | game keeps its FPS |
| Recognition and quickmt on the CPU (today's default), voice 1.7B | 1.07–1.10× faster | 0.6–1.0 s | 80–90 (130–145 without) |
| The same, voice 0.6B (`--tts 0.6B`) | 1.1–1.3× faster | 0.55–1.3 s | 80–90 (130–145 without) |

This setup runs recognition and translation on the CPU and only the voice on the GPU; measured without heavy
effects. Headroom is small (1.07–1.3×); buffering a whole phrase when generation falls below real time is in the
roadmap; the voice pace limit (§3) is available as `VT_TTS_PACE`, its effect in a game is not measured yet.

## 7. What exists elsewhere and why we did not use it (as of November 2026)

No open project found does local Ukrainian speech → English speech in the speaker's voice while they talk. Open
simultaneous systems output text; voice has to be added.

| Project | What it is | Why not used here |
|---|---|---|
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | streaming ASR + simultaneous translation (LocalAgreement, AlignAtt), server | text only; ASR backends Whisper, Voxtral, Qwen3-ASR, Canary — no Parakeet; Whisper is weak on Ukrainian (large-v3 20.5% vs 4–4.5% for NeMo Ukrainian models on CV10, [community benchmark](https://github.com/egorsmkv/speech-recognition-uk)). We use the same kind of agreement policy. |
| [SimulStreaming](https://github.com/ufal/SimulStreaming) (IWSLT 2025 winner) | Whisper + AlignAtt + EuroLLM-9B | text only; Whisper large-v3 alone needs ≥10 GB VRAM, nothing left for the voice |
| [SeamlessStreaming](https://huggingface.co/facebook/seamless-streaming) | one speech-to-speech model, accepts Ukrainian | generic voice; ending offset X→English 2.66–2.90 s ([paper](https://arxiv.org/html/2312.05187)); non-commercial license; on Windows only via WSL2/Docker; last updated Dec 2023 |
| [Hibiki-Zero](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) | simultaneous speech-to-speech with voice transfer | fr/es/pt/de→en only; adapting to Italian took ~850 h of synthetic data and the training code is not published ([paper](https://arxiv.org/html/2602.11072)); non-commercial |
| [Nemotron 3.5 ASR Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | the only truly streaming recognizer with good Ukrainian support | measured in §1: 18–25% WER vs ~8% for Parakeet, 0–0.5 s gain |
| [Vosk uk](https://alphacephei.com/vosk/models) | streaming Kaldi models | ~53% WER on CV10 (community figure), no punctuation, archived |
| Other recognizers: [Canary-1B-v2](https://huggingface.co/nvidia/canary-1b-v2), [nemotron-labs-unified-asr](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b), Qwen3-ASR, Voxtral Realtime, Kyutai STT, Moonshine | recent ASR models | Canary: Ukrainian, but not streaming; unified Nemotron: NeMo on Linux only so far; the rest: no Ukrainian (Moonshine only in older non-streaming models) |
| Ready-made pipelines: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech), [LiveTranscriber](https://github.com/alexwing/LiveTranscriber) | VAD/ASR → LLM or NLLB → TTS | the first is a voice assistant that waits for the end of the turn; the second works by sentences with 5.9–12.6 s latency, Ukrainian input not confirmed |
| Simultaneous MT models: [AlignAtt4LLM](https://github.com/QuentinFuxa/Alignatt4LLM), [Hy-SimulMT-30B-A3B](https://huggingface.co/tencent/Hy-SimulMT-30B-A3B), [TalTech](https://aclanthology.org/2026.eacl-demo.40), [StreamSpeech](https://github.com/ictnlp/StreamSpeech) | LLM or speech models trained or prompted for simultaneous translation | AlignAtt4LLM needs vLLM (Linux/WSL2); Hy-SimulMT is Chinese↔English only (~20 GB); TalTech has Ukrainian only as a target, no weights; StreamSpeech is fr/es/de→en only |

Translators: on FLORES-200 Ukrainian→English, quickmt-uk-en reports 39.88 BLEU vs 39.07 for OPUS-MT tc-big zle-en and
33.59 for NLLB-1.3B ([quickmt-uk-en](https://huggingface.co/quickmt/quickmt-uk-en)); no open translator was found that
beats Hy-MT2-7B for Ukrainian. No translator trained on incomplete sentences exists for Ukrainian→English. The
reference point is the IWSLT 2026 winner on Czech→English (a Slavic source with free word order): Parakeet + an
unmodified LLM (Qwen3.5-27B) + longest-common-prefix agreement and an "input may be incomplete, do not guess"
instruction, COMET 0.89 at an average lag (LongYAAL) of 1.9 s ([findings](https://aclanthology.org/2026.iwslt-1.39.pdf),
[NeMo system](https://aclanthology.org/2026.iwslt-1.23.pdf)). Average lag is not the delay of the first word, so it
compares with our numbers only roughly. Next steps are in the [roadmap](../ROADMAP.md).

## 8. Pitfalls checklist

What bit us, how we noticed, and what we did — for anyone building a similar local pipeline on Windows.

| Pitfall | How it showed up | What we did |
|---|---|---|
| quickmt without the end-of-source token `</s>` | short phrases looped ("Thank you. Thank you…"), 1–2 s on the CPU | append `</s>` as the quickmt library does: 40–80 ms |
| CTranslate2 with several OpenMP threads while the CPU is busy | single phrases up to 86 s (6 threads) or 23 s (2 threads) | 1 thread on the CPU (median 0.16 s, max 0.55 s at 100% CPU), or the GPU |
| Resampling every 0.1 s mic chunk with torchaudio | 36% of all cores while silent | `scipy.signal.resample_poly`: 0.6% |
| sherpa-onnx CUDA after PyTorch has started CUDA | CUDA provider fails (DLL error 1114) and silently falls back to the CPU | load recognition before the voice model |
| Parakeet on CUDA with < 0.8 s of audio | ~220 ms instead of ~80 ms — and the first pass of every phrase is that short | pad short audio with silence to 1 s |
| Recognizing on a busy CPU | passes 1–2 s instead of 0.26 s, the simultaneous mode falls behind | recognition and translation on the GPU (~2–3% CPU) |
| Re-reading a growing utterance with no full stops | window up to 28 s, passes up to 15 s apart | close long sentences at a comma, cut the translated audio |
| Commas flipping between passes | translations differ, the first word waits 4+ s | open (roadmap: compare without punctuation) |
| Committing confirmed words for good | early-pass errors locked in, divergence 20% → 32% | reverted: confirmation recomputed every pass |
| A clause translated without the clause before it | "…because That I stopped a little" | previous clause as source context, its English as forced prefix |
| Qwen3-TTS leading silence | ~0.45 s before every piece; with 37 pieces the queue grew to 5 s | trim leading silence (first sound 0.4 s earlier) |
| Runaway pauses inside a piece | 2.75 s + 1.25 s of silence in a 12-word piece, voice 11 s behind | pauses > 0.6 s squeezed, piece capped at 0.8 s per word |
| Choosing the speed once per piece | the voice raced to the edge of the translation, then stalled | speed only when there is something to catch up to, hysteresis |
| Time-stretching audio that is not sped up | listeners heard cuts | stretch only pieces played faster |
| Our own replay harness | the last chunk of each file counted as a full 0.1 s: replays drifted up to ~0.9 s | count the real chunk length |
| PortAudio fixes the default output when the program starts | after switching Windows to headphones the voice stayed on the speakers | play through "Microsoft Sound Mapper", which follows the Windows default |
| Device numbers change after a reboot | a test recorded the wrong microphone | find devices by (part of) their name |
| VB-CABLE installer | makes the cable the default microphone and speakers | `Setup-VirtualCable.bat` remembers and restores your devices |
| VRAM overflow next to a game | no error: Windows pages the program's memory to RAM, the game keeps priority, speech generation becomes 8–20× slower than real time | game mode: only the voice on the GPU (~5 GB) |
