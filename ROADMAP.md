# Roadmap: towards real simultaneous translation

Push-to-talk works well (~0.6–0.7 s from releasing the key to the translated speech). The simultaneous mode
(`Start-StreamVoice.bat`) is a working prototype. Below is what is still unsatisfying, what we measured, and what we would
try next. Numbers are from Ukrainian → English, an RTX 4070 12 GB and one speaker's live tests.

[Українська версія](ROADMAP.uk.md)

## Where the ~3.5 s start goes

Median from the start of a phrase on live speech: first recognition pass 0.9 s → first confirmed Ukrainian word 1.5 s
(+0.6) → **first English word 3.5 s (+2.0)** → first sound 3.9 s (+0.4; the TTS model itself needs 0.32 s).

Half of it is the translator: offline translators change their translation of the beginning when more words arrive
(«А ні, я сказав» → "Oh no, I said"; with «навпаки» → "No, I said the opposite"), so English is released only after
~4 Ukrainian words agree across two translations, minus 2 held-back words.

## Open problems and ideas

### Start latency (goal: ≤ 2 s)
- [ ] **Translator trained for incomplete sentences** (prefix-to-prefix / "simultaneous" fine-tune). None exists for
  Ukrainian→English as of Nov 2026. Cheap recipe to try: self-distilled prefix pairs (a prefix's target = the common
  start of its translation and the full sentence's translation), LoRA on Hy-MT2-1.8B or a 4B LLM.
- [ ] **LLM translator with an "input may be incomplete, do not guess" instruction** — the IWSLT 2026 winner (Parakeet +
  an unmodified LLM + common-prefix agreement) did this; worth a word-by-word simulation on Hy-MT2-7B.
- [ ] **Punctuation-insensitive agreement** — Parakeet flips commas between passes («Так, ну» ↔ «Так ну»); translations
  then differ and the first word can wait 4+ s. ~1 in 4 phrases is affected.
- [ ] Ready-made translations for discourse markers at the start («так», «ну», «коротше» → "so", "well", "anyway").
- Tried and rejected: re-reading every 0.25 s instead of 0.5 s (−0.25 s, but translation diverged 26% vs 20% from the
  whole-phrase one); holding back 1 word instead of 2 (−0.25 s, lost/duplicated words at joins); AlignAtt on quickmt's
  attention (same latency/quality curve as holding back); never revoking confirmed words (locked in errors of early
  passes: 32% divergence).

### Recognition
- [ ] **Fine-tune Parakeet on the user's voice** — 1–2 h of the user's own speech recorded for it, with corrected
  transcripts (a "listen and fix" page); expected to fix systematic errors («не»→«на», «Та ні»→«Тані»).
- [ ] **Language slips** — Parakeet sometimes writes Ukrainian speech in Russian or Polish («Обережно, за тим будинком» →
  «Obereżne za tym budynkiem»), more often on short partial passes. A Ukrainian-only fallback model (NeMo FastConformer
  uk) fixed this but was triggered by English words too (QuickMT, "in this spirit") and recognized worse — so it is off.
  Better trigger: only Russian-only / Polish-only letters, not Latin in general.
- [ ] fp32 export of Parakeet Ultra for the GPU (int8 Ultra on the CPU is already better than v3 fp32 on the GPU).
- [ ] A truly streaming Ukrainian recognizer of Parakeet quality. Nemotron 3.5 ASR Streaming was measured: 18–25% WER on
  the user's voice vs ~8% for Parakeet, for only 0–0.5 s gain — rejected. NVIDIA's unified streaming/offline models
  (Nemo only, Linux) are worth watching.

### Voice in simultaneous mode
- [ ] **Audible joins** — every clause is a separate TTS generation with its own intonation. Ideas: continuing
  generation with the previous piece as acoustic context; fewer, longer pieces where latency allows.
- [ ] **The voice runs behind on long monologues** — English is about as long as Ukrainian, so the lag grows; catch-up
  playback (≤1.3×, the listener's ceiling) helps only partly. Shorter translations ("summarize while translating") are
  an option.
- [ ] Interjections ("Mm-hmm", "Hmm") make quickmt invent continuations ("Mm-hmm is a good idea"): pass them through
  untranslated, cap the translation length.

### Games and resources
- Measured in Path of Exile 2: the full mode next to the game generated speech 8–20× slower than real time (VRAM
  overflow, the game keeps priority); the game mode (`Start-VoiceTranslate-Game.bat`, recognition and translation on the
  CPU) speaks 1.07–1.10× faster than real time with the 1.7B voice (first sound 0.6–1.0 s) and 1.1–1.3× with 0.6B
  (0.55–1.3 s); game 80–90 FPS (130–145 without), measured while running without heavy effects.
- [ ] The game mode has little headroom (1.07–1.3×): when generation falls below real time, buffer the whole phrase
  and play it in one piece instead of stuttering.
- [ ] TTS keeps the GPU busy ~88% of the time while speaking (~1/3 of its power): a duty-cycle limit (e.g. 70%) for
  games.
- [ ] Less VRAM: fp16 Parakeet export (~1.2 GB instead of 2.4).
- Tried and rejected: Hy-MT2-30B-A3B with experts in RAM (7B quality at ~1.9 GB VRAM, but 1.1–2.4 s per phrase);
  speculative decoding with a 1.8B draft (slower: experts in RAM make verification expensive).

### Product
- [ ] Detecting that the other person interrupts (the program does not hear the call) — today: the "be quiet" key.
- [ ] One-click installer; Linux support.
- Other language pairs: set in `.env` (`VT_SOURCE_LANG` / `VT_TARGET_LANG`, see README → Languages); briefly checked
  uk→de, uk→fr, en→de.
- [ ] A voice model that speaks Ukrainian (Qwen3-TTS has 10 languages without it), for English → Ukrainian.
- [ ] Measure quality and speed on other pairs; simultaneous mode for Chinese / Japanese / Korean output (no spaces
  between words).
- [ ] Program messages in English too (today they are Ukrainian only); test phrases for recognizing other languages.
