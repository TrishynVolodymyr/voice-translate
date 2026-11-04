# voice-translate

**Speak your language — others hear another one, in your own voice.** A local, offline voice translator for Windows:
hold a key, say a phrase, and a moment later it is spoken in the other language with your cloned timbre (without an
accent) — into your headphones and, through a virtual microphone, into Discord, games or any call. Nothing leaves your PC.

Out of the box it translates **Ukrainian → English**, the pair everything was tuned and measured on. **Other pairs are
two lines in `.env`**: you can speak 25 European languages, and the voice speaks 10 — English, German, French, Spanish,
Italian, Portuguese, Russian, Chinese, Japanese, Korean. See [Languages](#languages).

[Українська версія](README.uk.md) · [Roadmap](ROADMAP.md) · [Research and measurements](docs/RESEARCH.md) · License: MIT (code); models have their own licenses (see below)

## What it can do

- **Push-to-talk translation** — hold Right Ctrl, speak, release: the translation starts **~0.55 s after you release
  the key** (measured on an RTX 4070; up to ~1 s while a game loads the CPU). The whole phrase is translated at once, so quality is best here.
- **Your voice, without your accent** — the voice is cloned from a ~30 s recording of you speaking your own language;
  only the timbre is taken, the pronunciation of the other language comes from the model.
- **Or any other voice** — cloning is optional: create a voice from a text description ("a calm male voice in his
  thirties, slightly husky…"), or use any WAV file you have the right to use.
- **Discord and games** — the voice goes into a virtual microphone (VB-CABLE). While the translator runs, that
  microphone becomes the Windows default, so even games without a microphone setting hear it; your own default is
  restored when you quit.
- **Simultaneous mode (experimental)** — no key: the translated voice follows you while you keep talking, ~2.5–3.5 s
  behind, and speeds up a little (≤1.3×) to catch up.
- **Fully local** — speech recognition, translation and voice synthesis all run on your GPU/CPU; internet is needed only
  for installation.

## How it works

```
 your microphone
      │  (push-to-talk, or Silero VAD in simultaneous mode)
      ▼
 Parakeet (speech → text)                      NVIDIA Parakeet-TDT-0.6B-v3 / Parakeet Ultra, sherpa-onnx
      ▼
 translator (text → the other language)        quickmt (fast)  or  Hy-MT2 via llama.cpp (most accurate)  or  your GGUF
      ▼
 Qwen3-TTS (the translation, in your voice)    Qwen3-TTS 1.7B Base, voice clone from your sample, faster-qwen3-tts
      ▼
 your headphones  +  «CABLE Output» virtual microphone → Discord / game / call
```

Every stage is a separate, replaceable model. In simultaneous mode the recognizer re-reads the growing phrase every
0.5 s, words are released once two passes agree, the translator continues the already spoken translation, and the voice
speaks clause-sized pieces as they become stable.

## Requirements

- **Windows 10/11, 64-bit** (Windows only).
- **NVIDIA GPU with 12 GB VRAM** and a recent driver that supports CUDA 12.8. Push-to-talk with quickmt uses ~5 GB, with
  Hy-MT2-7B ~9.6 GB, the simultaneous mode ~8.5 GB; on an 8 GB card the quickmt launchers should fit (not tested).
- **~25 GB of free disk space**: environment ~6 GB, core models ~8 GB, optional models up to ~12 GB more.
- A microphone; **headphones** (in simultaneous mode the microphone must not hear the translated voice).
- Internet for the installation only. No accounts or tokens are needed.

**Tested on** one PC so far: Windows 11, NVIDIA RTX 4070 12 GB (driver 621.01), AMD Ryzen 5 7500F (6 cores /
12 threads), 96 GB DDR5 — including a from-scratch install of a fresh clone. Windows 10, other GPUs and 8 GB cards are
expected to work but have not been tried; reports are welcome.

## Installation

Everything is installed **inside this folder** — Python, packages, models. Delete the folder and it is gone (the only
exception is the optional VB-CABLE driver, see below). Use a short path without spaces, e.g. `C:\voice-translate`.

1. **Get the code**

   ```bash
   git clone https://github.com/TrishynVolodymyr/voice-translate.git C:\voice-translate
   ```

   or download the ZIP and unpack it.

2. **Build the environment** — double-click **`Install.bat`** (~6 GB of downloads; 2–15 minutes depending on your connection). It downloads `uv`,
   Python 3.11, PyTorch with CUDA and the other packages into this folder, checking SHA-256 checksums, and prints
   `CUDA: True` and the name of your GPU at the end.

3. **Download the models** — double-click **`Download-Models.bat`** (~8 GB). Optional extras, from a terminal:

   | Command | Adds | Size |
   |---|---|---|
   | `Download-Models.bat -Llm` | Hy-MT2-7B translator + llama.cpp — the most accurate translator (`VT_MT=hymt7`) | ~5 GB |
   | `Download-Models.bat -VoiceDesign` | model that creates a voice from a description (`Design-Voice.bat`) | ~4.2 GB |
   | `Download-Models.bat -Small` | smaller 0.6B voice (−2 GB VRAM, less natural) | ~2.4 GB |

   Every file is checked against its Hugging Face / GitHub checksum.

4. **Choose a voice** — one of:
   - **Your own voice:** run **`Record-Samples.bat`** and read the text shown on screen (~30 s) — Enter starts and
     stops the recording. Read it calmly, in your usual voice: the translator takes only the timbre, so the translation
     comes out without your accent. The text is Ukrainian or English, following `VT_SOURCE_LANG`; for another language
     read any ~30 s of your own. For Ukrainian, twenty short test phrases follow; they are optional (useful for checking
     recognition quality) — stop with Ctrl+C when the reference is done.
   - **A voice from a description:** `Download-Models.bat -VoiceDesign`, then
     `Design-Voice.bat "A calm female voice in her twenties, warm and clear, speaking at a relaxed pace" --name warm`
     and start the translator with `--ref warm` (or `VT_REF=warm` in `.env`). Each run gives a slightly different voice;
     keep the one you like.
   - **Any WAV:** put a clean 15–30 s recording into `voices\` and use `--ref <file name without .wav>`.

   Only clone voices you have the right to use — your own, or with the person's consent.

5. **Run** — double-click **`Start-VoiceTranslate.bat`**, wait for «Готово», hold **Right Ctrl**, speak, release.
   Quit with **`q` + Enter** in the window.

## Launchers

| File | Mode | Voice goes to | VRAM (app) |
|---|---|---|---|
| **`Start-VoiceTranslate.bat`** | **push-to-talk — the main one**: Discord, calls, games. Recognition and translation (quickmt) on the CPU, the voice on the GPU | headphones + virtual mic (if VB-CABLE is installed) | ~5 GB (`--tts 0.6B`: ~3 GB; `VT_MT=hymt7`: ~9.6 GB) |
| `Start-StreamVoice.bat` | simultaneous, no key (experimental) | headphones + virtual mic | ~8.5 GB |
| `Start-StreamTest.bat` | simultaneous, translated text only | screen | ~3.8 GB (`VT_ASR_DEVICE=cpu`: 0) |
| `Record-Samples.bat` | record your voice sample | — | — |
| `Design-Voice.bat` | create a voice from a description | — | ~5 GB |
| `Setup-VirtualCable.bat` | help installing VB-CABLE, keep your audio devices | — | — |

Launchers accept arguments, e.g. from PowerShell: `.\Start-VoiceTranslate.bat --ref warm --tts 0.6B`.

| Argument | Meaning |
|---|---|
| `--ref NAME` | voice sample from `voices\` (default `ref_<VT_SOURCE_LANG>`, e.g. `ref_uk`) or a path to a WAV |
| `--tts 1.7B` / `0.6B` | voice model size (0.6B: −2 GB VRAM, less natural) |
| `--mt quickmt` / `hymt7` | translator: fast and literal / most accurate (needs `-Llm`) |
| `--cable` | force the virtual microphone on (it is on by itself when the cable is found; `VT_CABLE=0` in `.env` turns it off) |
| `--key ctrl_r` | push-to-talk key (pynput names: `alt_r`, `f9`, `x`…) |
| `--mic NAME`, `--out NAME` | microphone / output device (part of its name or its number; `Record-Samples.bat --list`) |
| `--asr-device cuda` / `cpu` | where recognition (and quickmt) runs; also `VT_ASR_DEVICE` in `.env`. Push-to-talk uses the CPU (more accurate Parakeet Ultra, ~0.15 s slower than the GPU, which would take ~3.5 GB more); the simultaneous mode uses the GPU (a busy CPU cannot keep up with it) |

In simultaneous mode **Right Ctrl** means "be quiet": the voice drops everything it has not said yet.

## Languages

Set the pair in `.env` with ISO codes, for example Polish → German:

```
VT_SOURCE_LANG=pl
VT_TARGET_LANG=de
```

| Part | Languages |
|---|---|
| You speak (Parakeet recognizes it, no setting needed) | 25: bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk |
| The voice speaks (Qwen3-TTS) | 10: en zh ja ko de fr ru pt es it — **no Ukrainian**, so English → Ukrainian is not possible yet |
| Translator Hy-MT2-7B (`Start-VoiceTranslate.bat`, needs `Download-Models.bat -Llm`) | from cs de en es fr it nl pl pt ru uk into any of the 10 voice languages; other source languages may work, untested |
| Translator quickmt (`-QuickMT`, `-Cable`, `-Game`, simultaneous mode) | only pairs into or from English (`sv-en`, `en-de`, …), one small model per pair |

After changing the pair, run `Download-Models.bat` again: it fetches the quickmt model for the new pair if there is one.
The program checks the pair at start and says what is missing; its own messages are in Ukrainian for now (English is
on the roadmap). Your voice sample can be in any language — only its timbre is used.

### Translator

The translator is a replaceable part: quickmt by default, anything else through `VT_MT` in `.env`:

| `VT_MT` | What it is | VRAM | Per phrase (RTX 4070) |
|---|---|---|---|
| *(empty)* | quickmt | | |
| `quickmt` | small dedicated model per pair, only into / from English | little, or none on the CPU | ~0.05 s |
| `hymt7` | Hy-MT2-7B, the most accurate | ~5 GB | ~0.2 s |
| `hymt1.8` | Hy-MT2-1.8B, any pair of its languages at a small cost | ~1.9 GB | ~0.1 s (on the CPU: ~0.7–0.9 s) |
| `models\my-model.gguf` | **any GGUF** — e.g. your own fine-tune; a path relative to this folder or absolute | depends | |

`Download-Models.bat` fetches what `VT_MT` names (and llama.cpp for any GGUF); your own GGUF you put in place yourself.
A GGUF runs in the folder's llama-server with its own chat template and gets the Hy-MT2 instruction
(«Translate the following text into German…»), so a Hy-MT2 fine-tune fits as is; any chat model that follows such an
instruction works too. A `--mt …` typed after a launcher wins over `.env`.

**Where it runs.** At start the program prints which part runs on the CPU and which on the GPU. A GGUF translator goes
to the GPU. Next to a demanding game put it on the **CPU** with `VT_MT_NGL=0` in `.env`, to leave the VRAM to the game —
slower (Hy-MT2-1.8B ~0.7–0.9 s per phrase instead of ~0.1 s); a number of layers splits it between the two.

Example — Polish → German, where quickmt has no model: `VT_MT=hymt1.8`, then `Download-Models.bat`, and every launcher
uses Hy-MT2-1.8B.

Checked: Ukrainian → English in depth (every number in this README); Ukrainian → German, Ukrainian → French and
English → German briefly (translation and voice, the voice recognized back word for word). The simultaneous mode counts
words by spaces, so for Chinese, Japanese or Korean output use push-to-talk.

## Configuration (`.env`)

Copy `.env.example` to `.env` next to the launchers and fill in only what you want to change — an empty value means the
default. `.env` stays on your PC (it is git-ignored).

- **When it is read:** by every launcher and by `Download-Models.bat`, on each start. After a change close the window
  and start again; after changing the language pair or `VT_MT` run `Download-Models.bat` again — it fetches what the
  new setting needs.
- **Which value wins:** an argument typed after the launcher > a variable already set in Windows > `.env` > the default.
- **Did it take effect?** At start the window shows the language pair (when it is not Ukrainian → English) and the
  «Де що працює» table: which model does what, on the CPU or the GPU. A setting that cannot work stops the start with
  a message saying what to fix.

All of this was checked on one PC (Windows 11, RTX 4070 12 GB, Ryzen 5 7500F). "Not checked" below means we could not try it
there; the reason it should work is given next to it.

| Variable | What it sets | Empty = | Checked |
|---|---|---|---|
| `VT_SOURCE_LANG`, `VT_TARGET_LANG` | language pair, ISO codes ([Languages](#languages)) | `uk`, `en` | Ukrainian → English in depth (every number here); Ukrainian → German, Ukrainian → French, English → German briefly (translation and voice). Other pairs not checked — the models list these languages; a language recognition or the voice does not know stops the start, one missing from Hy-MT2's list gets a warning. |
| `VT_MT` | translator ([Translator](#translator)) | quickmt | quickmt, `hymt7`, `hymt1.8`, and a GGUF given by its path. Not checked: someone else's fine-tune — it gets the same prompt through its own chat template, as the GGUF test did. |
| `VT_MT_NGL` | layers of a GGUF translator on the GPU | all | `0`, `10`, `999` (Hy-MT2-1.8B with the 1.7B voice: 6.5 / 7.2 / 8.4 GB VRAM for the whole card) |
| `VT_ASR_DEVICE` | where recognition and quickmt run: `cpu` / `cuda` | push-to-talk: CPU; simultaneous: GPU | both, with timings ([Performance](#performance-ukrainian--english-rtx-4070-12-gb-ryzen-5-7500f-measured)) |
| `VT_CABLE` | virtual microphone; `0` = off | on when the cable is found | found by itself; `0` keeps it off; your microphone comes back after `q` + Enter and after closing the window |
| `VT_CABLE_PLAY`, `VT_CABLE_MIC` | another virtual cable: where the voice is played / the microphone others hear | VB-CABLE | Not checked — only VB-CABLE was installed. The same lookup by part of the name finds VB-CABLE; a name it cannot find is reported at start, and the voice then stays in your headphones. |
| `VT_MODELS_DIR`, `VT_VOICES_DIR`, `VT_LOGS_DIR` | folders, relative to this one or absolute | `models`, `voices`, `logs` | `VT_VOICES_DIR` in another folder. Others not checked — all scripts read them through the same code. The launcher's own files (llama.cpp log, saved microphone) always stay in `logs\`. |
| `VT_REF` | voice sample: a file name in the voices folder | `ref_<VT_SOURCE_LANG>` | your recorded voice and a voice made by `Design-Voice.bat` |
| `VT_MIC`, `VT_OUT` | microphone / output: part of the name or a number | the Windows default | both by name; the default output follows Windows when you switch between speakers and headphones |
| `VT_PUSH_KEY` | push-to-talk key, also "be quiet" in simultaneous mode | `ctrl_r` | Right Ctrl; other key names not checked |
| `VT_TTS_PACE` | push-to-talk: generate the voice at most this many times faster than it plays (e.g. `1.5`), so the GPU rests between chunks | no limit (~2.8×) | without a game: GPU busy 73% → 63% (1.5) / 56% (1.2), the first sound and continuity unchanged, total work and power about the same ([RESEARCH §3](docs/RESEARCH.md)). Effect on game FPS not measured yet. |
| `VT_LLAMA_PORT` | port of the local llama.cpp server (127.0.0.1 only) | `8093` | the default port only |

## Discord and games (virtual microphone)

1. Install **VB-CABLE** (donationware, free for personal use) — the easiest way is **`Setup-VirtualCable.bat`**: it
   remembers your microphone and speakers, opens <https://vb-audio.com/Cable/>, and after you install it there
   (unpack, `VBCABLE_Setup_x64.exe` as administrator → *Install Driver*) switches your devices back — the VB-CABLE
   installer makes the cable your default microphone *and* speakers. Reboot afterwards. This is the only thing installed
   outside the folder. (Its license does not allow bundling it, so you download it yourself from the official page.)
2. If you installed it by hand, check Windows sound settings and choose your own microphone and speakers again, or from
   PowerShell: `. .\scripts\AudioDefault.ps1; Get-Mics; Set-DefaultMic 'your mic'; Get-Speakers; Set-DefaultSpeaker 'your headphones'`.
3. Run **`Start-VoiceTranslate.bat`** — it finds the cable by itself. While it runs, «CABLE Output» is the default
   microphone; quit with `q` + Enter, or simply close the window: your microphone is restored either way (a small
   hidden helper waits for the window to close). `VT_CABLE=0` in `.env` turns the cable off, e.g. to play the
   translation through speakers to someone in the room.
   **Another virtual cable** (VoiceMeeter, Virtual Audio Cable, your own): name its two sides in `.env` —
   `VT_CABLE_PLAY` (the playback device the voice goes into) and `VT_CABLE_MIC` (the microphone other programs hear);
   part of the name is enough.
4. **Discord:** Settings → Voice & Video → Input device = «CABLE Output». Voice activity mode works: the cable is
   silent while the voice is silent.
5. **Games:** the same `Start-VoiceTranslate.bat` — recognition and quickmt run on the CPU, only the voice on the GPU
   (~5 GB), which fits next to a demanding game in 12 GB. With recognition or a big translator on the GPU (~8.5–9.6 GB)
   it does not: the voice then becomes 8–20× slower than real time while the game keeps its FPS. With a GGUF translator
   set `VT_MT_NGL=0`; if the voice stutters in heavy scenes, add `--tts 0.6B`. To experiment: `VT_TTS_PACE=1.5` makes
   the voice take the GPU in smaller bursts (not yet measured in a game). Capping the game's FPS leaves the voice more GPU
   time. Games use the default microphone automatically; otherwise Windows Settings → System → Sound → Volume mixer → the
   game → Input device = «CABLE Output», or Steam → Settings → Voice → Recording device.

Your own voice is not heard there — only the translated one. Checked with Discord and Telegram voice messages.

## If something does not work

- **`CUDA: False` at the end of `Install.bat`** — update the NVIDIA driver (it must support CUDA 12.8) and run it again.
- **Windows blocks the `.bat` files** (SmartScreen, "Windows protected your PC") — for a downloaded ZIP: right-click the
  ZIP → Properties → *Unblock* before unpacking, or *More info* → *Run anyway*.
- **It hears the wrong microphone / plays to the wrong device** — `Record-Samples.bat --list` shows the devices; set
  `VT_MIC` / `VT_OUT` in `.env` (part of the name is enough).
- **Out of GPU memory** — close games or other GPU programs, or use `--tts 0.6B`, `--asr-device cpu`.
- **The voice sounds cut or robotic in a call** — try turning off the app's noise suppression (Krisp in Discord), echo
  cancellation and automatic gain: they are made for live microphones, not for synthesized speech.

## Performance (Ukrainian → English, RTX 4070 12 GB, Ryzen 5 7500F, measured)

| What | Result |
|---|---|
| Push-to-talk: key released → first sound of the translation | ~0.55 s median, max ~0.77 s (quickmt and recognition on the CPU; 20 phrases of 2.5–6.4 s); ~1.05 s with the CPU loaded 100%; ~0.41 s with `VT_ASR_DEVICE=cuda`; ~0.65 s with Hy-MT2-7B |
| Simultaneous: start of a phrase → first sound of the translation | ~2.5–3.5 s on live speech |
| Recognition errors on one speaker (40 phrases) | ~8% WER (Parakeet v3 fp32 on GPU / Parakeet Ultra on CPU) |
| CPU use | push-to-talk: busy only while a phrase is processed (~0.2–0.4 s); simultaneous mode: ~2–3% (recognition on the GPU) |
| In Path of Exile 2 (recognition and quickmt on the CPU) | first sound 0.6–1.0 s after release (1.7B voice; 0.6B: 0.55–1.3 s); game 80–90 FPS while speaking (130–145 without) |

Translation quality: Hy-MT2-7B is the most accurate (times, numbers, negation, uses the previous phrases as context);
quickmt is literal but ~30 ms per phrase.

## Folder layout

| Folder | Content | In git |
|---|---|---|
| `scripts\` | the program (Python) and PowerShell helpers | yes |
| `*.bat` | launchers | yes |
| `voices\` | your voice samples | **no** |
| `logs\` | recordings and measurements of your sessions | **no** |
| `models\`, `runtime\`, `.venv\`, `python\`, `tools\`, `cache\`, `downloads\` | created by the installers | no |
| `.env` | your local settings | no |

**Uninstall:** delete the folder. If you installed VB-CABLE: run `VBCABLE_Setup_x64.exe` as administrator → *Remove
Driver*, reboot.

## Licenses

The code in this repository is **MIT** (see [LICENSE](LICENSE)). The models and libraries are downloaded from their
authors and keep their own licenses — all of them allow personal and commercial use:

| Component | License | In short |
|---|---|---|
| NVIDIA Parakeet-TDT-0.6B-v3, Moondream Parakeet Ultra, quickmt models | CC-BY-4.0 | anything, with attribution |
| Qwen3-TTS (Base, VoiceDesign), Hy-MT2 (7B, 1.8B), sherpa-onnx, transformers | Apache-2.0 | anything, keep the notices |
| Silero VAD, llama.cpp, faster-qwen3-tts, CTranslate2, audiotsm | MIT | anything, keep the notice |
| pynput (keyboard hook) | LGPL-3.0 | used unmodified as a library |
| VB-CABLE (optional, not bundled) | donationware | free for personal use |

Details: [THIRD_PARTY.md](THIRD_PARTY.md). This project would not exist without the open licenses of these components.

## What is not solved yet

The simultaneous mode is a working prototype: the voice starts ~3 s after you, pieces are audibly joined, and errors of
the recognizer are spoken before they can be corrected. See [ROADMAP.md](ROADMAP.md) for what we measured and what we
would try next.
