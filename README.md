# voice-translate

**Speak Ukrainian — others hear English, in your own voice.** A local, offline voice translator for Windows: hold a key,
say a phrase in Ukrainian, and a moment later it is spoken in English with your cloned timbre (without an accent) —
into your headphones and, through a virtual microphone, into Discord, games or any call. Nothing leaves your PC.

[Українська версія](README.uk.md) · [Roadmap](ROADMAP.md) · [Research and measurements](docs/RESEARCH.md) · License: MIT (code); models have their own licenses (see below)

## What it can do

- **Push-to-talk translation** — hold Right Ctrl, speak, release: English starts **~0.6–0.7 s after you release the
  key** (measured on an RTX 4070). The whole phrase is translated at once, so quality is best here.
- **Your voice, without your accent** — the voice is cloned from a ~30 s Ukrainian recording; only the timbre is taken,
  the English pronunciation comes from the model.
- **Or any other voice** — cloning is optional: create a voice from a text description ("a calm male voice in his
  thirties, slightly husky…"), or use any WAV file you have the right to use.
- **Discord and games** — the voice goes into a virtual microphone (VB-CABLE). While the translator runs, that
  microphone becomes the Windows default, so even games without a microphone setting hear it; your own default is
  restored when you quit.
- **Simultaneous mode (experimental)** — no key: the English voice follows you while you keep talking, ~2.5–3.5 s
  behind, and speeds up a little (≤1.3×) to catch up.
- **Fully local** — speech recognition, translation and voice synthesis all run on your GPU/CPU; internet is needed only
  for installation.

## How it works

```
 your microphone
      │  (push-to-talk, or Silero VAD in simultaneous mode)
      ▼
 Parakeet (speech → Ukrainian text)            NVIDIA Parakeet-TDT-0.6B-v3 / Parakeet Ultra, sherpa-onnx
      ▼
 translator (Ukrainian → English text)         quickmt-uk-en (fast)  or  Hy-MT2-7B via llama.cpp (most accurate)
      ▼
 Qwen3-TTS (English speech in your voice)      Qwen3-TTS 1.7B Base, voice clone from your sample, faster-qwen3-tts
      ▼
 your headphones  +  «CABLE Output» virtual microphone → Discord / game / call
```

Every stage is a separate, replaceable model. In simultaneous mode the recognizer re-reads the growing phrase every
0.5 s, words are released once two passes agree, the translator continues the already spoken English, and the voice
speaks clause-sized pieces as they become stable.

## Requirements

- **Windows 10/11, 64-bit** (Windows only).
- **NVIDIA GPU with 12 GB VRAM** and a recent driver that supports CUDA 12.8. The default modes use ~8.5–9.6 GB; on an
  8 GB card start them with `--tts 0.6B` or `--asr-device cpu` (expected to fit, not tested).
- **~25 GB of free disk space**: environment ~6 GB, core models ~8 GB, optional models up to ~12 GB more.
- A microphone; **headphones** (in simultaneous mode the microphone must not hear the English voice).
- Internet for the installation only. No accounts or tokens are needed.

**Tested on** one PC so far: Windows 11, RTX 4070 12 GB, Ryzen 5 — including a from-scratch install of a
fresh clone. Windows 10, other GPUs and 8 GB cards are expected to work but have not been tried; reports are welcome.

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
   | `Download-Models.bat -Llm` | Hy-MT2-7B translator + llama.cpp — needed by `Start-VoiceTranslate.bat` (best quality) | ~5 GB |
   | `Download-Models.bat -VoiceDesign` | model that creates a voice from a description (`Design-Voice.bat`) | ~4.2 GB |
   | `Download-Models.bat -Small` | smaller 0.6B voice (−2 GB VRAM, less natural) | ~2.4 GB |

   Every file is checked against its Hugging Face / GitHub checksum.

4. **Choose a voice** — one of:
   - **Your own voice:** run **`Record-Samples.bat`** and read the text shown on screen (~30 s) — Enter starts and
     stops the recording. Read it in Ukrainian, calmly, in your usual voice: the translator takes only the timbre, so
     the English comes out without your accent. Twenty short test phrases follow; they are optional (useful for checking
     recognition quality) — stop with Ctrl+C when the reference is done.
   - **A voice from a description:** `Download-Models.bat -VoiceDesign`, then
     `Design-Voice.bat "A calm female voice in her twenties, warm and clear, speaking at a relaxed pace" --name warm`
     and start the translator with `--ref warm` (or `VT_REF=warm` in `.env`). Each run gives a slightly different voice;
     keep the one you like.
   - **Any WAV:** put a clean 15–30 s recording into `voices\` and use `--ref <file name without .wav>`.

   Only clone voices you have the right to use — your own, or with the person's consent.

5. **Run** — double-click a launcher, wait for «Готово», hold **Right Ctrl**, speak Ukrainian, release.
   Quit with **`q` + Enter** in the window.

## Launchers

| File | Mode | Voice goes to | VRAM (app) |
|---|---|---|---|
| `Start-VoiceTranslate-QuickMT.bat` | push-to-talk, fast translator (quickmt) | headphones | ~8.5 GB |
| `Start-VoiceTranslate.bat` | push-to-talk, most accurate translator (Hy-MT2-7B; needs `-Llm`) | headphones | ~9.6 GB |
| **`Start-VoiceTranslate-Cable.bat`** | push-to-talk (quickmt) **for Discord and calls** | headphones + virtual mic | ~8.5 GB |
| **`Start-VoiceTranslate-Game.bat`** | push-to-talk **next to a demanding game**: recognition and translation on the CPU, 1.7B voice on the GPU (`--tts 0.6B`: lighter, needs `-Small`) | headphones + virtual mic | ~5 GB (0.6B: ~3 GB) |
| `Start-StreamVoice.bat` | simultaneous, no key (experimental) | headphones | ~8.5 GB |
| `Start-StreamTest.bat` / `Start-StreamTest-CPU.bat` | simultaneous, English text only | screen | ~3.8 GB / 0 |
| `Record-Samples.bat` | record your voice sample | — | — |
| `Design-Voice.bat` | create a voice from a description | — | ~5 GB |
| `Setup-VirtualCable.bat` | help installing VB-CABLE, keep your audio devices | — | — |

Launchers accept arguments, e.g. from PowerShell: `.\Start-VoiceTranslate.bat --ref warm --tts 0.6B --cable`.

| Argument | Meaning |
|---|---|
| `--ref NAME` | voice sample from `voices\` (default `ref_uk`) or a path to a WAV |
| `--tts 1.7B` / `0.6B` | voice model size (0.6B: −2 GB VRAM, less natural) |
| `--mt quickmt` / `hymt7` | translator: fast and literal / most accurate (needs `-Llm`) |
| `--cable` | also send the voice into the VB-CABLE virtual microphone |
| `--key ctrl_r` | push-to-talk key (pynput names: `alt_r`, `f9`, `x`…) |
| `--mic NAME`, `--out NAME` | microphone / output device (part of its name or its number; `Record-Samples.bat --list`) |
| `--asr-device cpu` | recognition on the CPU (frees ~3.4 GB VRAM, sensitive to CPU load) |

In simultaneous mode **Right Ctrl** means "be quiet": the voice drops everything it has not said yet.

## Configuration (`.env`)

Copy `.env.example` to `.env` to change defaults without arguments: model / voice / log folders, default microphone and
output, voice sample, push-to-talk key, llama.cpp port. `.env` stays on your PC (it is git-ignored).

## Discord and games (virtual microphone)

1. Install **VB-CABLE** (donationware, free for personal use) — the easiest way is **`Setup-VirtualCable.bat`**: it
   remembers your microphone and speakers, opens <https://vb-audio.com/Cable/>, and after you install it there
   (unpack, `VBCABLE_Setup_x64.exe` as administrator → *Install Driver*) switches your devices back — the VB-CABLE
   installer makes the cable your default microphone *and* speakers. Reboot afterwards. This is the only thing installed
   outside the folder. (Its license does not allow bundling it, so you download it yourself from the official page.)
2. If you installed it by hand, check Windows sound settings and choose your own microphone and speakers again, or from
   PowerShell: `. .\scripts\AudioDefault.ps1; Get-Mics; Set-DefaultMic 'your mic'; Get-Speakers; Set-DefaultSpeaker 'your headphones'`.
3. Run **`Start-VoiceTranslate-Cable.bat`**. While it runs, «CABLE Output» is the default microphone; quit with
   `q` + Enter and your microphone is restored (if you close the window with ×, it is restored on the next start).
4. **Discord:** Settings → Voice & Video → Input device = «CABLE Output». Voice activity mode works: the cable is
   silent while the voice is silent.
5. **Games:** use **`Start-VoiceTranslate-Game.bat`** — the full mode (~8.5 GB) does not fit next to a demanding game in
   12 GB, and then the voice becomes 8–20× slower than real time while the game keeps its FPS. The game mode
   (recognition and translation on the CPU, only the voice on the GPU) fits; if the voice stutters in heavy scenes, add
   `--tts 0.6B`. Capping the game's FPS leaves the voice more GPU
   time. Games use the default microphone automatically; otherwise Windows Settings → System → Sound → Volume mixer → the
   game → Input device = «CABLE Output», or Steam → Settings → Voice → Recording device.

Your own Ukrainian voice is not heard there — only the English one. Checked with Discord and Telegram voice messages.

## If something does not work

- **`CUDA: False` at the end of `Install.bat`** — update the NVIDIA driver (it must support CUDA 12.8) and run it again.
- **Windows blocks the `.bat` files** (SmartScreen, "Windows protected your PC") — for a downloaded ZIP: right-click the
  ZIP → Properties → *Unblock* before unpacking, or *More info* → *Run anyway*.
- **It hears the wrong microphone / plays to the wrong device** — `Record-Samples.bat --list` shows the devices; set
  `VT_MIC` / `VT_OUT` in `.env` (part of the name is enough).
- **Out of GPU memory** — close games or other GPU programs, or use `--tts 0.6B`, `--asr-device cpu`.
- **The voice sounds cut or robotic in a call** — try turning off the app's noise suppression (Krisp in Discord), echo
  cancellation and automatic gain: they are made for live microphones, not for synthesized speech.

## Performance (RTX 4070 12 GB, Ryzen 5, measured)

| What | Result |
|---|---|
| Push-to-talk: key released → first English sound | ~0.6–0.7 s (quickmt), ~0.65 s (Hy-MT2-7B) |
| Simultaneous: start of a phrase → first English sound | ~2.5–3.5 s on live speech |
| Recognition errors on one speaker (40 phrases) | ~8% WER (Parakeet v3 fp32 on GPU / Parakeet Ultra on CPU) |
| CPU use while translating | ~2–3% (recognition and translation on the GPU) |
| In Path of Exile 2 (`Start-VoiceTranslate-Game.bat`) | first sound 0.6–1.0 s after release (1.7B voice; 0.6B: 0.55–1.3 s); game 80–90 FPS while speaking (130–145 without) |

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
| NVIDIA Parakeet-TDT-0.6B-v3, Moondream Parakeet Ultra, quickmt-uk-en | CC-BY-4.0 | anything, with attribution |
| Qwen3-TTS (Base, VoiceDesign), Hy-MT2-7B, sherpa-onnx, transformers | Apache-2.0 | anything, keep the notices |
| Silero VAD, llama.cpp, faster-qwen3-tts, CTranslate2, audiotsm | MIT | anything, keep the notice |
| pynput (keyboard hook) | LGPL-3.0 | used unmodified as a library |
| VB-CABLE (optional, not bundled) | donationware | free for personal use |

Details: [THIRD_PARTY.md](THIRD_PARTY.md). This project would not exist without the open licenses of these components.

## What is not solved yet

The simultaneous mode is a working prototype: the voice starts ~3 s after you, pieces are audibly joined, and errors of
the recognizer are spoken before they can be corrected. See [ROADMAP.md](ROADMAP.md) for what we measured and what we
would try next.
