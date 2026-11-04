# Third-party components

None of these are included in the repository: the installers download them from their authors (with SHA-256 checks)
into this folder. They keep their own licenses.

## Models

| Model | Author / source | License | Used for |
|---|---|---|---|
| Parakeet-TDT-0.6B-v3 (sherpa-onnx fp32 export `csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3`) | NVIDIA; export by the k2-fsa team | CC-BY-4.0 | speech recognition on the GPU |
| Parakeet Ultra (int8 export `chrmongeau/sherpa-onnx-nemo-parakeet-ultra-int8`) | Moondream, post-trained from NVIDIA Parakeet-TDT-0.6B-v3; community export | CC-BY-4.0 | speech recognition on the CPU |
| quickmt-uk-en (`quickmt/quickmt-uk-en`); other pairs `quickmt/quickmt-<from>-<to>` if your `.env` asks for them | quickmt | CC-BY-4.0 | fast translation (Ukrainian→English by default) |
| Hy-MT2-7B GGUF (`tencent/Hy-MT2-7B-GGUF`), optional | Tencent | Apache-2.0 | most accurate translation |
| Hy-MT2-1.8B GGUF (`tencent/Hy-MT2-1.8B-GGUF`), optional (`VT_MT=hymt1.8`) | Tencent | Apache-2.0 | lighter translation for pairs quickmt lacks |
| Qwen3-TTS-12Hz-1.7B-Base / 0.6B-Base (`Qwen/…`) | Alibaba Qwen | Apache-2.0 | speech in a cloned voice (10 languages) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign (`Qwen/…`), optional | Alibaba Qwen | Apache-2.0 | creating a voice from a description |
| Silero VAD (`silero_vad.onnx` from the sherpa-onnx releases) | Silero | MIT | detecting speech in simultaneous mode |

CC-BY-4.0 requires attribution: «Parakeet-TDT-0.6B-v3 by NVIDIA», «Parakeet Ultra by Moondream», «quickmt-uk-en by
quickmt» (and any other quickmt pair you download), each under CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>).

## Programs and libraries

| Component | License |
|---|---|
| llama.cpp b12040 (ggml-org), optional — runs Hy-MT2 and other GGUF translators | MIT |
| uv (Astral) — installs Python and packages | MIT / Apache-2.0 |
| Python 3.11 | PSF License |
| PyTorch, torchaudio | BSD-3-Clause |
| transformers, huggingface-hub, sentencepiece, qwen-tts-hf | Apache-2.0 |
| faster-qwen3-tts | MIT |
| sherpa-onnx (k2-fsa), onnxruntime | Apache-2.0 / MIT |
| CTranslate2 | MIT |
| sounddevice, audiotsm | MIT |
| soundfile, NumPy, SciPy | BSD |
| pynput — keyboard hook, used unmodified as a library | LGPL-3.0 |
| VB-CABLE (VB-Audio), optional, installed by the user | donationware, free for personal use |
