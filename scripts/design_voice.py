"""Creates a voice sample from a text description with Qwen3-TTS VoiceDesign — for people who do not want to clone
their own voice. The sample is saved into the voices folder (VT_VOICES_DIR) and then works exactly like a recorded one:
the translator takes only its timbre (xvec mode) and speaks the target language with it.

    Design-Voice.bat "A calm male voice in his thirties, low and slightly husky, speaking at a relaxed pace" --name calm
    Start-VoiceTranslate.bat --ref calm

The VoiceDesign model is downloaded with Download-Models.bat -VoiceDesign (~4.2 GB) and is needed only for this step.
Each run gives a slightly different voice; run it a few times and keep the one you like (--name)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import MODELS, VOICES, load_qwen_tts  # noqa: E402

MODEL = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# ~15 s of varied, natural English: enough for the voice embedding the translator takes from it
TEXT = ("Hi, this is a short recording of my normal speaking voice. I usually talk like this when I play games with "
        "friends or join a call. Sometimes I speak a little faster, sometimes slower, but this is how I sound most of "
        "the time. Thanks for listening.")


def main():
    ap = argparse.ArgumentParser(description="Створити голос за текстовим описом (Qwen3-TTS VoiceDesign)")
    ap.add_argument("description", help="опис голосу англійською: стать, вік, тембр, темп, манера")
    ap.add_argument("--name", default="designed", help="ім'я зразка (буде voices\\<ім'я>.wav; далі --ref <ім'я>)")
    ap.add_argument("--text", default=TEXT, help="що проказати у зразку (англійською)")
    a = ap.parse_args()
    if not (MODELS / MODEL).exists():
        sys.exit(f"Немає моделі {MODEL}. Завантажте: Download-Models.bat -VoiceDesign")

    print("Завантаження VoiceDesign…", flush=True)
    model = load_qwen_tts(MODEL)
    wavs, sr = model.generate_voice_design(text=a.text, instruct=a.description, language="English")
    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    VOICES.mkdir(parents=True, exist_ok=True)
    out = VOICES / f"{a.name}.wav"
    sf.write(out, audio, sr, subtype="PCM_16")
    prompts_file = VOICES / "prompts.json"  # the transcript, for the optional ICL mode (--mode icl)
    prompts = json.loads(prompts_file.read_text(encoding="utf-8")) if prompts_file.exists() else {}
    prompts[a.name] = a.text
    prompts_file.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {out} ({len(audio) / sr:.1f} с). Послухайте його; якщо подобається — запускайте з --ref {a.name}")


if __name__ == "__main__":
    main()
