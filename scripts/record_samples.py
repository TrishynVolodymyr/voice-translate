"""Records the user's voice references and Ukrainian test phrases into the voices folder (VT_VOICES_DIR).

Each prompt is shown on screen; Enter starts recording, Enter stops it. Existing files are skipped unless --redo is
given, so the session can be interrupted and resumed. Transcripts go to prompts.json in that folder (ICL cloning needs
the exact reference text).
"""
import argparse
import json
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import VOICES  # noqa: E402

OUT = VOICES
SR = 48000

# The Ukrainian reference is the main one: in xvec mode only the timbre is taken from it, and the English accent
# comes from the TTS model itself. The English reference is optional (--with-en), for comparison only.
REFS = [
    ("ref_uk", "Український зразок голосу. Читайте спокійно, своїм звичайним голосом:",
     "Привіт, це короткий запис мого звичайного голосу. Я так зазвичай говорю, коли граю з друзями або "
     "долучаюся до дзвінка. Іноді я говорю трохи швидше, іноді повільніше, але здебільшого звучу саме так. "
     "Мені подобаються ігри, де треба думати й домовлятися з командою, а не просто бігати й стріляти. "
     "Дякую, що слухаєте."),
]
REF_EN = ("ref_en", "Англійський зразок голосу (необов'язковий). Читайте спокійно, своїм звичайним голосом:",
          "Hi, this is a short recording of my normal speaking voice. I usually talk like this when I play games "
          "with friends or join a call. Sometimes I speak a little faster, sometimes slower, but this is how I sound "
          "most of the time. Thanks for listening.")

PHRASES = [
    "Привіт, усім чути мене?",
    "Зачекай секунду, я зараз повернуся.",
    "Хто йде зі мною на правий фланг?",
    "Обережно, за тим будинком ще двоє.",
    "У мене майже немає набоїв, хтось може поділитися?",
    "Давай спочатку зберемо ресурси, а потім підемо на боса.",
    "Я не встигаю, прикрийте мене, будь ласка.",
    "Нормально зіграли, але наступного разу треба тримати точку довше.",
    "Скинь мені посилання в чат, я гляну пізніше.",
    "Сьогодні я можу грати тільки до одинадцятої вечора.",
    "Доброго дня, я відповідаю за цю частину проєкту.",
    "Можете, будь ласка, повторити останнє питання?",
    "Я думаю, що ми встигнемо закінчити це до кінця тижня, якщо ніхто не захворіє.",
    "Нам треба обговорити бюджет, бо поточна оцінка виглядає занадто оптимістичною.",
    "Вибачте, у мене був поганий інтернет, я пропустив останні дві хвилини.",
    "Коротше, план такий: ти відволікаєш їх зліва, а ми заходимо з тилу і забираємо прапор.",
    "Я вчора пробував нову модель для синтезу мовлення, і вона звучить набагато природніше, ніж попередня.",
    "Якщо чесно, мені ця ідея не дуже подобається, але давайте спробуємо і подивимося, що вийде.",
    "Добре, тоді домовилися, завтра о сьомій збираємося в дискорді.",
    "Дякую всім, гарного вечора і до зустрічі.",
]


def record(path: Path, device) -> None:
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time, status):
        q.put(indata.copy())

    input("   Enter — почати запис... ")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=cb, device=device):
        input("   ● Запис іде. Enter — зупинити. ")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    audio = np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)
    sf.write(path, audio, SR, subtype="PCM_16")
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"   Збережено {path.name}: {audio.size / SR:.1f} с, пік {peak:.2f}"
          + ("  ← дуже тихо, можна перезаписати з --redo" if peak < 0.05 else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="перезаписати вже наявні файли")
    ap.add_argument("--device", default=None, help="номер або назва мікрофона (див. --list)")
    ap.add_argument("--list", action="store_true", help="показати аудіопристрої")
    ap.add_argument("--with-en", action="store_true", help="записати ще й англійський зразок (для порівняння)")
    ap.add_argument("--set", default=None,
                    help="записати лише 20 тестових фраз у теку <голоси>-<назва> (напр. --set cardioid) — для A/B мікрофона")
    a = ap.parse_args()
    global OUT
    if a.set:
        OUT = VOICES.parent / f"{VOICES.name}-{a.set}"
    if a.list:
        print(sd.query_devices())
        return
    device = int(a.device) if a.device and a.device.isdigit() else a.device
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Мікрофон: {sd.query_devices(device, 'input')['name']}\n")

    prompts = {REF_EN[0]: REF_EN[2]}
    items = [] if a.set else list(REFS) + ([REF_EN] if a.with_en else [])
    items += [(f"phrase_{i:02d}", f"Тестова фраза {i}/{len(PHRASES)} — скажіть природно, як у розмові:", t)
              for i, t in enumerate(PHRASES, 1)]
    for name, title, text in items:
        prompts[name] = text
        path = OUT / f"{name}.wav"
        if path.exists() and not a.redo:
            continue
        print(f"\n{title}\n\n   {text}\n")
        record(path, device)
    (OUT / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово. Файли в {OUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
