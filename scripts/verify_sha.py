"""Checks SHA-256 of every downloaded LFS file in models\\ against Hugging Face metadata."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import MODELS  # noqa: E402

# (folder in models\, Hugging Face repository); one folder may hold files of several repositories (hy-mt2)
REPOS = [
    ("parakeet-tdt-0.6b-v3-int8", "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"),
    ("parakeet-tdt-0.6b-v3-fp32", "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3"),
    ("parakeet-ultra-int8", "chrmongeau/sherpa-onnx-nemo-parakeet-ultra-int8"),
    ("nemo-fastconformer-ctc-uk", "NeonIH/sherpa-onnx-nemo-fast-conformer-ctc-uk"),
    ("Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    ("Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("hy-mt2", "tencent/Hy-MT2-7B-GGUF"),
    ("hy-mt2", "tencent/Hy-MT2-1.8B-GGUF"),
]
# quickmt: one model per language pair (VT_SOURCE_LANG-VT_TARGET_LANG), the folder is named as the repository
REPOS += [(p.name, f"quickmt/{p.name}") for p in sorted(MODELS.glob("quickmt-*")) if p.is_dir()]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


bad = 0
for local, repo in REPOS:
    if not (MODELS / local).exists():  # optional model not downloaded
        continue
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true"))
    for s in meta["siblings"]:
        lfs = s.get("lfs")
        p = MODELS / local / s["rfilename"]
        if not lfs or not p.exists():
            continue
        ok = sha256(p) == lfs["sha256"]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {local}/{s['rfilename']}")
sys.exit(1 if bad else 0)
